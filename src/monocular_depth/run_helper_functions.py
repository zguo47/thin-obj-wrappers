import torch, torchvision
import torchvision.transforms.functional as functional
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def crop_inputs(inputs, shape, intrinsics=None, crop_type=['bottom']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[torch.Tensor]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : tensor[float32]
            3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[torch.Tensor] : list of cropped inputs
        tensor[float32] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, :, y_start:y_end, x_start:x_end] for T in inputs
    ]

    if intrinsics is not None:
        # Adjust intrinsics
        intrinsics = intrinsics + torch.tensor([[0.0, 0.0, -x_start],
                                   [0.0, 0.0, -y_start],
                                   [0.0, 0.0, 0.0     ]]).cuda()

        return outputs, intrinsics
    else:
        return outputs


def resize_to_scale(tensors, intrinsics, resize_shape):
    '''
    Resize images, ground truth, spare depth, and adjust intrinsics accordingly

    Arg(s):
        tensors : list[torch.Tensor]
            list of N x C x H x W tensors
        intrinsics : tensor.Tensor
            1 x 3 x 3 intrinsics matrix
        resize_shape : list[int, int]
            height and width to resize
    Returns:
        list[torch.Tensor] : list of transformed N x C x H x W image tensors
        torch.Tensor : adjusted 1 x 3 x 3 intrinsics matrix
        list[tuple] : list of original height and width for each tensor
    '''
    resize_height, resize_width = resize_shape
    resized_tensors = []
    original_shapes = []

    for tensor in tensors:
        # Get original dimensions
        n, c, h, w = tensor.shape
        original_shapes.append((h, w))

        # Calculate the scale for resizing
        scale_x = resize_width / w
        scale_y = resize_height / h

        # Resize the tensor
        resized_tensor = functional.resize(tensor, size=[resize_height, resize_width])
        resized_tensors.append(resized_tensor)

    if intrinsics is not None:
        # Ensure intrinsics is of shape [1, 3, 3]
        if intrinsics.shape != (1, 3, 3):
            raise ValueError("Intrinsics matrix must be of shape [1, 3, 3]")

        # Extract the single 3x3 intrinsics matrix
        intrinsics_matrix = intrinsics[0]

        # Adjust intrinsics matrix
        intrinsics_matrix[0, 0] *= scale_x  # fx
        intrinsics_matrix[1, 1] *= scale_y  # fy
        intrinsics_matrix[0, 2] *= scale_x  # cx
        intrinsics_matrix[1, 2] *= scale_y  # cy

        # Convert back to [1, 3, 3] shape
        intrinsics = intrinsics_matrix.unsqueeze(0)

    return resized_tensors, intrinsics, original_shapes

def undo_resize(tensors, intrinsics, original_shapes):
    '''
    Undo resize images

    Arg(s):
        tensors : list[torch.Tensor]
            list of N x C x H x W tensors
        intrinsics : torch.Tensor
            1 x 3 x 3 intrinsics matrix
        original_shapes : list[tuple]
            list of original height and width for each tensor
    Returns:
        list[torch.Tensor] : list of restored N x C x H x W image tensors
        torch.Tensor : adjusted 1 x 3 x 3 intrinsics matrix
    '''
    restored_tensors = []

    for tensor, (original_height, original_width) in zip(tensors, original_shapes):
        # Get current dimensions
        n, c, h, w = tensor.shape

        # Calculate the scale for restoring
        scale_x = original_width / w
        scale_y = original_height / h

        # Resize the tensor back to the original dimensions
        restored_tensor = functional.resize(tensor, [original_height, original_width])
        restored_tensors.append(restored_tensor)

    if intrinsics is not None:
        # Ensure intrinsics is of shape [1, 3, 3]
        if intrinsics.shape != (1, 3, 3):
            raise ValueError("Intrinsics matrix must be of shape [1, 3, 3]")

        # Extract the single 3x3 intrinsics matrix
        intrinsics_matrix = intrinsics[0]

        # Adjust intrinsics matrix back
        intrinsics_matrix[0, 0] /= scale_x  # fx
        intrinsics_matrix[1, 1] /= scale_y  # fy
        intrinsics_matrix[0, 2] /= scale_x  # cx
        intrinsics_matrix[1, 2] /= scale_y  # cy

        # Convert back to [1, 3, 3] shape
        intrinsics = intrinsics_matrix.unsqueeze(0)

    return restored_tensors, intrinsics

def resize_and_crop(images_arr,
                    do_resize_and_crop,
                    resize_shape,
                    start_yx,
                    end_yx,
                    interpolation_modes=[Image.NEAREST]):
    '''
    Resize and crop to shape

    Arg(s):
        images_arr : list[torch.Tensor]
            list of N x C x H x W tensors
        do_resize_and_crop : bool
            N booleans to determine if image will be resized and crop based on input (y, x)
        resize_shape : list[int, int]
            height and width to resize
        start_yx : list[int, int]
            top left corner y, x coordinate
        end_yx : list
            bottom right corner y, x coordinate
        interpolation_modes : list[int]
            list of enums for interpolation: Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.ANTIALIAS
    Returns:
        list[torch.Tensor] : list of transformed N x C x H x W image tensors
    '''

    n_images_arr = len(images_arr)

    if len(interpolation_modes) < n_images_arr:
        interpolation_modes = \
            interpolation_modes + [interpolation_modes[-1]] * (n_images_arr - len(interpolation_modes))

    for i, (images, interpolation_mode) in enumerate(zip(images_arr, interpolation_modes)):

        images_cropped = []

        for b, image in enumerate(images):
            if do_resize_and_crop[b]:

                r_height = resize_shape[0][b]
                r_width = resize_shape[1][b]

                # Resize image
                image = functional.resize(
                    image,
                    size=(r_height, r_width),
                    interpolation=interpolation_mode)

                start_y = start_yx[0][b]
                start_x = start_yx[1][b]
                end_y = end_yx[0][b]
                end_x = end_yx[1][b]

                # Crop image
                image = image[..., start_y:end_y, start_x:end_x]

                images_cropped.append(image)
            else:
                images_cropped.append(image)

        images_arr[i] = torch.stack(images_cropped, dim=0)

    return images_arr

def resize_image(tensor, width, height, keep_aspect_ratio=True, resize_method='lower_bound', ensure_multiple_of=1, interpolation_method=cv2.INTER_AREA):
    """
    Resize image tensor.

    Args:
        tensor : torch.Tensor
            N x C x H x W image tensor
        width : int
            desired output width
        height : int
            desired output height
        keep_aspect_ratio : bool
            whether to keep the aspect ratio
        resize_method : str
            resize method: 'lower_bound', 'upper_bound', 'minimal'
        ensure_multiple_of : int
            output width and height are constrained to be multiples of this value
        interpolation_method : str
            interpolation method: 'nearest', 'bilinear', etc.
    Returns:
        torch.Tensor : resized image tensor
        tuple : original height and width of the image tensor
    """
    # Get original dimensions
    _, _, h, w = tensor.shape

    # Calculate the scale for resizing
    scale_height = height / h
    scale_width = width / w

    if keep_aspect_ratio:
        if resize_method == "lower_bound":
            if scale_width > scale_height:
                scale_height = scale_width
            else:
                scale_width = scale_height
        elif resize_method == "upper_bound":
            if scale_width < scale_height:
                scale_height = scale_width
            else:
                scale_width = scale_height
        elif resize_method == "minimal":
            if abs(1 - scale_width) < abs(1 - scale_height):
                scale_height = scale_width
            else:
                scale_width = scale_height
        else:
            raise ValueError(f"resize_method {resize_method} not implemented")

    new_height = int(np.round(scale_height * h / ensure_multiple_of) * ensure_multiple_of)
    new_width = int(np.round(scale_width * w / ensure_multiple_of) * ensure_multiple_of)

    if new_height < 0:
        new_height = int(np.ceil(scale_height * h / ensure_multiple_of) * ensure_multiple_of)
    if new_width < 0:
        new_width = int(np.ceil(scale_width * w / ensure_multiple_of) * ensure_multiple_of)

    # Convert tensor to NumPy array
    np_tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Resize using OpenCV
    resized_np_tensor = cv2.resize(np_tensor, (new_width, new_height), interpolation=interpolation_method)

    # Convert back to PyTorch tensor
    resized_tensor = torch.from_numpy(resized_np_tensor).permute(2, 0, 1).unsqueeze(0)

    return resized_tensor

def write_depth(path, depth, grayscale, bits=1):
    """Write depth map to png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
        grayscale (bool): use a grayscale colormap?
    """
    if not grayscale:
        bits = 1

    if not np.isfinite(depth).all():
        depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        print("WARNING: Non-finite depth values present")

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    if not grayscale:
        out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

    if bits == 1:
        cv2.imwrite(path, out.astype("uint8"))
    elif bits == 2:
        cv2.imwrite(path, out.astype("uint16"))

    return

def visualize_error_plt(image, ground_truth, relative_depth, global_scale, shift_matrix, mask, save_path):
    '''
    Visualize error between relative depth, gs depth, output depth and ground truth
    Args:
        image : tensor[float32]
            N x C x H x W image tensor
        ground_truth : ndarray[float32]
            H x W ground truth numpy array
        
    '''
    if torch.is_tensor(image):
        image = image.squeeze().cpu().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.squeeze().cpu().numpy()
    if torch.is_tensor(relative_depth):
        relative_depth = relative_depth.squeeze().cpu().numpy()
    if torch.is_tensor(global_scale):
        global_scale = global_scale.squeeze().cpu().numpy()
    if torch.is_tensor(shift_matrix):
        shift_matrix = shift_matrix.squeeze().cpu().numpy()

    _, H, W = image.shape

    assert ground_truth.shape == (H, W), "Ground truth shape mismatch"
    assert relative_depth.shape == (H, W), "Relative depth shape mismatch"
    assert global_scale.shape == (H, W), "Global scale shape mismatch"
    assert shift_matrix.shape == (H, W), "Shift matrix shape mismatch"

    output_depth = relative_depth * global_scale + shift_matrix
    global_scale_output_depth = relative_depth * global_scale

    # Normalize image for display
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)

    relative_error_map = relative_depth - ground_truth
    error_map = output_depth - ground_truth
    gs_error_map = global_scale_output_depth - ground_truth

    mask_value = -9999
    masked_relative_error_map = np.full(relative_error_map.shape, mask_value)
    masked_relative_error_map[mask] = relative_error_map[mask]

    masked_error_map = np.full(error_map.shape, mask_value)
    masked_error_map[mask] = error_map[mask]

    masked_gs_error_map = np.full(gs_error_map.shape, mask_value)
    masked_gs_error_map[mask] = gs_error_map[mask]

    # print(">>>> mean of error map:", np.mean(relative_error_map[mask]), " std:", np.std(relative_error_map[mask]))
    # print(">>>> max of error map:", np.max(relative_error_map[mask]), " min:", np.min(relative_error_map[mask]))

    # max_range = np.max(np.abs(relative_error_map[mask])) * 0.4
    max_range = np.std(relative_error_map[mask])
    min_range = -max_range

    # print(">>>> Max range:", max_range, " Min range:", min_range)

    # RMSE calculations
    mde_rmse = np.sqrt(np.mean((relative_depth[mask] - ground_truth[mask]) ** 2))
    gs_rmse = np.sqrt(np.mean((global_scale_output_depth[mask] - ground_truth[mask]) ** 2))
    out_rmse = np.sqrt(np.mean((output_depth[mask] - ground_truth[mask]) ** 2))

    # print(f"RMSE of MDE: {mde_rmse:.2f}, GS: {gs_rmse:.2f}, Output: {out_rmse:.2f}")

    # Create a custom colormap: black for mask_value, bwr for error
    bwr = plt.get_cmap('bwr')
    # 256 colors, first color is black
    colors = bwr(np.linspace(0, 1, 255))
    colors = np.vstack((np.array([0, 0, 0, 1]), colors)) 
    custom_cmap = mcolors.ListedColormap(colors)
    # Set norm so mask_value maps to 0, error range maps to 1-255
    norm = mcolors.Normalize(vmin=mask_value, vmax=max_range)

    fig, axs = plt.subplots(7, 1, figsize=(16, 32), gridspec_kw={'hspace': 0})  

    plt.subplots_adjust(
    left=0, right=1, top=1, bottom=0,
    hspace=0, wspace=0
    )

    axs[0].imshow(image)
    axs[0].set_title('IMG, MDE, MDE ERR, GS, GS ERR, OUT, OUT ERR', fontsize=10)
    axs[0].axis('off')

    rel = axs[1].imshow(relative_depth, cmap='viridis')
    axs[1].axis('off')
    axs[1].text(0.01, 0.99, f'RMSE: {mde_rmse:.2f}', color='white', fontsize=16,
                transform=axs[1].transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='black', alpha=0.5))

    rel_err = axs[2].imshow(masked_relative_error_map, cmap=custom_cmap, vmin=min_range, vmax=max_range)
    axs[2].axis('off')

    gs_out = axs[3].imshow(global_scale_output_depth, cmap='viridis')
    axs[3].axis('off')
    axs[3].text(0.01, 0.99, f'RMSE: {gs_rmse:.2f}', color='white', fontsize=16,
                transform=axs[3].transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='black', alpha=0.5))

    gs_out_err = axs[4].imshow(masked_gs_error_map, cmap=custom_cmap, vmin=min_range, vmax=max_range)
    axs[4].axis('off')

    out = axs[5].imshow(output_depth, cmap='viridis')
    axs[5].axis('off')
    axs[5].text(0.01, 0.99, f'RMSE: {out_rmse:.2f}', color='white', fontsize=16,
                transform=axs[5].transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='black', alpha=0.5))

    out_err = axs[6].imshow(masked_error_map, cmap=custom_cmap, vmin=min_range, vmax=max_range)
    axs[6].axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
