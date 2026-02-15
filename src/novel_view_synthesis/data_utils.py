import cv2
import numpy as np
from PIL import Image
import torch
import re, os, sys
from collections import Counter
import random
from pathlib import Path
from typing import List, Union


def meshgrid(n_batch, n_height, n_width, device, homogeneous=True):
    '''
    Creates N x 2 x H x W meshgrid in x, y directions
    Arg(s):
        n_batch : int
            batch size
        n_height : int
            height of tensor
        n_width : int
            width of tensor
        device : torch.device
            device on which to create meshgrid
        homoegenous : bool
            if set, then add homogeneous coordinates (N x H x W x 3)
    Return:
        torch.Tensor[float32]: N x 2 x H x W meshgrid of x, y and 1 (if homogeneous)
    '''

    x = torch.linspace(start=0.0, end=n_width-1, steps=n_width, device=device)
    y = torch.linspace(start=0.0, end=n_height-1, steps=n_height, device=device)

    # Create H x W grids
    grid_y, grid_x = torch.meshgrid(y, x)

    if homogeneous:
        # Create 3 x H x W grid (x, y, 1)
        grid_xy = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
    else:
        # Create 2 x H x W grid (x, y)
        grid_xy = torch.stack([grid_x, grid_y], dim=0)

    grid_xy = torch.unsqueeze(grid_xy, dim=0) \
        .repeat(n_batch, 1, 1, 1)

    return grid_xy

def backproject_to_camera(depth, intrinsics, shape):
    '''
    Backprojects pixel coordinates to 3D camera coordinates
    Arg(s):
        depth : torch.Tensor[float32]
            N x 1 x H x W depth map
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 4 x (H x W)
    '''
    n_batch, _, n_height, n_width = shape

    # Create homogeneous coordinates [x, y, 1]
    xy_h = meshgrid(n_batch, n_height, n_width, device=depth.device, homogeneous=True)

    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = xy_h.view(n_batch, 3, -1)

    # Reshape depth as N x 1 x (H x W)
    depth = depth.view(n_batch, 1, -1)

    # K^-1 [x, y, 1] z
    points = torch.matmul(torch.inverse(intrinsics), xy_h) * depth

    # Make homogeneous
    return torch.cat([points, torch.ones_like(depth)], dim=1)

def project_to_pixel(points, pose, intrinsics, shape):
    '''
    Projects points in camera coordinates to 2D pixel coordinates
    Arg(s):
        points : torch.Tensor[float32]
            N x 4 x (H x W) depth map
        pose : torch.Tensor[float32]
            N x 4 x 4 transformation matrix
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 2 x H x W
    '''

    n_batch, _, n_height, n_width = shape

    # Convert camera intrinsics to homogeneous coordinates
    column = torch.zeros([n_batch, 3, 1], device=points.device)
    row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=points.device) \
        .view(1, 1, 4) \
        .repeat(n_batch, 1, 1)
    intrinsics = torch.cat([intrinsics, column], dim=2)
    intrinsics = torch.cat([intrinsics, row], dim=1)

    # Apply the transformation and project: \pi K g p
    T = torch.matmul(intrinsics, pose)
    T = T[:, 0:3, :]
    points = torch.matmul(T, points)
    points = points / (torch.unsqueeze(points[:, 2, :], dim=1) + 1e-7)
    points = points[:, 0:2, :]

    # Reshape to N x 2 x H x W
    return points.view(n_batch, 2, n_height, n_width)

def grid_sample(image, target_xy, shape, padding_mode='zeros'):
    '''
    Samples the image at x, y locations to target x, y locations
    Arg(s):
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        target_xy : torch.Tensor[float32]
            N x 2 x H x W target x, y locations in image space
        shape : list[int]
            shape of tensor in (N, C, H, W)
        padding_mode : str
            padding to use when sampled out of bounds
    Return:
        torch.Tensor[float32] : N x 3 x H x W RGB image
    '''

    n_batch, _, n_height, n_width = shape

    # Swap dimensions to N x H x W x 2 for grid sample
    target_xy = target_xy.permute(0, 2, 3, 1)
    # Normalize coordinates between -1 and 1
    target_xy[..., 0] /= (n_width - 1.0)
    target_xy[..., 1] /= (n_height - 1.0)
    target_xy = 2.0 * (target_xy - 0.5)

    # Sample the image at normalized target x, y locations
    return torch.nn.functional.grid_sample(
        image,
        grid=target_xy,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)

def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip()

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def return_paths(root_dir):
    """
    Recursively list all image file paths under root_dir.

    Returns:
        List of absolute paths (as strings), sorted.
    """
    root = Path(root_dir).expanduser().resolve()

    return sorted(str(p) for p in root.rglob("*.png") if p.is_file())

def load_image(path, normalize=True, data_format='HWC'):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    image = Image.open(path).convert('RGB')

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    # Normalize
    image = image / 255.0 if normalize else image

    return image

def save_image(image, path, normalized=True, data_type='color', data_format='HWC'):
    '''
    Saves an RGB, gray, or label (with validity map in alpha) image to 8-bit PNG

    Arg(s):
        image : numpy
            RGB, gray, or label (with validity map in alpha) image
        path : str
            path to store image
        normalized : bool
            if set, then treat image as normalized range [0, 1] and multiply by 255
        data_type : str
            color or label
        data_format : str
            data format of input 'CHW', or 'HWC'
    '''

    # Put image between [0, 255] range, if it was normalized to [0, 1]
    image = 255.0 * image if normalized else image
    image = np.uint8(image)

    if image.ndim == 2:
        # Append channel dimension
        image = np.expand_dims(image, axis=-1)

    # Put image into H x W x C format
    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if data_type == 'color':
        image = Image.fromarray(np.uint8(image))
    elif data_type == 'gray':
        image = np.squeeze(image)
        image = Image.fromarray(np.uint8(image), mode='L')
    elif data_type == 'label':
        # Add alpha channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image = np.concatenate([image, 255.0 * np.ones_like(image)], axis=-1)
        elif image.ndim == 3 and image.shape[-1] > 2:
            raise ValueError('ERROR: too many channels in label')

        image = Image.fromarray(np.uint8(image), mode='LA')
    else:
        raise ValueError('Unsupported data type: {}'.format(data_type))

    image.save(path)

def load_depth_with_validity_map(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map and validity map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
        numpy[float32] : binary validity map for available depth measurement locations
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0
    v = z.astype(np.float32)
    v[z > 0] = 1.0

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
        v = np.expand_dims(v, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
        v = np.expand_dims(v, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z, v

def load_depth(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def save_depth(z, path, multiplier=256.0):
    '''
    Saves a depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
        multiplier : float
            multiplier for encoding float as unsigned integer
    '''

    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(path)

def load_validity_map(path, ignore_empty=False, data_format='HW'):
    '''
    Loads a validity map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : binary validity map for available depth measurement locations
    '''

    # Loads validity map from 16-bit PNG file
    v = np.array(Image.open(path), dtype=np.float32)

    if ignore_empty:
        values = np.unique(v)
        assert np.all(values == [0, 256]) or np.all(values == [0]) or np.all(values == [256])
    else:
        assert np.all(np.unique(v) == [0, 256])

    v[v > 0] = 1

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        v = np.expand_dims(v, axis=0)
    elif data_format == 'HWC':
        v = np.expand_dims(v, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return v

def save_validity_map(v, path):
    '''
    Saves a validity map to a 16-bit PNG file

    Arg(s):
        v : numpy[float32]
            validity map
        path : str
            path to store validity map
    '''

    v[v <= 0] = 0.0
    v[v > 0] = 1.0
    v = np.uint32(v * 256.0)
    v = Image.fromarray(v, mode='I')
    v.save(path)

def load_exr(path, data_format='HW'):
    '''
    Loads an exr image

    Arg(s):
        path : str
            path to exr file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : exr depth image
    '''

    z = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    z = np.array(z, dtype=np.float32)

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def load_calibration(path):
    '''
    Loads the calibration matrices for each camera (KITTI) and stores it as map

    Arg(s):
        path : str
            path to file to be read
    Returns:
        dict[str, float] : map containing camera intrinsics keyed by camera id
    '''

    float_chars = set("0123456789.e+- ")
    data = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.asarray(
                        [float(x) for x in value.split(' ')])
                except ValueError:
                    pass
    return data

def read_pfm(path):
    """
    Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale

def write_pfm(path, image, scale=1):
    """
    Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)

def load_velodyne(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # Homogeneous
    return points

def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def velodyne2depth(calibration_dir, velodyne_path, shape, camera_id=2):
    # Load calibration files
    cam2cam = load_calibration(os.path.join(calibration_dir, 'calib_cam_to_cam.txt'))
    velo2cam = load_calibration(os.path.join(calibration_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # Compute projection matrix from velodyne to image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(camera_id)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # Load velodyne points and remove all that are behind image plane (approximation)
    # Each row of the velodyne data refers to forward, left, up, reflectance
    velo = load_velodyne(velodyne_path)
    velo = velo[velo[:, 0] >= 0, :]

    # Project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]
    velo_pts_im[:, 2] = velo[:, 0]

    # Check if in bounds (use minus 1 to get the exact same value as KITTI matlab code)
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0])-1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1])-1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < shape[1]) & (velo_pts_im[:, 1] < shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # Project to image
    depth = np.zeros(shape)
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # Find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    # Clip all depth values less than 0 to 0
    depth[depth < 0] = 0
    return depth.astype(np.float32)

def load_text(text_path, dataset, select_random=True):
    '''
    Args:
        path (str): path to text file
        dataset (str): dataset name
        select_random (bool): flag to select a random line

    Returns:
        list: list of words
    '''
    assert text_path.lower().endswith('.txt'), (
        f"load_text expects a .txt file but got {text_path}"
    )

    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        raise RuntimeError(f"Could not decode {text_path} as UTF‑8")

    if select_random:
        line = random.choice(lines).strip()
    else:
        line = lines[0]

    if 'rsa' in dataset:
        return [line]
    else:
        text_list = line.split(',')

        return text_list


def load_mask(mask_path, target_shape):
    """
    Load a mask image and return a boolean array of shape target_shape.
    Non-zero pixels are considered True.
    """
    img = Image.open(mask_path)
    mask = np.array(img)

    # If mask has channels, collapse to one by taking first channel
    if mask.ndim == 3:
        mask = mask[..., 0]

    if mask.shape != target_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match depth shape {target_shape} for {mask_path}"
        )

    # Non-zero = True
    return mask.astype(np.float32) > 0.5