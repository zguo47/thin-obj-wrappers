import torch
import numpy as np
import data_utils


def random_sample(T):
    '''
    Arg(s):
        T : numpy[float32]
            C x N array
    Returns:
        numpy[float32] : random sample from T
    '''

    index = np.random.randint(0, T.shape[0])
    return T[index, :]

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    if len(inputs) > 1:
        assert inputs[0].shape[1] == inputs[1].shape[1] and inputs[0].shape[2] == inputs[1].shape[2]

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape  # Expect C x H x W

    # Get delta of crop and original height and width

    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    # If left alignment, then set starting height to 0
    if 'left' in crop_type:
        x_start = 0

    # If right alignment, then set starting height to right most position
    elif 'right' in crop_type:
        x_start = d_width

    elif 'horizontal' in crop_type:

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

    # If top alignment, then set starting height to 0
    if 'top' in crop_type:
        y_start = 0

    # If bottom alignment, then set starting height to lowest position
    elif 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    elif 'center' in crop_type:
        pass

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width

    outputs = []
    for T in inputs:
        if len(T.shape) == 3:
            # HACK BECAUSE SOME IMAGES GOT SAWED AND SOME DIDN'T
            if T.shape[2] > 2000:
                temp_start, temp_end = x_start + o_width, x_end + o_width
            else:
                temp_start, temp_end = x_start, x_end
            output = T[:, y_start:y_end, temp_start:temp_end] if T.size != 0 else np.array([])
            # print(o_width, x_start, x_end, T.shape, output.shape, T.size)
        elif len(T.shape) == 2:
            output = T[y_start:y_end, x_start:x_end] if T.size != 0 else np.array([])
        outputs.append(output)

    # Adjust intrinsics
    if intrinsics is not None:
        offset_principal_point = np.array([
            [0.0, 0.0, -x_start],
            [0.0, 0.0, -y_start],
            [0.0, 0.0,  0.0]
        ], dtype=np.float32)

        intrinsics = [
            in_ + offset_principal_point for in_ in intrinsics
        ]

        return outputs, intrinsics
    else:
        return outputs


class MonocularDepthEstimationDataset(torch.utils.data.Dataset):
    '''
    Returns image in style of output format

    Arg(s):

    '''
    def __init__(self,
                 dataset_name,
                 image_paths,
                 intrinsics_paths,
                 ground_truth_paths,
                 output_format=None,
                 shape=None,
                 random_crop_shape=None,
                 random_crop_type=None):
        self.dataset_name = dataset_name
        self.image_paths = image_paths

        self.n_sample = len(image_paths)

        if ground_truth_paths is not None:
            self.ground_truth_paths = ground_truth_paths
            self.output_format = output_format
        else:
            self.ground_truth_paths = [None] * self.n_sample

        if intrinsics_paths is not None:
            self.intrinsics_paths = intrinsics_paths
        else:
            self.intrinsics_paths = [None] * self.n_sample

        assert self.n_sample == len(self.ground_truth_paths)

        self.shape = shape

        # Shape is not None and it does not contains (None, None)
        self.do_resize = self.shape is not None and None not in self.shape

        self.random_crop_type = random_crop_type
        self.random_crop_shape = random_crop_shape
        self.do_random_crop = \
            not self.do_resize and random_crop_shape is not None and all([x > 0 for x in random_crop_shape])

        self.data_format = 'CHW'

    def __getitem__(self, index):

        image = data_utils.load_image(
            self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        if self.dataset_name == 'kitti':
            _, height, width = image.shape
            top = height - 352
            left = (width - 1216) // 2
            image = image[:, top : top + 352, left : left + 1216]

        inputs = [image]

        if self.ground_truth_paths[index] is not None:
            if 'bts' in self.dataset_name:
                multiplier = 1000.0
            else:
                multiplier = 256.0

            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                multiplier,
                data_format=self.data_format)

            if self.dataset_name == 'kitti':
                _, height, width = ground_truth.shape
                top = height - 352
                left = (width - 1216) // 2
                ground_truth = ground_truth[:, top : top + 352, left : left + 1216]

            inputs.append(ground_truth)

        if self.intrinsics_paths[index] is not None:
            intrinsics = np.load(self.intrinsics_paths[index])
        else:
            intrinsics = np.eye(3)

        if self.do_random_crop:
            inputs, intrinsics = random_crop(
                inputs,
                shape=self.random_crop_shape,
                intrinsics=[intrinsics],
                crop_type=self.random_crop_type)

        inputs.append(intrinsics)

        # Convert to float32
        inputs = [T.astype(np.float32) for T in inputs]

        return inputs

    def __len__(self):
        '''
        Returns the number of elements in dataset

        Returns:
            int : number of elements in dataset
        '''

        return self.n_sample
