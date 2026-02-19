import os, sys
import numpy as np
import cv2, torch
sys.path.insert(0, os.path.join('external_src', 'monocular_depth', 'Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2BaseModel
from run_helper_functions import resize_image


class DepthAnythingV2Model(torch.nn.Module):
    '''
    Class for interfacing with DepthAnything V2 model

    Arg(s):
        device : torch.device
            device to run model on
        encoder : string
            type of depthanything v2 model to build
        use_pretrained : bool
            if set, then configure using legacy settings
    '''
    def __init__(self, device=torch.device('cuda'), encoder='vitl', use_pretrained=True):
        super().__init__()
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.model = DepthAnythingV2BaseModel(**model_configs[encoder])

        if use_pretrained:
            ckpt_path = os.path.join('model_ckpts', 'depthanything_v2', f'depth_anything_v2_{encoder}.pth')
            self.model.load_state_dict(torch.load(ckpt_path))

        self.device = device
        self.to(self.device)

    def forward(self, image, intrinsics=None):
        '''
        Forwards inputs through the network

        Arg(s):
            image : tensor[float32]
                N x 3 x H x W image
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''
        images = image
        if torch.max(images) > 1.0:
            images = images / 255.0

        images_batch = []

        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        for b, img in enumerate(images):
            img = img.unsqueeze(0)
            image_input = resize_image(
                img, width=518, height=518, keep_aspect_ratio=True,
                resize_method='lower_bound', ensure_multiple_of=14,
                interpolation_method=cv2.INTER_CUBIC)

            normalized_image = (image_input.cpu().numpy() - mean) / std
            prepared_image = np.ascontiguousarray(normalized_image).astype(np.float32)
            image_input = torch.from_numpy(prepared_image).squeeze(0)
            images_batch.append(image_input)

        image_input = torch.stack(images_batch, dim=0).to(self.device)

        output = self.model(image_input)

        output = torch.nn.functional.interpolate(
            output.unsqueeze(1),  # output[None],
            image.shape[2:],
            mode='bilinear',
            align_corners=True)  # [0, 0]

        # -- NOTE: DepthAnything returns inverse depth ----
        # epsilon = 1
        # output = 1.0 / (output + epsilon)

        output = (output - output.min()) / (output.max() - output.min())

        output[output < 1e-8] = 1e-8
        output = 1.0 / output

        return output

    def compute_loss(self, output_depth, ground_truth_depth, l1_weight=1.0, l2_weight=1.0):
        '''

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 2 x H x W ground_truth depth and ground truth validity map
            l1_weight : float
                weight of l1 loss
            l2_weight : float
                weight of l2 loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''
        l1_loss = torch.abs(output_depth - ground_truth_depth[:, 0:1])
        l2_loss = (output_depth - ground_truth_depth[:, 0:1]) ** 2

        loss = l1_weight * l1_loss.mean() + l2_weight * l2_loss.mean()
        loss_info = {
            "l1_loss": l1_loss.mean().item(),
            "l2_loss": l2_loss.mean().item(),
            "total_loss": loss.item()
        }
        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = []
        parameters = torch.nn.ParameterList(self.model.parameters())

        return parameters

    def train(self, mode=True):
        '''
        Sets model to training mode
        '''

        super().train(mode)
        self.model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model = torch.nn.DataParallel(self.model)

    def set_device(self, rank):
        self.model.module.set_device(rank)

    def distributed_data_parallel(self, rank):
        '''
        Allows multi-gpu split along batch
        '''
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank],
                                                               find_unused_parameters=True)

    def restore_model(self, restore_path, optimizer=None, learning_schedule=None, learning_rates=None,
                      n_step_per_epoch=None):
        '''
        Loads weights from checkpoint and loads and returns optimizer

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            torch.optimizer if optimizer is passed in
        '''
        # TODO: Implement code to restore model
        return

    def save_model(self, checkpoint_path, step, optimizer, meanvar=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''
        # TODO: Implement code to save model
        return
