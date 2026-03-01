import os, sys
import numpy as np
import cv2, torch

from datetime import datetime, timedelta
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from typing import List, Union
from PIL import Image

sys.path.insert(0, os.path.join('external_src', 'monocular_depth', 'Marigold'))
from marigold.marigold_depth_pipeline import MarigoldDepthPipeline, MarigoldDepthOutput

from run_helper_functions import resize_image

# NOTE: This model is implemented for inference ONLY!!!
class MarigoldModel(torch.nn.Module):
    '''
    Class for interfacing with Marigold model

    Arg(s):
        device : torch.device
            device to run model on
        encoder : string
            type of marigold model to build
        use_pretrained : bool
            if set, then configure using legacy settings
    '''
    def __init__(self, denoise_steps, ensemble_size,processing_res, match_input_res, batch_size, color_map, show_progress_bar=False, resample_method='bilinear', seed=1234, device=torch.device('cuda'), variant="fp16"):
        super().__init__()

        if variant == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        ckpt_path = os.path.join('model_ckpts', 'marigold', 'marigold-depth-v1-1')
        self.model : MarigoldDepthPipeline = MarigoldDepthPipeline.from_pretrained(ckpt_path, variant=variant, torch_dtype=dtype)

        # TODO: these are used when model is called. Check if I should just use the yaml/any of these fields require special handling or init
        # or put them into a dict/subclass for better organization
        self.denoise_steps = denoise_steps
        self.ensemble_size = ensemble_size
        self.processing_res = processing_res
        self.match_input_res = match_input_res
        self.batch_size = batch_size
        self.color_map = color_map
        self.resample_method = resample_method
        self.show_progress_bar = show_progress_bar
        if seed is None:
            self.generator = None 
        else:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(seed)

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
        depth_outputs = []

        for ind_image in image:
            rgb_int = ind_image.cpu().squeeze().numpy().astype(np.uint8)  # [3, H, W]
            rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
            image_input = Image.fromarray(rgb_int)

            # NOTE: this is only for inference, it's called with torch.no_grad
            output = self.model(
                        image_input,
                        denoising_steps=self.denoise_steps,
                        ensemble_size=self.ensemble_size,
                        processing_res=self.processing_res,
                        match_input_res=self.match_input_res,
                        batch_size=self.batch_size,
                        color_map=self.color_map,
                        show_progress_bar=self.show_progress_bar,
                        resample_method=self.resample_method,
                        generator=self.generator,
                    )

            # TODO: check output size & type. Range = [0,1], but need to see if it's inverse
            depth_pred: np.ndarray = output.depth_np
            depth_outputs.append(depth_pred)

            depth_outputs = np.stack(depth_outputs)[:, None, :, :]
            depth_outputs = torch.as_tensor(depth_outputs, device=self.device)

            depth_outputs[depth_outputs == 0] += 1e-8
        return depth_outputs

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

        parameters = list(self.model.unet.parameters()) + list(self.model.vae.parameters()) + list(self.model.text_encoder.parameters())

        return parameters

    def train(self, mode=True):
        '''
        Sets model to training mode
        '''

        super().train(mode)
        # NOTE: do nothing else because this model currently only hosts MarigoldDepthPipeline, which is only for inference

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        # NOTE: do nothing because this model's MarigoldDepthPipeline is defaulted to eval() and is only used for inference
        return

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
