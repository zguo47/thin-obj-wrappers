import os, sys
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

# TODO: Add the necessary paths for your model
# Note that if you import your model, your code should be stored in external_src
sys.path.insert(0, os.path.join('external_src', 'multiview'))                 # for your wrapper code
sys.path.insert(0, os.path.join('external_src', 'multiview', 'vggt'))  

# TODO: Import necessary classes or packages for your model
import data_utils
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


class VGGTModel(object):
    def __init__(self,
                 dataset_name=None,
                 network_modules=[],
                 min_predict_depth=-1.0,
                 max_predict_depth=-1.0,
                 device=torch.device('cuda')):

        self.device = device

        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.dtype = torch.float16

    def transform_inputs(self, image):
        def pad_to_multiple_torch(img_nchw, base=14):
            n, c, h, w = img_nchw.shape
            pad_h = (base - (h % base)) % base
            pad_w = (base - (w % base)) % base
            img_padded = F.pad(img_nchw, (0, pad_w, 0, pad_h), mode="replicate")
            return img_padded, (h, w)

        if image.dtype == torch.uint8:
            image = image.float()

\
        image = image / 255.0
        image = image.to(self.device)

        image_padded, (h0, w0) = pad_to_multiple_torch(image, base=14)
        orig_sizes = [(h0, w0) for _ in range(image.shape[0])]

        return image_padded, orig_sizes

    def forward_depth(self, image):
        imgs_pad, orig_sizes = self.transform_inputs(image)
        imgs_t = imgs_pad.to(self.device, dtype=self.dtype).unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.dtype):
            aggregated_tokens_list, ps_idx = self.model.aggregator(imgs_t)
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, imgs_t.shape[-2:])
            depth_map, _ = self.model.depth_head(aggregated_tokens_list, imgs_t, ps_idx)

        dm = depth_map.squeeze(0)                
        if dm.ndim == 4 and dm.shape[-1] == 1:    
            dm = dm.squeeze(-1)                   

        depth_b1hw = dm.unsqueeze(1)   

        outs = []
        for i, (h0, w0) in enumerate(orig_sizes):
            d_i = depth_b1hw[i:i+1]
            d_rs = F.interpolate(d_i, size=(h0, w0), mode="bilinear", align_corners=False)
            outs.append(d_rs)

        output_depth = torch.cat(outs, dim=0).to(dtype=torch.float32)
        return output_depth

    # scene batching + saving
    @torch.no_grad()
    def run_scene(self,
                  scene_dir: str,
                  out_root: str,
                  batch_size: int = 30,
                  stride: int = 30):
        """
        Same batching behavior, but saving uses data_utils.save_image / data_utils.save_depth.
        Creates:
          out_root/scene_name/depth_npy/{stem}.npy
          out_root/scene_name/depth_png/{stem}.png

        """
        scene_dir = Path(scene_dir)
        out_root = Path(out_root)
        out_dir = out_root / scene_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        depth_png_dir = out_dir / "depths"
        depth_png_dir.mkdir(parents=True, exist_ok=True)

        img_dir = scene_dir / "images"
        img_paths = sorted(img_dir.glob("*.png"))
        num_imgs = len(img_paths)

        for start_idx in range(0, num_imgs, stride):
            end_idx = min(start_idx + batch_size, num_imgs)
            batch_paths = img_paths[start_idx:end_idx]

            # load batch 
            imgs = []
            for p in batch_paths:
                img = iio.imread(p)  
                imgs.append(img)

            imgs_np = np.stack(imgs, axis=0)  
            imgs_t = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).contiguous()  

            # forward
            depth_b1hw = self.forward_depth(imgs_t)  # N 1 H W float32 (original sizes)
            depth_np = depth_b1hw.squeeze(1).detach().cpu().numpy().astype(np.float32)  

            for d_hw, p in zip(depth_np, batch_paths):
                name = p.stem
                data_utils.save_depth(
                    d_hw,
                    str(depth_png_dir / f"{name}.png")
                )

            # cleanup
            del imgs_t, depth_b1hw, depth_np, imgs_np, imgs
            torch.cuda.empty_cache()
            gc.collect()

        return str(out_dir)

    def forward_pose(self, image0, image1):
        '''
        Forwards a pair of images through the network to output pose from time 0 to 1

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W tensor
            image1 : torch.Tensor[float32]
                N x C x H x W tensor
        Returns:
            torch.Tensor[float32] : N x 4 x 4  pose matrix
        '''

        assert self.model_pose is not None

        pose = None

        return pose

    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map0,
                     intrinsics,
                     pose0to1,
                     pose0to2,
                     w_losses):
        '''
        Computes loss function

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time step t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at time step t+1
            output_depth0 : list[torch.Tensor[float32]]
                list of N x 1 x H x W output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            validity_map0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t-1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t+1
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
            w_pose : float
                weight of pose consistency term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        # TODO: Implement or call your model's compute loss function
        # This includes any quantities or images you would like to log into the loss_info dictionary
        # Each key-value pair in loss_info will be logged onto tensorboard and
        # should contain image reconstructions, 'image1to0', 'image2to0' and any scalars e.g., 'loss'
        loss, loss_info = None, None

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        # TODO: Return the parameters of your entire (depth and pose) model
        return None

    def parameters_depth(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        # TODO: Return the parameters of your depth model
        return None

    def parameters_pose(self):
        '''
        Fetches model parameters for pose network modules

        Returns:
            list[torch.Tensor[float32]] : list of model parameters for pose network modules
        '''

        if 'pose' in self.network_modules:
            return None
        else:
            raise ValueError('Unsupported pose network architecture: {}'.format(self.network_modules))

    def train(self):
        '''
        Sets model to training mode
        '''

        # TODO: Set your model into training mode
        pass

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        # TODO: Set your model into evaluation mode
        pass

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device

        # TODO: Moves your model to device
        pass

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        # TODO: Implement or call your model DataParallel function
        pass

    def restore_model(self, restore_path, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optim
                optimizer
        '''

        # TODO: Implement or call your model restore function
        _, optimizer = None, None

        return optimizer

    def save_model(self, checkpoint_path, step, optimizer):
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

        # TODO: Implement or call your model saving function
        pass
