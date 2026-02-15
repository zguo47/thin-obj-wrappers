import os
from pathlib import Path
from typing import Optional, Tuple, List
import torch
import copy
# TODO: Add the necessary paths for your model
# Note that if you import your model, your code should be stored in external_src

# TODO: Import necessary classes or packages for your model
# Nerfstudio 
import numpy as np
import data_utils
from nerfstudio.configs.method_configs import method_configs
from nerfstudio.utils.eval_utils import eval_setup

class NerfactoModel(object):
    '''
    Template for interfacing with your model model

    Arg(s):
        dataset_name : str
            model for a given dataset
        network_modules : list[str]
            network modules to build for model
        min_predict_depth : float
            minimum value of predicted depth
        max_predict_depth : float
            maximum value of predicted depth
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 dataset_name=None,
                 network_modules=[],
                 min_predict_depth=-1.0,
                 max_predict_depth=-1.0,
                 device=torch.device('cuda')):

        # TODO: Instantiate your depth completion model
        self.device = device 

        self.train_renderings = None
        self.train_depths = None
        self.test_renderings = None
        self.test_depths = None

        self.cfg = None
        self.trainer = None
        self.pipeline = None

        self.load_config_path = None

        self.output_dir = None
        self.model_type = None

    def transform_inputs(self, image):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
        '''

        pass

    def view_key(self, dm, split: str, i: int) -> str:
        """
        Get a stable filename stem for frame i in split using nerfstudio dataset/dataparser metadata.
        """
        split = split.lower()

        dataset = dm.train_dataset if split == "train" else dm.eval_dataset

        paths = getattr(dataset, "image_filenames", None)
        base = os.path.basename(str(paths[i]))
        return os.path.splitext(base)[0]


    def optimize(self,
                scene_dir,
                output_root,
                model_type="nerfacto"):
        """
        Trains Nerfstudio (nerfacto / depth-nerfacto) on a COLMAP scene.
        Returns (cfg, trainer, pipeline).
        """
        import copy
        from pathlib import Path

        # COLMAP dataparser
        from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

        scene_dir = Path(scene_dir).resolve()
        output_root = Path(output_root).resolve()

        spec = method_configs[model_type]
        cfg = copy.deepcopy(spec.config if hasattr(spec, "config") else spec)

        dm_cfg = cfg.pipeline.datamanager

        colmap_dp = ColmapDataParserConfig(data=scene_dir)
        dm_cfg.dataparser = colmap_dp

        cfg.data = str(scene_dir)
        cfg.output_dir = str(output_root)
        setattr(cfg, "experiment_name", "model")
        cfg.timestamp = "."
        cfg.vis = "tensorboard"

        trainer = cfg.setup()
        trainer.setup()
        trainer.train()

        pipeline = getattr(trainer, "pipeline", None) or getattr(trainer, "_pipeline", None)

        self.cfg = cfg
        self.trainer = trainer
        self.pipeline = pipeline
        self.output_dir = str(output_root)
        self.model_type = model_type

        return cfg, trainer, pipeline



    @torch.no_grad()
    def render(self,
            split= "test",
            load_config_path=None):
        """
        Render using the pipeline saved in self.pipeline by optimize().

        Args:
            split: "train" or "test"
            load_config_path: optional fallback path to config.yml (only used if self.pipeline is None)

        Returns:
            renderings: [N,3,H,W] float32 on CPU
            depths:     [N,1,H,W] float32 on CPU, or None if depth not available
        """
        if load_config_path is None:
            load_config_path = self.load_config_path

        pipeline = self.pipeline

        if pipeline is None:
            if load_config_path is None:
                raise ValueError("self.pipeline is None. Run optimize() first or provide load_config_path.")
            load_config_path = str(Path(load_config_path).resolve())
            out = eval_setup(load_config_path, test_mode=split)
            pipeline = out[0]
            self.pipeline = pipeline  

        pipeline.to(self.device).eval()


        # get cameras
        dm = pipeline.datamanager
        if split == "train":
            cameras = dm.train_dataset.cameras
        else:
            cameras = dm.eval_dataset.cameras


        # render
        rgbs: List[torch.Tensor] = []
        depths: List[torch.Tensor] = []

        for i in range(len(cameras)):
            cam_i = cameras[i:i+1].to(self.device)
            outputs = pipeline.model.get_outputs_for_camera(cam_i)

            rgb = outputs["rgb"].permute(2, 0, 1).contiguous()  # [3,H,W]
            rgbs.append(rgb)

            depth = outputs["depth"]

            # normalize to [H, W]
            depth = depth[..., 0]

            # store as [1, H, W] so stacking gives [N,1,H,W]
            depths.append(depth.unsqueeze(0).contiguous())


        renderings = torch.stack(rgbs, dim=0)
        depth_tensor = torch.stack(depths, dim=0)

        if split == "train":
            self.train_renderings, self.train_depths = renderings, depth_tensor
        else:
            self.test_renderings, self.test_depths = renderings, depth_tensor
        
        # save outputs 
        if getattr(self, "output_dir", None) is not None:
            render_dir = os.path.join(self.output_dir, split, "renders")
            depth_dir  = os.path.join(self.output_dir, split, "depths")
            os.makedirs(render_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            # renderings: [N,3,H,W], depth_tensor: [N,1,H,W]
            for i in range(renderings.shape[0]):
                cam = cameras[i]
                key = self.view_key(dm, split, i)

                rgb = renderings[i].detach().cpu().numpy()  # [3,H,W]
                dep = depth_tensor[i].detach().cpu().numpy()  # [1,H,W] or [H,W]

                # save rgb (CHW, normalized=True)
                data_utils.save_image(
                    rgb,
                    os.path.join(render_dir, f"{key}.png"),
                    normalized=True,
                    data_type="color",
                    data_format="CHW",
                )

                # save depth (HxW float -> 16-bit png)
                dep = np.squeeze(dep)
                data_utils.save_depth(
                    dep.astype("float32"),
                    os.path.join(depth_dir, f"{key}.png"),
                )

        return renderings, depth_tensor

    @torch.no_grad()
    def run_render(self, split="test"):
        '''
        Renders the model for a given split
        '''
        return self.render(split=split, load_config_path=None)

    def set_render_inputs(self, load_config_path=None):
        '''
        Sets inputs required for rendering
        '''
        self.load_config_path = load_config_path

    def get_train_outputs(self):
        return self.train_renderings, self.train_depths

    def get_test_outputs(self):
        return self.test_renderings, self.test_depths

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
