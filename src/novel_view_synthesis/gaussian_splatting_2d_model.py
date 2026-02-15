import os, sys
import torch
from random import randint
import numpy as np
# TODO: Add the necessary paths for your model
# Note that if you import your model, your code should be stored in external_src
sys.path.insert(0, os.path.join('external_src', 'novel_view_synthesis'))
sys.path.insert(0, os.path.join('external_src', 'novel_view_synthesis', '2d-gaussian-splatting'))
# TODO: Import necessary classes or packages for your model
import data_utils
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.mesh_utils import GaussianExtractor
from utils.general_utils import safe_state
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

class GaussianSplatting2DModel(object):
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

        self.gaussians = None
        self.scene = None

        self.train_renderings = None
        self.train_depths = None

        self.test_renderings = None
        self.test_depths = None

        self.dataset = None
        self.pipe = None
        self.iteration = -1

        self.source_path = None
        self.model_path = None

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

    def view_key(self, view):
        """
        Returns a stable filename stem for this view.
        Prefers image_name
        """
        name = getattr(view, "image_name", None)
        if name is not None:
            base = os.path.basename(str(name))
            return os.path.splitext(base)[0]


    def optimize(self, dataset, opt, pipe, checkpoint_path=None,):
        '''
        Runs the training loop as in the official repo

        Arg(s):
            dataset : object returned by ModelParams.extract(args)
            opt     : object returned by OptimizationParams.extract(args)
            pipe    : object returned by PipelineParams.extract(args)
            checkpoint_path : str or None
                path to checkpoint saved as (gaussians.capture(), iteration)
        Returns:
            
        '''

        if dataset is None or opt is None or pipe is None:
            dataset, opt, pipe = self.set_default_params()
        
        self.dataset = dataset
        self.pipe = pipe
        self.iteration = opt.iterations 

        # --- init model ---
        first_iter = 0
        self.gaussians = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians)
        self.gaussians.training_setup(opt)

        if checkpoint_path is not None:
            model_params, first_iter = torch.load(checkpoint_path, map_location="cpu")
            first_iter = int(first_iter)
            self.gaussians.restore(model_params, opt)

        # --- background ---
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        viewpoint_stack = self.scene.getTrainCameras().copy()

        ema_loss_for_log = 0.0
        progress_bar = tqdm(
            range(first_iter, opt.iterations),
            desc="Training progress",
            dynamic_ncols=True
        )
        first_iter += 1

    
        # --- main loop ---
        for iteration in range(first_iter, opt.iterations + 1):

            self.gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # sample camera
            if len(viewpoint_stack) == 0:
                viewpoint_stack = self.scene.getTrainCameras().copy()

            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))


            render_pkg = render(viewpoint_cam, self.gaussians, pipe, background)
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            # loss
            gt_image = viewpoint_cam.original_image.to(self.device)
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]

            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * normal_error.mean()
            dist_loss = lambda_dist * rend_dist.mean()

            total_loss = loss + dist_loss + normal_loss

            total_loss.backward()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.7f}",
                        "Pts": int(self.gaussians.get_xyz.shape[0]),
                    })
                    progress_bar.update(10)

                if iteration == opt.iterations:
                    progress_bar.close()

                if iteration < opt.densify_until_iter:
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter]
                    )
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            opt.opacity_cull,
                            self.scene.cameras_extent,
                            size_threshold
                        )

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        self.gaussians.reset_opacity()

                # optimizer step
                if iteration < opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.scene.save(opt.iterations) 
        torch.save((self.gaussians.capture(), opt.iterations), os.path.join(dataset.model_path, f"chkpnt{opt.iterations}.pth")
   )
        
        return self.scene, self.gaussians


    @torch.no_grad()
    def render(self,
               dataset,
               pipe,
               iteration: int = -1,
               skip_train: bool = False,
               skip_test: bool = False):

        if dataset is None:
            dataset = self.dataset
        if pipe is None:
            pipe = self.pipe

        scene = self.scene
        gaussians = self.gaussians

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        gaussExtractor = GaussianExtractor(
            gaussians,
            render,
            pipe,
            bg_color=bg_color
        )

        # Use the passed iteration if given; otherwise fall back to self.iteration
        it = iteration if iteration != -1 else self.iteration

        if not skip_train:
            gaussExtractor.reconstruction(scene.getTrainCameras())
            self.train_renderings, self.train_depths = gaussExtractor.export_renders()

            # ---- save train renders/depths ----
            render_dir = os.path.join(dataset.model_path, "train", "renders")
            depth_dir  = os.path.join(dataset.model_path, "train", "depths")
            os.makedirs(render_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            train_views = scene.getTrainCameras()
            gaussExtractor.reconstruction(train_views)
            self.train_renderings, self.train_depths = gaussExtractor.export_renders()

            render_dir = os.path.join(dataset.model_path, "train", "renders")
            depth_dir  = os.path.join(dataset.model_path, "train", "depths")
            os.makedirs(render_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            for view, rgb_t, dep_t in zip(train_views, self.train_renderings, self.train_depths):
                key = self.view_key(view) or "unknown"
                rgb = rgb_t.detach().cpu().numpy()
                dep = dep_t.detach().cpu().numpy()

                data_utils.save_image(
                    rgb, os.path.join(render_dir, f"{key}.png"),
                    normalized=True, data_type="color", data_format="CHW"
                )

                if dep.ndim == 3 and dep.shape[0] == 1:
                    dep = dep[0]
                data_utils.save_depth(
                    dep.astype(np.float32),
                    os.path.join(depth_dir, f"{key}.png")
                )


        if (not skip_test) and (len(scene.getTestCameras()) > 0):
            gaussExtractor.reconstruction(scene.getTestCameras())
            self.test_renderings, self.test_depths = gaussExtractor.export_renders()

            # ---- save test renders/depths ----
            render_dir = os.path.join(dataset.model_path, "test", "renders")
            depth_dir  = os.path.join(dataset.model_path, "test", "depths")
            os.makedirs(render_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)

            test_views = scene.getTestCameras()

            for view, rgb_t, dep_t in zip(test_views, self.test_renderings, self.test_depths):
                key = self.view_key(view) or "unknown"
                rgb = rgb_t.detach().cpu().numpy()
                dep = dep_t.detach().cpu().numpy()

                data_utils.save_image(
                    rgb,
                    os.path.join(render_dir, f"{key}.png"),
                    normalized=True,
                    data_type="color",
                    data_format="CHW"
                )

                if dep.ndim == 3 and dep.shape[0] == 1:
                    dep = dep[0]
                data_utils.save_depth(
                    dep.astype(np.float32),
                    os.path.join(depth_dir, f"{key}.png")
                )

       
    @torch.no_grad()
    def run_render(self, split: str = "test"):
        split = split.lower()

        if split == "train":
            self.render(
                dataset=self.dataset,
                pipe=self.pipe,
                iteration=self.iteration,
                skip_train=False,
                skip_test=True
            )
            return self.get_train_outputs()
        else:
            self.render(
                dataset=self.dataset,
                pipe=self.pipe,
                iteration=self.iteration,
                skip_train=True,
                skip_test=False
            )
            return self.get_test_outputs()

    def get_train_outputs(self):
        '''
        Fetches training outputs after rendering

        Returns:
            torch.Tensor[float32] : N x 3 x H x W rendered images
            torch.Tensor[float32] : N x 1 x H x W depth maps
        '''

        return self.train_renderings, self.train_depths
    
    def get_test_outputs(self):
        '''
        Fetches testing outputs after rendering

        Returns:
            torch.Tensor[float32] : N x 3 x H x W rendered images
            torch.Tensor[float32] : N x 1 x H x W depth maps
        '''

        return self.test_renderings, self.test_depths
    
    def set_render_inputs(self, dataset, pipe, iteration: int = -1):
        '''
        Sets inputs required for rendering
        '''
        self.dataset = dataset
        self.pipe = pipe
        self.iteration = iteration

    def set_paths(self, source_path: str, model_path: str):
        self.source_path = source_path
        self.model_path = model_path
    
    def set_default_params(self):
        """
        Build (dataset, opt, pipe) using official argument defaults,
        overriding only source_path/model_path.
        """

        parser = ArgumentParser(add_help=False)
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)

        args = parser.parse_args([])  
        args.source_path = self.source_path
        args.model_path = self.model_path

        safe_state(True)

        dataset = lp.extract(args)
        opt = op.extract(args)
        pipe = pp.extract(args)
        return dataset, opt, pipe


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
