import os, sys
import torch
import numpy as np
from random import randint
from tqdm import tqdm
from os import makedirs
import torchvision
# TODO: Add the necessary paths for your model
# Note that if you import your model, your code should be stored in external_src
sys.path.insert(0, os.path.join('external_src', 'novel_view_synthesis'))
sys.path.insert(0, os.path.join('external_src', 'novel_view_synthesis', 'gaussian-splatting'))
# TODO: Import necessary classes or packages for your model
import data_utils
from gaussian_renderer import render
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    fused_ssim = None
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam   
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

class GaussianSplatting3DModel(object):
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
        self.pipeline = None
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
            stem = os.path.splitext(base)[0]
            return stem


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
        self.pipeline = pipe
        self.iteration = opt.iterations

        # --- init model ---
        first_iter = 0
        self.gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        self.scene = Scene(dataset, self.gaussians)
        self.gaussians.training_setup(opt)

        if checkpoint_path is not None:
            model_params, first_iter = torch.load(checkpoint_path, map_location="cpu")
            self.gaussians.restore(model_params, opt)
            first_iter = int(first_iter)

        # --- background ---
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        use_sparse_adam = (opt.optimizer_type == "sparse_adam") and SPARSE_ADAM_AVAILABLE
        depth_l1_weight = get_expon_lr_func(
            opt.depth_l1_weight_init,
            opt.depth_l1_weight_final,
            max_steps=opt.iterations
        )

        viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
    

        ema_loss_for_log = 0.0
        ema_Ll1depth_for_log = 0.0

        # tqdm expects number of steps, so use range over remaining iters
        progress_bar = tqdm(
            range(first_iter, opt.iterations),
            desc="Training progress",
            dynamic_ncols=True
        )
        first_iter += 1  # match official logic

        # --- main loop ---
        for iteration in range(first_iter, opt.iterations + 1):

            self.gaussians.update_learning_rate(iteration)

            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # sample camera
            if not viewpoint_stack:
                viewpoint_stack = self.scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))

            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            _ = viewpoint_indices.pop(rand_idx)

            bg = torch.rand((3), device=self.device) if opt.random_background else background

            render_pkg = render(
                viewpoint_cam,
                self.gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=use_sparse_adam
            )

            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            if getattr(viewpoint_cam, "alpha_mask", None) is not None:
                image = image * viewpoint_cam.alpha_mask.to(self.device)

            # loss
            gt_image = viewpoint_cam.original_image.to(self.device)
            Ll1 = l1_loss(image, gt_image)

            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

            # depth regularization (optional)
            Ll1depth = 0.0
            if depth_l1_weight(iteration) > 0 and getattr(viewpoint_cam, "depth_reliable", False):
                invDepth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.to(self.device)
                depth_mask = viewpoint_cam.depth_mask.to(self.device)

                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                Ll1depth = (depth_l1_weight(iteration) * Ll1depth_pure).item()
                loss = loss + depth_l1_weight(iteration) * Ll1depth_pure

            # backward
            loss.backward()

            # --- densification & optimizer step ---
            with torch.no_grad():
                # -------- progress bar update (EMA like official) --------
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.7f}",
                        "Depth Loss": f"{ema_Ll1depth_for_log:.7f}",
                        "Pts": int(self.gaussians.get_xyz.shape[0]),
                    })
                    progress_bar.update(10)

                if iteration == opt.iterations:
                    progress_bar.close()

                # -------- densification --------
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
                            0.005,
                            self.scene.cameras_extent,
                            size_threshold,
                            radii
                        )

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        self.gaussians.reset_opacity()

                # -------- optimizer step --------
                if iteration < opt.iterations:
                    self.gaussians.exposure_optimizer.step()
                    self.gaussians.exposure_optimizer.zero_grad(set_to_none=True)

                    if use_sparse_adam:
                        visible = radii > 0
                        self.gaussians.optimizer.step(visible, radii.shape[0])
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                    else:
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.scene.save(opt.iterations)
        torch.save(
        (self.gaussians.capture(), opt.iterations),
        os.path.join(dataset.model_path, "chkpnt" + str(opt.iterations) + ".pth")
    )
        # return trained objects 
        return self.scene, self.gaussians


    @torch.no_grad()
    def render_set(self,
                   model_path: str,
                   name: str,
                   iteration: int,
                   views,
                   gaussians,
                   pipeline,
                   background: torch.Tensor,
                   train_test_exp: bool,
                   separate_sh: bool):
        """
        
        """
        render_list, depth_list, key_list = [], [], []

        for view in tqdm(views, desc=f"Rendering {name}"):
            pkg = render(
                view, gaussians, pipeline, background,
                use_trained_exp=train_test_exp,
                separate_sh=separate_sh
            )
            rendering = pkg["render"]
            depth = pkg["depth"]

            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]

            render_list.append(rendering)
            depth_list.append(depth)
            key_list.append(self.view_key(view))

        if len(render_list) == 0:
            return None, None, None

        renderings = torch.stack(render_list, dim=0)  # [N,3,H,W]
        depths = torch.stack(depth_list, dim=0)       # [N,1,H,W]
        return renderings, depths, key_list

    @torch.no_grad()
    def render_sets(self,
                    dataset,
                    pipeline,
                    iteration: int = -1,
                    skip_train: bool = False,
                    skip_test: bool = False):
        """
        Minimal port of render_sets(...) from official render.py.

        """

        if dataset is None:
            dataset = self.dataset
        if pipeline is None:
            pipeline = self.pipeline

        separate_sh = SPARSE_ADAM_AVAILABLE

        scene = self.scene
        gaussians = self.gaussians

        if iteration is None or iteration < 0:
            iteration = self.iteration

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=self.device)

        if not skip_train:
            train_renderings, train_depths, train_keys = self.render_set(
                dataset.model_path, "train", iteration, scene.getTrainCameras(),
                gaussians, pipeline, background, dataset.train_test_exp, separate_sh
            )
            self.train_renderings, self.train_depths = train_renderings, train_depths

            if train_renderings is not None:
                render_dir = os.path.join(dataset.model_path, "train", "renders")
                depth_dir  = os.path.join(dataset.model_path, "train", "depths")
                os.makedirs(render_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                for key, rgb_t, dep_t in zip(train_keys, train_renderings, train_depths):
                    # if key is None for some reason, fallback to a safe sequential id
                    if key is None:
                        key = "unknown"

                    rgb = rgb_t.detach().cpu().numpy()  # [3,H,W]
                    dep = dep_t.detach().cpu().numpy()  # [1,H,W] or [H,W]

                    data_utils.save_image(
                        rgb,
                        os.path.join(render_dir, f"{key}.png"),
                        normalized=True, data_type="color", data_format="CHW"
                    )

                    if dep.ndim == 3 and dep.shape[0] == 1:
                        dep = dep[0]
                    data_utils.save_depth(
                        dep.astype(np.float32),
                        os.path.join(depth_dir, f"{key}.png")
                    )


        if not skip_test:
            test_renderings, test_depths, test_keys = self.render_set(
                dataset.model_path, "test", iteration, scene.getTestCameras(),
                gaussians, pipeline, background, dataset.train_test_exp, separate_sh
            )
            self.test_renderings, self.test_depths = test_renderings, test_depths

            if test_renderings is not None:
                render_dir = os.path.join(dataset.model_path, "test", "renders")
                depth_dir  = os.path.join(dataset.model_path, "test", "depths")
                os.makedirs(render_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                for key, rgb_t, dep_t in zip(test_keys, test_renderings, test_depths):
                    if key is None:
                        key = "unknown"

                    rgb = rgb_t.detach().cpu().numpy()
                    dep = dep_t.detach().cpu().numpy()

                    data_utils.save_image(
                        rgb,
                        os.path.join(render_dir, f"{key}.png"),
                        normalized=True, data_type="color", data_format="CHW"
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
            self.render_sets(
                dataset=self.dataset,
                pipeline=self.pipeline,
                iteration=self.iteration,
                skip_train=False,
                skip_test=True
            )
            return self.get_train_outputs()
        else:
            self.render_sets(
                dataset=self.dataset,
                pipeline=self.pipeline,
                iteration=self.iteration,
                skip_train=True,
                skip_test=False
            )
            return self.get_test_outputs()

    def set_paths(self, source_path: str, model_path: str):
        '''
        Sets paths required for training and rendering
        '''
        self.source_path = source_path
        self.model_path = model_path

    def set_render_inputs(self, dataset, pipeline, iteration: int = -1):
        '''
        Sets inputs required for rendering
        '''
        self.dataset = dataset
        self.pipeline = pipeline
        self.iteration = iteration

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

    def get_train_outputs(self):
        '''
        Fetches training set outputs

        Returns:
            torch.Tensor[float32] : N x 3 x H x W rendered images
            torch.Tensor[float32] : N x 3 x H x W ground truth images
            torch.Tensor[float32] : N x 1 x H x W depth maps
        '''

        return self.train_renderings, self.train_depths
    
    def get_test_outputs(self):
        '''
        Fetches test set outputs

        Returns:
            torch.Tensor[float32] : N x 3 x H x W rendered images
            torch.Tensor[float32] : N x 3 x H x W ground truth images
            torch.Tensor[float32] : N x 1 x H x W depth maps
        '''

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
