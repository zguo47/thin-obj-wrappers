import os, sys
import torch
# TODO: Add the necessary paths for your model
# Note that if you import your model, your code should be stored in external_src
sys.path.insert(0, os.path.join('external_src', 'depth_completion'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'OMNI-DC', 'src'))
from model.ognidc import OGNIDC
from config import args  as args_config
import torchvision.transforms as T

# TODO: Import necessary classes or packages for your model


class OMNIDCModel(object):
    '''
    Template for interfacing with your model model

    Arg(s):
        dataset_name : str
            model for a given dataset
        network_modules : list[str]
            network modules to build for model
        min_predict_depth : float
            minimum value of predicted depth
        max_predict_depth : flaot
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
        args_main = self.check_args(args_config)
        self.args = args_main
        self.net = OGNIDC.from_pretrained("zuoym15/OMNI-DC", args=args_main)
        self.t_rgb = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # TODO: Move your model to device
        self.device = device
        self.net.to(self.device)

    def check_args(self, args):
        new_args = args
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            if args.resume:
                checkpoint = torch.load(args.pretrain)

                # new_args = checkpoint['args']
                new_args.test_only = args.test_only
                new_args.pretrain = args.pretrain
                new_args.dir_data = args.dir_data
                new_args.resume = args.resume
                new_args.start_epoch = checkpoint['epoch'] + 1

        return new_args
    
    
    def transform_inputs(self, image, sparse_depth):
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

        # TODO: Implement or call your model's data transformation function
        # e.g., normalization
        image = self.t_rgb(image)

        _, _, H, W = image.shape
        diviser = int(4 * 2 ** (self.args.num_resolution - 1))
        if not H % diviser == 0:
            H_new = (H // diviser + 1) * diviser
            H_pad = H_new - H
            image = torch.nn.functional.pad(image, (0, 0, 0, H_pad))
            sparse_depth = torch.nn.functional.pad(sparse_depth, (0, 0, 0, H_pad))
        else:
            H_new = H
            H_pad = 0

        if not W % diviser == 0:
            W_new = (W // diviser + 1) * diviser
            W_pad = W_new - W
            image = torch.nn.functional.pad(image, (0, W_pad, 0, 0))
            sparse_depth = torch.nn.functional.pad(sparse_depth, (0, W_pad, 0, 0))
        else:
            W_new = W
            W_pad = 0
        return image, sparse_depth, (H_new, H_pad, W_new, W_pad)

    def forward_depth(self, image, sparse_depth, validity_map, intrinsics, return_all_outputs=False):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations of projected sparse point cloud
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
            return_all_outputs : bool
                if set, return all outputs
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''
        image, sparse_depth, (H_new, H_pad, W_new, W_pad) = self.transform_inputs(image, sparse_depth)
        
        input_dict = {
            'rgb': image,
            'dep': sparse_depth,
            'K': intrinsics,
            'pattern': 0
        }
        # TODO: Implement or call your model's forward function
        output = self.net(input_dict)
        

        output_depth = output['pred'][..., :H_new - H_pad, :W_new - W_pad]
        depth_inter = [pred[..., :H_new - H_pad, :W_new - W_pad] for pred in output['pred_inter']]

        if return_all_outputs:
            # TODO: Return model output along with any auxiliary outputs
            # e.g., multiscale outputs for loss computation
            return [output_depth]
        else:
            # TODO: Return model output
            return output_depth

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
        return self.net.parameters()

    def parameters_depth(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''
        
        # TODO: Return the parameters of your depth model
        return self.net.parameters()

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
        self.net.train()
        pass

    def eval(self):
        '''
        Sets model to evaluation mode
        '''
        self.net.eval()
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
        self.net.to(self.device)
        # TODO: Moves your model to device
        pass

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        # TODO: Implement or call your model DataParallel function
        pass
        self.net = torch.nn.DataParallel(self.net)

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
