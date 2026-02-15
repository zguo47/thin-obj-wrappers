import os, sys, argparse
import torch, torchvision
sys.path.insert(0, os.path.join('external_src', 'monocular_depth', 'UniDepth'))
from unidepth.models import UniDepthV1, UniDepthV2, UniDepthV2old

class UniDepthModel(object):
    '''
    Class for interfacing with UniDepth model

    Arg(s):
        device : torch.device
            device to run model on
        max_depth : float
            value to clamp ground truths to in computing loss
        use_pretrained : bool
            if set, then configure using legacy settings
    '''
    def __init__(self, use_pretrained=True, model_type='unidepth-v2-vitl14', inference=True, device=torch.device('cuda')):
        super(UniDepthModel, self).__init__()
        version, backbone = model_type.split('-')[1:3]

        if use_pretrained:
            if version == 'v1':
                self.model = UniDepthV1.from_pretrained(f"lpiccinelli/unidepth-v1-{backbone}")
            elif version == 'v2':
                self.model = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{backbone}")
                # self.model = UniDepthV2old.from_pretrained("lpiccinelli/unidepth-v2old-vitl14")       # Old v2 model
            else:
                raise ValueError(f'Unsupported UniDepth version: {version}. Supported versions are "v1" and "v2".')
        else:
            # Randomly initialize the weights for freedom with the model
            sd = self.model.state_dict()
            for k, v in sd.items():
                sd[k] = torch.empty_like(v).normal_(mean=0.0, std=0.02)
            self.model.load_state_dict(sd, strict=False)

        self.inference = inference
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

        output = self.model.infer(image, intrinsics)

        return output['depth']

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

    def train(self):
        '''
        Sets model to training mode
        '''

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

    def restore_model(self, restore_path, optimizer=None):
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

        if isinstance(restore_path, list):
            restore_path = restore_path[0]

        checkpoint_dict = torch.load(restore_path, map_location=self.device)

        if 'state_dict' in checkpoint_dict:

            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint_dict['state_dict'])
            elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(checkpoint_dict['state_dict'])
            else:
                self.model.load_state_dict(checkpoint_dict['state_dict'])

            if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
                optimizer.load_state_dict(checkpoint_dict['optimizer'])

            if 'train_step' in checkpoint_dict.keys():
                train_step = checkpoint_dict['train_step']
                return train_step, optimizer
            else:
                return optimizer
        else:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.load_state_dict(checkpoint_dict)
            elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(checkpoint_dict)
            else:
                self.model.load_state_dict(checkpoint_dict)

            return None

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

        if isinstance(self.model, torch.nn.DataParallel):
            checkpoint = {
                'state_dict': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            checkpoint = {
                'net': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        else:
            checkpoint = {
                'net': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }

        torch.save(checkpoint, checkpoint_path)