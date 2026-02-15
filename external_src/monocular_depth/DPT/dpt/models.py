import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x, affinity=None, texts=None, class_params=None):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # TODO: Print shapes of layer_1_rn, ..., layer_4_rn, affinity,
        # print(f"[DEBUG] DPT layer_1_rn shape: {layer_1_rn.shape}")
        # print(f"[DEBUG] DPT layer_2_rn shape: {layer_2_rn.shape}")
        # print(f"[DEBUG] DPT layer_3_rn shape: {layer_3_rn.shape}")
        # print(f"[DEBUG] DPT layer_4_rn shape: {layer_4_rn.shape}")
        # if affinity is not None:
        #     print(f"[DEBUG] DPT affinity shape: {affinity.shape}")
        # else:
        #     print(f"[DEBUG] DPT affinity: None")

        # TODO: Modulate layer_1_rn, ..., layer_4_rn with parameters and affinity matrix
        if affinity is not None and class_params is not None:
            B, N_cls, H_aff, W_aff = affinity.shape

            modulation = torch.zeros_like(layer_1_rn)

            for class_id in range(len(texts)):

                # Get class name (str)
                class_name = texts[class_id]

                # Extract affinity map according to class; note: class name and affinity order match
                # Note: Seems like min affinities are high... need peakier softmax?
                affinity_matrix = torch.nn.functional.interpolate(
                    affinity[:, class_id:class_id+1, ...],
                    size=layer_1_rn.shape[-2:],
                    mode='bilinear',
                    align_corners=None)

                # Get corresponding parameters for class
                class_param = class_params[class_name]

                # print(class_param.shape, affinity_matrix.shape, modulation.shape, class_name)
                # print(affinity_matrix.min(), affinity_matrix.max())

                # Sum over contributions of all classes
                modulation = modulation + affinity_matrix * class_param
                # modulation_mult = modulation_mult + affinity_matrix * class_param

                # print("modulation.shape", modulation.shape)

            layer_1_rn = layer_1_rn + modulation
            # layer_1_rn = modulation_mult*layer_1_rn + modulation_add

        # TODO: Pass the results to self.scratch.refinenet4, ...,  self.scratch.refinenet1 for predictions
        # [DEBUG] LSeg affinity shape: torch.Size([1, 32, 240, 240])
        # [DEBUG] DPT layer_1_rn shape: torch.Size([1, 256, 120, 120])
        # [DEBUG] DPT layer_2_rn shape: torch.Size([1, 256, 60, 60])
        # [DEBUG] DPT layer_3_rn shape: torch.Size([1, 256, 30, 30])
        # [DEBUG] DPT layer_4_rn shape: torch.Size([1, 256, 15, 15])
        # [DEBUG] DPT affinity shape: torch.Size([1, 32, 240, 240])

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out

    def apply_affinity_attention(self, features, affinity_matrix):
        B, C, H, W = features.shape
        N = H * W

        flat = features.view(B, C, N).permute(0, 2, 1)  # B x N x C
        propagated = torch.matmul(affinity_matrix, flat)  # B x N x C

        out = propagated.permute(0, 2, 1).view(B, C, H, W)
        return out


class DPTDepthModel(DPT):
    def __init__(
        self, path=None, non_negative=True, scale=1.0, shift=0.0, invert=False, **kwargs
    ):
        features = kwargs["features"] if "features" in kwargs else 256

        self.scale = scale
        self.shift = shift
        self.invert = invert

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x, affinity=None, texts=None, class_params=None):
        inv_depth = super().forward(x, affinity, texts, class_params).squeeze(dim=1)

        if self.invert:
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
            return depth
        else:
            return inv_depth


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

        if path is not None:
            self.load(path)