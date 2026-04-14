# 2-D custom ResNet backbone, ported from FlashOCC for OpenOccupancy.
# Input: [B, C, H, W] (2-D BEV feature map after Z-collapse)
# Output: list of [B, C_i, H_i, W_i] multi-scale feature maps

import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.models import BACKBONES


@BACKBONES.register_module()
class CustomResNet2D(nn.Module):
    """2-D ResNet BEV backbone (FlashOCC-style).

    Args:
        numC_input (int): Number of input channels (= numC_Trans * Dz after
            Z-collapse in the view transformer).
        num_layer (list[int]): Number of blocks per stage.
        num_channels (list[int] | None): Output channels per stage.
            Defaults to [2x, 4x, 8x] of numC_input.
        stride (list[int]): Stride of the first block in each stage.
        backbone_output_ids (list[int] | None): Which stage indices to return.
            Defaults to all stages.
        norm_cfg (dict): BN config.
        with_cp (bool): Use gradient checkpointing.
        block_type (str): 'Basic' or 'BottleNeck'.
    """

    def __init__(
            self,
            numC_input,
            num_layer=None,
            num_channels=None,
            stride=None,
            backbone_output_ids=None,
            norm_cfg=None,
            with_cp=False,
            block_type='Basic',
    ):
        super().__init__()

        if num_layer is None:
            num_layer = [2, 2, 2]
        if stride is None:
            stride = [2, 2, 2]
        if norm_cfg is None:
            norm_cfg = dict(type='BN')

        assert len(num_layer) == len(stride)
        num_channels = (
            [numC_input * 2 ** (i + 1) for i in range(len(num_layer))]
            if num_channels is None
            else num_channels
        )
        self.backbone_output_ids = (
            range(len(num_layer))
            if backbone_output_ids is None
            else backbone_output_ids
        )

        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        inplanes=curr_numC,
                        planes=num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                        norm_cfg=norm_cfg,
                    )
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(
                        inplanes=curr_numC,
                        planes=num_channels[i] // 4,
                        stride=1,
                        downsample=None,
                        norm_cfg=norm_cfg,
                    )
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        inplanes=curr_numC,
                        planes=num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                        norm_cfg=norm_cfg,
                    )
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(
                        inplanes=curr_numC,
                        planes=num_channels[i],
                        stride=1,
                        downsample=None,
                        norm_cfg=norm_cfg,
                    )
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            raise ValueError(f'Unsupported block_type: {block_type}')

        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

    def forward(self, x):
        """
        Args:
            x: (B, numC_input, H, W)  2-D BEV feature map

        Returns:
            feats: list of tensors at selected stage outputs, e.g.:
                [(B, 2*C, H/2, W/2),
                 (B, 4*C, H/4, W/4),
                 (B, 8*C, H/8, W/8)]
        """
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
