# 2-D FPN neck (FPN_LSS), ported from FlashOCC for OpenOccupancy.
# Takes multi-scale 2-D BEV features and fuses them into a single map.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet.models import NECKS


@NECKS.register_module()
class FPN_LSS(nn.Module):
    """2-D FPN neck from FlashOCC (LSS-style feature pyramid).

    Fuses a low-resolution high-level feature with a higher-resolution
    low-level feature by upsampling and concatenation, then optionally
    performs a second upsample to restore the original BEV resolution.

    Args:
        in_channels (int): Total input channels after concatenating the two
            selected feature maps (low-level + upsampled high-level).
        out_channels (int): Output channels of the neck.
        scale_factor (int): Upsample factor for the high-level feature map
            before concatenation. Default: 4.
        input_feature_index (tuple[int]): Indices into the backbone output
            list selecting (low-level, high-level) feature maps.
            Default: (0, 2).
        norm_cfg (dict): Normalization layer config.
        extra_upsample (int | None): If set, apply an additional upsample of
            this factor at the end. Default: 2.
        lateral (int | None): If set, apply a 1×1 lateral conv to the
            low-level feature before fusion (channel size = lateral).
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor=4,
            input_feature_index=(0, 2),
            norm_cfg=None,
            extra_upsample=2,
            lateral=None,
            use_input_conv=False,
    ):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='BN')

        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.out_channels = out_channels

        # Upsample the high-level feature to match the low-level spatial size
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor,
                      kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor,
                      kernel_size=3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )

        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            )

        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Multi-scale 2-D BEV feature maps from the
                backbone, e.g.:
                  [(B, C1, H, W),
                   (B, C2, H/2, W/2),
                   (B, C3, H/4, W/4)]

        Returns:
            x (Tensor): Fused BEV feature map.
                Without extra_upsample: (B, out_channels*2, H, W)
                With    extra_upsample: (B, out_channels,   2*H, 2*W)
        """
        x2 = feats[self.input_feature_index[0]]   # low-level  (higher res)
        x1 = feats[self.input_feature_index[1]]   # high-level (lower res)

        if self.lateral:
            x2 = self.lateral_conv(x2)

        x1 = self.up(x1)                          # upsample to low-level size
        x1 = torch.cat([x2, x1], dim=1)           # concatenate along channels
        x = self.conv(x1)

        if self.extra_upsample:
            x = self.up2(x)

        return x
