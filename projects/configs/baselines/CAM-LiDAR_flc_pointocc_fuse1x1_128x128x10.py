_base_ = ['CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py']
# ============================================================
# Ablation: Point-wise (1×1) fusion convolution
#
# Default (Version A): fuse_conv is Conv3×3 with spatial context.
# This variant: fuse_conv is Conv1×1 (point-wise), no spatial interaction.
#
# Rationale for comparison: the 3×3 kernel provides local cross-modal
# smoothing that compensates for depth uncertainty and calibration noise.
# A 1×1 variant tests whether that spatial context matters.
#
# Only fuse_conv_cfg.kernel_size and padding differ; everything else is
# identical to Version A.
# ============================================================
model = dict(
    fuse_conv_cfg=dict(
        in_channels=384,    # cam_adapter_out(256) + lidar_adapter_out(128)
        out_channels=256,
        kernel_size=1, stride=1, padding=0, bias=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
    ),
)
