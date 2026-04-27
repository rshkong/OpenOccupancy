_base_ = ['CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py']
# ============================================================
# Ablation: Project-first Z strategy (shared Linear 192→64, then flatten)
#
# Default (Version A): flatten-first
#   lidar_3d [B,192,X,Y,Z] → reshape → [B,1920,X,Y] → Conv1x1 1920→128
#   Each Z bin has independent projection weights (~246K params in adapter).
#
# This variant: project-first
#   lidar_3d [B,192,X,Y,Z] → Linear 192→64 (shared across all Z bins, ~12K)
#                          → reshape → [B,640,X,Y] → Conv1x1 640→128
#   All Z bins share the same 192→64 projection matrix.
#
# lidar_adapter.in_channels must equal lidar_z_proj_ch * Dz = 64 * 10 = 640.
# fuse_conv.in_channels is unchanged: cam_adapter_out(256) + lidar_adapter_out(128) = 384.
# ============================================================
model = dict(
    lidar_z_flatten_first=False,
    lidar_z_proj_ch=64,
    lidar_adapter_cfg=dict(
        in_channels=640,    # lidar_z_proj_ch(64) * Dz(10)
        out_channels=128,
        kernel_size=1, stride=1, padding=0, bias=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
    ),
)
