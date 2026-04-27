_base_ = ['CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py']
# ============================================================
# Ablation: Zero-LiDAR control
#
# Identical to FLC-PointOcc Version A except debug_zero_lidar=True.
# The full LiDAR branch (CylinderEncoder → TPVSwin → TPVFuser) still runs
# and consumes GPU memory / time, but lidar_feat is zeroed before fusion.
# This isolates whether the fusion pathway itself blocks the camera signal
# (if mIoU ≈ FlashOcc it does not; if worse it does).
# ============================================================
model = dict(
    debug_zero_lidar=True,
)
