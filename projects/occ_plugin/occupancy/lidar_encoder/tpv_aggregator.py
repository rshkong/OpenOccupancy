"""TPVAggregator: full PointOcc classification head on 3 TPV planes.

Ported from PointOcc TPVAggregator_Occ, with the loss block removed — the
detector owns loss computation so OpenOccupancy-style training hooks work.

Pipeline
--------
    tpv_list (3 planes) + voxels_coarse (B, N, 3)
        → interpolate each plane to (tpv_dim * scale) resolution
        → grid_sample × 3 at coarse voxel positions
        → element-wise sum of 3 samples  (B, C, N)
        → decoder MLP: Linear → Softplus → Linear  (B, N, out_dims)
        → classifier: Linear(out_dims → nbr_classes)
        → logits (B, nbr_classes, W_c, H_c, D_c)

Differences vs. TPVFuser (the fusion-model variant)
    - TPVFuser returns a dense 3D feature [B, C, W, H, D]; no MLP, no classifier.
    - TPVAggregator returns logits ready for CE / Lovasz / sem_scal / geo_scal.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class TPVAggregator(BaseModule):
    """PointOcc TPVAggregator_Occ ported to mmdet2, decoupled from loss.

    Args:
        tpv_h, tpv_w, tpv_z: native TPV plane dims (before scale).
        grid_size_occ: target occupancy grid [W_occ, H_occ, D_occ].
        coarse_ratio: downscale factor so coarse grid is grid_size_occ // ratio.
        nbr_classes: number of semantic classes (including empty).
        in_dims: channel count coming from TPVFPN (per plane).
        hidden_dims, out_dims: decoder MLP dims.
        scale_h, scale_w, scale_z: bilinear upsample factors applied to each
            TPV plane before grid_sample.  voxels_coarse must be produced in
            range [0, tpv_dim * scale) on the matching axis.
        use_checkpoint: pass decoder/classifier through torch.utils.checkpoint.
    """

    def __init__(self,
                 tpv_h, tpv_w, tpv_z,
                 grid_size_occ, coarse_ratio=1,
                 nbr_classes=17,
                 in_dims=192, hidden_dims=384, out_dims=None,
                 scale_h=2, scale_w=2, scale_z=2,
                 use_checkpoint=False):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.grid_size_occ = np.asarray(grid_size_occ).astype(np.int32)
        self.coarse_ratio = coarse_ratio
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint

        out_dims = in_dims if out_dims is None else out_dims
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims),
        )
        self.classifier = nn.Linear(out_dims, nbr_classes)

    def forward(self, tpv_list, voxels_coarse):
        """Forward pass.

        Args:
            tpv_list: [tpv_xy, tpv_yz, tpv_zx]
                tpv_xy: (B, C, W, H)
                tpv_yz: (B, C, H, Z)
                tpv_zx: (B, C, Z, W)
            voxels_coarse: (B, N, 3) — continuous cylindrical grid coords
                already rescaled so axis i lies in [0, self_tpv_dim_i * scale_i).

        Returns:
            logits: (B, nbr_classes, W_c, H_c, D_c) — coarse occupancy logits.
        """
        tpv_xy, tpv_yz, tpv_zx = tpv_list[0], tpv_list[1], tpv_list[2]
        tpv_hw = tpv_xy.permute(0, 1, 3, 2)  # (B, C, H, W)
        tpv_wz = tpv_zx.permute(0, 1, 3, 2)  # (B, C, W, Z)
        tpv_zh = tpv_yz.permute(0, 1, 3, 2)  # (B, C, Z, H)
        bs, c, _, _ = tpv_hw.shape

        # Interpolate each plane to target resolution.
        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(int(self.tpv_h * self.scale_h),
                      int(self.tpv_w * self.scale_w)),
                mode='bilinear', align_corners=False)
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(int(self.tpv_z * self.scale_z),
                      int(self.tpv_h * self.scale_h)),
                mode='bilinear', align_corners=False)
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(int(self.tpv_w * self.scale_w),
                      int(self.tpv_z * self.scale_z)),
                mode='bilinear', align_corners=False)

        # Normalise voxel coords to [-1, 1] for grid_sample.
        _, n, _ = voxels_coarse.shape
        voxels_coarse = voxels_coarse.reshape(bs, 1, n, 3).clone()
        voxels_coarse[..., 0] = voxels_coarse[..., 0] / (
            self.tpv_w * self.scale_w) * 2 - 1
        voxels_coarse[..., 1] = voxels_coarse[..., 1] / (
            self.tpv_h * self.scale_h) * 2 - 1
        voxels_coarse[..., 2] = voxels_coarse[..., 2] / (
            self.tpv_z * self.scale_z) * 2 - 1

        sample_xy = voxels_coarse[..., [0, 1]]
        tpv_hw_vox = F.grid_sample(
            tpv_hw, sample_xy, padding_mode='border',
            align_corners=False).squeeze(2)  # (B, C, N)
        sample_yz = voxels_coarse[..., [1, 2]]
        tpv_zh_vox = F.grid_sample(
            tpv_zh, sample_yz, padding_mode='border',
            align_corners=False).squeeze(2)
        sample_zx = voxels_coarse[..., [2, 0]]
        tpv_wz_vox = F.grid_sample(
            tpv_wz, sample_zx, padding_mode='border',
            align_corners=False).squeeze(2)

        fused = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox  # (B, C, N)
        fused = fused.permute(0, 2, 1)                # (B, N, C)

        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
            logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        else:
            fused = self.decoder(fused)
            logits = self.classifier(fused)          # (B, N, nbr_classes)

        logits = logits.permute(0, 2, 1)             # (B, nbr_classes, N)
        W = int(self.grid_size_occ[0] / self.coarse_ratio)
        H = int(self.grid_size_occ[1] / self.coarse_ratio)
        D = int(self.grid_size_occ[2] / self.coarse_ratio)
        logits = logits.reshape(bs, self.classes, W, H, D)
        return logits
