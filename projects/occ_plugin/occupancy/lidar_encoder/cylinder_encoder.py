"""CylinderEncoder ported from PointOcc to OpenOccupancy (mmdet2).

Converts raw LiDAR points in cylindrical voxel space into three TPV planes
via sparse max-pooling along each axis.

Original: PointOcc/model/cylinder_encoder.py  (CylinderEncoder_Occ)
Changes:
  - Registry: mmdet3d.registry.MODELS -> mmdet3d.models.builder.BACKBONES
  - BaseModule: mmengine.model -> mmcv.runner
  - Kept API identical: forward(points, grid_ind) -> [tpv_xy, tpv_yz, tpv_zx]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import numpy as np
from spconv.pytorch import SparseConvTensor, SparseMaxPool3d
from mmcv.runner import BaseModule
from mmdet3d.models.builder import BACKBONES


@BACKBONES.register_module()
class CylinderEncoder(BaseModule):

    def __init__(self, grid_size, in_channels=10, out_channels=256,
                 fea_compre=None, base_channels=128, split=None,
                 track_running_stats=False):
        super().__init__()
        if split is None:
            split = [8, 8, 8]

        self.fea_compre = fea_compre
        self.grid_size = grid_size
        self.split = split

        # point-wise MLP
        self.point_mlp = nn.Sequential(
            nn.BatchNorm1d(in_channels, track_running_stats=track_running_stats),
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, track_running_stats=track_running_stats),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )

        # optional feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(out_channels, fea_compre),
                nn.ReLU(),
            )

        # sparse max-pooling along each axis to produce TPV planes
        self.pool_xy = SparseMaxPool3d(
            kernel_size=[1, 1, int(self.grid_size[2] / split[2])],
            stride=[1, 1, int(self.grid_size[2] / split[2])], padding=0)
        self.pool_yz = SparseMaxPool3d(
            kernel_size=[int(self.grid_size[0] / split[0]), 1, 1],
            stride=[int(self.grid_size[0] / split[0]), 1, 1], padding=0)
        self.pool_zx = SparseMaxPool3d(
            kernel_size=[1, int(self.grid_size[1] / split[1]), 1],
            stride=[1, int(self.grid_size[1] / split[1]), 1], padding=0)

        # per-plane MLP: collapse the split dim and project
        in_ch = [int(base_channels * s) for s in split]
        out_ch = [int(base_channels) for _ in split]
        self.mlp_xy = nn.Sequential(
            nn.Linear(in_ch[2], out_ch[2]), nn.ReLU(),
            nn.Linear(out_ch[2], out_ch[2]))
        self.mlp_yz = nn.Sequential(
            nn.Linear(in_ch[0], out_ch[0]), nn.ReLU(),
            nn.Linear(out_ch[0], out_ch[0]))
        self.mlp_zx = nn.Sequential(
            nn.Linear(in_ch[1], out_ch[1]), nn.ReLU(),
            nn.Linear(out_ch[1], out_ch[1]))

    def forward(self, points, grid_ind):
        """
        Args:
            points: (B, N, 10) raw point features in cylindrical space
            grid_ind: list[Tensor] of length B, each (N_i, 3) int32
                      cylindrical grid indices [rho_idx, phi_idx, z_idx]

        Returns:
            list of 3 TPV planes:
                tpv_xy: (B, C, grid_size[0], grid_size[1])
                tpv_yz: (B, C, grid_size[1], grid_size[2])
                tpv_zx: (B, C, grid_size[2], grid_size[0])
        """
        device = points.device

        # Prepend batch index to grid_ind
        cat_pt_ind = []
        for i_batch in range(len(grid_ind)):
            cat_pt_ind.append(
                F.pad(grid_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = points.reshape(-1, points.shape[-1])
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # Shuffle
        shuffled_ind = torch.randperm(pt_num, device=device)
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # Unique voxels and scatter-max
        unq, unq_inv, unq_cnt = torch.unique(
            cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        processed_cat_pt_fea = self.point_mlp(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(
            processed_cat_pt_fea, unq_inv, dim=0)[0]

        if self.fea_compre is not None:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        # Build sparse tensor and pool along each axis
        coors = unq.int()
        batch_size = coors[-1][0] + 1
        ret = SparseConvTensor(
            processed_pooled_data, coors,
            np.array(self.grid_size), batch_size)

        # Pool Z  -> XY plane: [B, C, rho, phi, split_z] -> flatten -> MLP
        tpv_xy = self.mlp_xy(
            self.pool_xy(ret).dense()
            .permute(0, 2, 3, 4, 1).flatten(start_dim=3)
        ).permute(0, 3, 1, 2)

        # Pool rho -> YZ plane: [B, C, split_rho, phi, z] -> flatten -> MLP
        tpv_yz = self.mlp_yz(
            self.pool_yz(ret).dense()
            .permute(0, 3, 4, 2, 1).flatten(start_dim=3)
        ).permute(0, 3, 1, 2)

        # Pool phi -> ZX plane: [B, C, rho, split_phi, z] -> flatten -> MLP
        tpv_zx = self.mlp_zx(
            self.pool_zx(ret).dense()
            .permute(0, 4, 2, 3, 1).flatten(start_dim=3)
        ).permute(0, 3, 1, 2)

        return [tpv_xy, tpv_yz, tpv_zx]
