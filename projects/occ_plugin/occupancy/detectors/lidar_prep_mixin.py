"""LidarPrepMixin: shared LiDAR preprocessing for PointOcc-style detectors.

Both `FLCPointOccNet` (camera + LiDAR fusion) and `PointOccNet` (LiDAR only)
need to:
  1. Convert raw LiDAR points into the 10-channel CylinderEncoder input.
  2. Compute per-point cylindrical grid indices (`grid_ind`).
  3. Compute Cartesian coarse voxel centres rescaled into TPV sampling space
     (`voxels_coarse`) for grid_sample.

The mixin expects the subclass to expose:
  - self.cyl_min_bound, self.cyl_max_bound  — np.array of length 3
  - self.cyl_grid_size                      — np.array of length 3
  - self.occ_grid_size                      — np.array of length 3
  - self.occ_coarse_ratio                   — int
  - self.pc_range                           — np.array (optional; default nuScenes)
  - self.lidar_tokenizer, self.lidar_backbone, self.lidar_neck  — built modules
"""
import numpy as np
import torch


class LidarPrepMixin:
    """Preprocessing + TPV extraction helpers for PointOcc-style LiDAR branches."""

    # Default nuScenes point cloud range; a subclass can override by setting
    # self.pc_range explicitly in __init__.
    _DEFAULT_PC_RANGE = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    def _cart2polar(self, xyz):
        """Cartesian [x,y,z] → cylindrical [rho, phi, z]."""
        rho = torch.sqrt(xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2)
        phi = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])
        return torch.stack([rho, phi, xyz[:, :, 2]], dim=-1)

    def _prepare_lidar_inputs(self, points, tpv_norm_shape=None):
        """Raw points → CylinderEncoder inputs.

        Args:
            points: list of (N_i, >=4) tensors [x, y, z, intensity, (ring)]
                    or (B, N, >=4) padded tensor.
            tpv_norm_shape: optional (3,) iterable [W_tpv*scale_w, H_tpv*scale_h,
                Z_tpv*scale_z]. When provided, `voxels_coarse` is scaled to
                [0, tpv_norm_shape) so the downstream grid_sample normalization
                lands in [-1, 1).  When None, falls back to grid-index space
                [0, cyl_grid_size) — only correct when cyl_grid_size already
                equals tpv_norm_shape.

        Returns:
            grid_ind: list of B tensors, each (N_i, 3) int32 — cylindrical
                      voxel index per point.
            voxels_coarse: (B, M, 3) float — cylindrical grid coords of every
                      coarse Cartesian voxel centre.

        Side effects:
            Stores 10-channel padded point feature tensor on
            `self._lidar_points_10ch` for `extract_lidar_tpv` to consume.
        """
        device = points[0].device if isinstance(points, list) else points.device
        min_bound = torch.tensor(self.cyl_min_bound, dtype=torch.float32, device=device)
        max_bound = torch.tensor(self.cyl_max_bound, dtype=torch.float32, device=device)
        cyl_grid = torch.tensor(self.cyl_grid_size, dtype=torch.float32, device=device)
        intervals = (max_bound - min_bound) / cyl_grid

        grid_ind_list = []
        point_feats_list = []

        if isinstance(points, list):
            batch_points = points
        else:
            batch_points = [points[b] for b in range(points.shape[0])]

        for pts in batch_points:
            if pts.ndim == 1:
                pts = pts.unsqueeze(0)
            xyz = pts[:, :3]
            feat = pts[:, 3:]  # (N, >=1) intensity [+ ring]
            if feat.shape[1] < 2:
                # LoadPointsFromMultiSweeps may strip ring — pad with zeros.
                feat = torch.cat([
                    feat,
                    torch.zeros(feat.shape[0], 2 - feat.shape[1], device=feat.device),
                ], dim=1)

            rho = torch.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
            phi = torch.atan2(xyz[:, 1], xyz[:, 0])
            xyz_pol = torch.stack([rho, phi, xyz[:, 2]], dim=-1)

            xyz_pol_clipped = torch.clamp(xyz_pol, min=min_bound, max=max_bound - 1e-3)
            gi = torch.floor(
                (xyz_pol_clipped - min_bound) / intervals).to(torch.int32)
            grid_ind_list.append(gi)

            voxel_centers = (gi.float() + 0.5) * intervals + min_bound
            return_xyz = xyz_pol - voxel_centers  # centred polar
            point_feat = torch.cat([
                return_xyz,
                xyz_pol,
                xyz[:, :2],
                feat,
            ], dim=-1)  # (N, 10)
            point_feats_list.append(point_feat)

        B = len(point_feats_list)
        max_n = max(pf.shape[0] for pf in point_feats_list)
        points_padded = torch.zeros(B, max_n, 10, device=device)
        for i, pf in enumerate(point_feats_list):
            points_padded[i, :pf.shape[0]] = pf
        self._lidar_points_10ch = points_padded

        # Build coarse Cartesian voxel centres → cylindrical coords.
        pc_range = np.asarray(getattr(self, 'pc_range', self._DEFAULT_PC_RANGE))
        occ_grid = self.occ_grid_size // self.occ_coarse_ratio
        voxel_size = (pc_range[3:] - pc_range[:3]) / occ_grid

        idx = np.indices(occ_grid)
        voxel_centres = (idx + 0.5) * voxel_size.reshape(3, 1, 1, 1) \
            + pc_range[:3].reshape(3, 1, 1, 1)
        voxel_centres = voxel_centres.reshape(3, -1).T  # (M, 3)

        vc = torch.tensor(voxel_centres, dtype=torch.float32, device=device)
        rho = torch.sqrt(vc[:, 0] ** 2 + vc[:, 1] ** 2)
        phi = torch.atan2(vc[:, 1], vc[:, 0])
        vc_pol = torch.stack([rho, phi, vc[:, 2]], dim=-1)

        vc_pol_clipped = torch.clamp(vc_pol, min=min_bound, max=max_bound - 1e-3)
        if tpv_norm_shape is not None:
            target = torch.tensor(
                list(tpv_norm_shape), dtype=torch.float32, device=device)
            vc_grid = (vc_pol_clipped - min_bound) \
                / (max_bound - min_bound) * target
        else:
            vc_grid = (vc_pol_clipped - min_bound) / intervals

        voxels_coarse = vc_grid.unsqueeze(0).expand(B, -1, -1)
        return grid_ind_list, voxels_coarse

    def extract_lidar_tpv(self, grid_ind):
        """CylinderEncoder → Swin → FPN → tpv_list (3 planes).

        Args:
            grid_ind: list of per-batch cylindrical index tensors (from
                _prepare_lidar_inputs).  The 10-channel point features are
                read from self._lidar_points_10ch.

        Returns:
            list of 3 TPV feature maps [tpv_xy, tpv_yz, tpv_zx].
        """
        points_10ch = self._lidar_points_10ch
        x_3view = self.lidar_tokenizer(points_10ch, grid_ind)
        tpv_list = []
        x_tpv = self.lidar_backbone(x_3view)
        for x in x_tpv:
            x = self.lidar_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            tpv_list.append(x)
        return tpv_list
