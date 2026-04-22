"""FLCPointOccNet: dual-branch Camera (FLC) + LiDAR (PointOcc TPV) fusion.

Architecture
------------
Camera branch:
    img → img_backbone → img_neck → ViewTransformerLSSFlash
    → cam_bev [B, C*Dz, X, Y]  (e.g. [B, 640, 128, 128])
    → cam_adapter  (Conv2d)
    → [B, cam_out, X, Y]

LiDAR branch:
    points → CylinderEncoder → TPVSwin → TPVFPN → tpv_list
    → TPVFuser (grid_sample fusion)
    → lidar_3d [B, C_tpv, X, Y, Z]  (e.g. [B, 192, 128, 128, 10])
    → lidar_voxel_proj (Linear on C dim)
    → flatten Z
    → lidar_bev [B, C_proj*Z, X, Y]
    → lidar_adapter (Conv2d)
    → [B, lidar_out, X, Y]

Fusion:
    cat([cam_feat, lidar_feat], dim=1)
    → fuse_conv (Conv2d + BN + ReLU)
    → [B, fused_out, X, Y]
    → occ_encoder_backbone → occ_encoder_neck → FLCOccHead
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import time

from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16, force_fp32
from mmcv.cnn import ConvModule
from mmdet3d.models import builder

from .occnet import OccNet


@DETECTORS.register_module()
class FLCPointOccNet(OccNet):
    """Dual-branch fusion detector: FLC camera BEV + PointOcc LiDAR TPV."""

    def __init__(self,
                 # LiDAR branch
                 lidar_tokenizer=None,
                 lidar_backbone=None,
                 lidar_neck=None,
                 tpv_fuser=None,
                 # Channel adapters
                 lidar_proj_in=192,
                 lidar_proj_out=64,
                 lidar_Dz=10,
                 cam_adapter_cfg=None,
                 lidar_adapter_cfg=None,
                 fuse_conv_cfg=None,
                 fused_channels=640,
                 # Cylindrical grid params for data pipeline
                 cyl_grid_size=None,
                 cyl_min_bound=None,
                 cyl_max_bound=None,
                 occ_grid_size=None,
                 occ_coarse_ratio=1,
                 # Debug flags
                 debug_zero_lidar=False,
                 **kwargs):
        super().__init__(**kwargs)

        # When True, zeros lidar_feat before fusion → degrades to pure camera (FlashOcc).
        # Use to verify fusion path has no bug before debugging fusion quality.
        self.debug_zero_lidar = debug_zero_lidar

        # ---- LiDAR feature extraction ----
        self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
        self.lidar_backbone = builder.build_backbone(lidar_backbone)
        self.lidar_neck = builder.build_neck(lidar_neck)
        self.tpv_fuser = builder.build_fusion_layer(tpv_fuser)

        # ---- Channel projection: reduce LiDAR C before Z-flatten ----
        self.lidar_proj_in = lidar_proj_in
        self.lidar_proj_out = lidar_proj_out
        self.lidar_Dz = lidar_Dz
        self.lidar_voxel_proj = nn.Sequential(
            nn.Linear(lidar_proj_in, lidar_proj_out),
            nn.ReLU(inplace=True),
        )

        # ---- Adapters ----
        lidar_bev_channels = lidar_proj_out * lidar_Dz  # e.g. 64*10=640
        if cam_adapter_cfg is not None:
            self.cam_adapter = ConvModule(**cam_adapter_cfg)
            cam_out = cam_adapter_cfg['out_channels']
        else:
            self.cam_adapter = None
            cam_out = fused_channels

        if lidar_adapter_cfg is not None:
            self.lidar_adapter = ConvModule(**lidar_adapter_cfg)
            lidar_out = lidar_adapter_cfg['out_channels']
        else:
            self.lidar_adapter = nn.Identity()
            lidar_out = lidar_bev_channels

        # ---- Fusion conv: cat → project to backbone input channels ----
        if fuse_conv_cfg is not None:
            self.fuse_conv = ConvModule(**fuse_conv_cfg)
        else:
            self.fuse_conv = ConvModule(
                cam_out + lidar_out,
                fused_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'),
            )

        # ---- Cylindrical grid params (for voxels_coarse generation) ----
        self.cyl_grid_size = np.array(cyl_grid_size) if cyl_grid_size is not None \
            else np.array([480, 360, 32])
        self.cyl_min_bound = np.array(cyl_min_bound) if cyl_min_bound is not None \
            else np.array([0.0, -np.pi, -5.0])
        self.cyl_max_bound = np.array(cyl_max_bound) if cyl_max_bound is not None \
            else np.array([50.0, np.pi, 3.0])
        self.occ_grid_size = np.array(occ_grid_size) if occ_grid_size is not None \
            else np.array([128, 128, 10])
        self.occ_coarse_ratio = occ_coarse_ratio

    # ------------------------------------------------------------------
    # LiDAR feature extraction
    # ------------------------------------------------------------------
    def extract_lidar_tpv(self, points, grid_ind):
        """CylinderEncoder → Swin → FPN → tpv_list (3 planes)."""
        x_3view = self.lidar_tokenizer(points, grid_ind)
        tpv_list = []
        x_tpv = self.lidar_backbone(x_3view)  # list of 3 lists
        for x in x_tpv:
            x = self.lidar_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            tpv_list.append(x)
        return tpv_list

    # ------------------------------------------------------------------
    # Override extract_feat for dual-branch
    # ------------------------------------------------------------------
    def extract_feat(self, points, img, img_metas):
        """Dual-branch feature extraction."""
        # ---- Camera branch ----
        img_voxel_feats = None
        depth = None
        img_feats = None
        if img is not None:
            if self.record_time:
                torch.cuda.synchronize()
                t0 = time.time()

            img_enc_feats = self.image_encoder(img[0])
            x = img_enc_feats['x']
            img_feats = img_enc_feats['img_feats']

            if self.record_time:
                torch.cuda.synchronize()
                t1 = time.time()
                self.time_stats['img_encoder'].append(t1 - t0)

            rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
            mlp_input = self.img_view_transformer.get_mlp_input(
                rots, trans, intrins, post_rots, post_trans, bda)
            geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
            img_voxel_feats, depth = self.img_view_transformer([x] + geo_inputs)

            if self.record_time:
                torch.cuda.synchronize()
                t2 = time.time()
                self.time_stats['view_transformer'].append(t2 - t1)

        # cam_bev: [B, C*Dz, X, Y]
        cam_bev = img_voxel_feats
        if self.cam_adapter is not None:
            cam_feat = self.cam_adapter(cam_bev)
        else:
            cam_feat = cam_bev

        # ---- LiDAR branch ----
        if self.record_time:
            torch.cuda.synchronize()
            t3 = time.time()

        pts_feats = None
        if points is not None:
            grid_ind, voxels_coarse = self._prepare_lidar_inputs(points)
            tpv_list = self.extract_lidar_tpv(points, grid_ind)
            lidar_3d = self.tpv_fuser(tpv_list, voxels_coarse)
            # lidar_3d: [B, C_tpv, X, Y, Z]

            # Channel projection: [B, C_tpv, X, Y, Z] → [B, C_proj, X, Y, Z]
            B, C, X, Y, Z = lidar_3d.shape
            lidar_3d = lidar_3d.permute(0, 2, 3, 4, 1).contiguous()  # [B,X,Y,Z,C]
            lidar_3d = self.lidar_voxel_proj(lidar_3d)  # [B,X,Y,Z,C_proj]
            lidar_3d = lidar_3d.permute(0, 4, 1, 2, 3).contiguous()  # [B,C_proj,X,Y,Z]

            # Flatten Z: [B, C_proj, X, Y, Z] → [B, C_proj*Z, X, Y]
            B, C_proj, X, Y, Z = lidar_3d.shape
            lidar_bev = lidar_3d.permute(0, 1, 4, 2, 3).contiguous()  # [B,C_proj,Z,X,Y]
            lidar_bev = lidar_bev.reshape(B, C_proj * Z, X, Y)

            lidar_feat = self.lidar_adapter(lidar_bev)
        else:
            # Camera-only fallback: zero lidar features
            lidar_feat = torch.zeros_like(cam_feat)

        # === Step 1 degrade switch: zero LiDAR → pure FlashOcc behaviour ===
        if self.debug_zero_lidar:
            lidar_feat = torch.zeros_like(lidar_feat)

        if self.record_time:
            torch.cuda.synchronize()
            t4 = time.time()
            self.time_stats['lidar_branch'].append(t4 - t3)

        # ---- Fusion ----
        fused = torch.cat([cam_feat, lidar_feat], dim=1)
        fused = self.fuse_conv(fused)  # [B, fused_channels, X, Y]

        if self.record_time:
            torch.cuda.synchronize()
            t5 = time.time()
            self.time_stats['fusion'].append(t5 - t4)

        # ---- Encoder ----
        voxel_feats_enc = self.occ_encoder(fused)
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        if self.record_time:
            torch.cuda.synchronize()
            t6 = time.time()
            self.time_stats['occ_encoder'].append(t6 - t5)

        return (voxel_feats_enc, img_feats, pts_feats, depth)

    # ------------------------------------------------------------------
    # Point cloud preprocessing (cylindrical grid)
    # ------------------------------------------------------------------
    def _cart2polar(self, xyz):
        """Cartesian [x,y,z] → cylindrical [rho, phi, z]."""
        rho = torch.sqrt(xyz[:, :, 0] ** 2 + xyz[:, :, 1] ** 2)
        phi = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])
        return torch.stack([rho, phi, xyz[:, :, 2]], dim=-1)

    def _prepare_lidar_inputs(self, points):
        """Generate cylindrical grid_ind and voxels_coarse from raw points.

        Args:
            points: list of (N_i, 5) tensors [x, y, z, intensity, ring]
                    or (B, N, 5) padded tensor

        Returns:
            grid_ind: list of B tensors, each (N_i, 3) int32
            voxels_coarse: (B, M, 3) float — cylindrical coords of
                           coarse Cartesian voxel centres
        """
        device = points[0].device if isinstance(points, list) else points.device
        min_bound = torch.tensor(self.cyl_min_bound, dtype=torch.float32, device=device)
        max_bound = torch.tensor(self.cyl_max_bound, dtype=torch.float32, device=device)
        cyl_grid = torch.tensor(self.cyl_grid_size, dtype=torch.float32, device=device)
        intervals = (max_bound - min_bound) / cyl_grid

        # ---- grid_ind: per-point cylindrical voxel index ----
        grid_ind_list = []
        point_feats_list = []

        if isinstance(points, list):
            batch_points = points
        else:
            # Padded tensor: split by batch
            batch_points = [points[b] for b in range(points.shape[0])]

        for pts in batch_points:
            if pts.ndim == 1:
                pts = pts.unsqueeze(0)
            xyz = pts[:, :3]  # (N, 3)
            feat = pts[:, 3:]  # (N, >=1) intensity [+ ring]
            # Pad to 2 columns if ring index is missing
            if feat.shape[1] < 2:
                feat = torch.cat([feat, torch.zeros(feat.shape[0], 2 - feat.shape[1], device=feat.device)], dim=1)

            # Convert to cylindrical
            rho = torch.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
            phi = torch.atan2(xyz[:, 1], xyz[:, 0])
            xyz_pol = torch.stack([rho, phi, xyz[:, 2]], dim=-1)  # (N, 3)

            # Clip and quantize
            xyz_pol_clipped = torch.clamp(
                xyz_pol,
                min=min_bound,
                max=max_bound - 1e-3)
            gi = torch.floor(
                (xyz_pol_clipped - min_bound) / intervals).to(torch.int32)  # (N, 3)
            grid_ind_list.append(gi)

            # Build 10-channel point features
            voxel_centers = (gi.float() + 0.5) * intervals + min_bound
            return_xyz = xyz_pol - voxel_centers  # centred polar coords
            point_feat = torch.cat([
                return_xyz,     # 3: centred rho, phi, z
                xyz_pol,        # 3: absolute rho, phi, z
                xyz[:, :2],     # 2: absolute x, y
                feat,           # 2: intensity, ring
            ], dim=-1)  # (N, 10)
            point_feats_list.append(point_feat)

        # Stack point features into (B, N_max, 10) padded tensor
        B = len(point_feats_list)
        max_n = max(pf.shape[0] for pf in point_feats_list)
        points_padded = torch.zeros(B, max_n, 10, device=device)
        for i, pf in enumerate(point_feats_list):
            points_padded[i, :pf.shape[0]] = pf

        # ---- voxels_coarse: Cartesian voxel centres → cylindrical coords ----
        occ_grid = self.occ_grid_size // self.occ_coarse_ratio
        pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        voxel_size = (pc_range[3:] - pc_range[:3]) / occ_grid

        # Generate coarse voxel centres in Cartesian
        idx = np.indices(occ_grid)  # (3, W, H, D)
        voxel_centres = (idx + 0.5) * voxel_size.reshape(3, 1, 1, 1) \
            + pc_range[:3].reshape(3, 1, 1, 1)
        voxel_centres = voxel_centres.reshape(3, -1).T  # (M, 3) in [x, y, z]

        # Convert to cylindrical and normalise to grid coords
        vc = torch.tensor(voxel_centres, dtype=torch.float32, device=device)
        rho = torch.sqrt(vc[:, 0] ** 2 + vc[:, 1] ** 2)
        phi = torch.atan2(vc[:, 1], vc[:, 0])
        vc_pol = torch.stack([rho, phi, vc[:, 2]], dim=-1)

        # Normalise to cylindrical grid coordinates (continuous float index)
        vc_pol_clipped = torch.clamp(vc_pol, min=min_bound, max=max_bound - 1e-3)
        vc_grid = (vc_pol_clipped - min_bound) / intervals  # (M, 3)

        # Expand to batch
        voxels_coarse = vc_grid.unsqueeze(0).expand(B, -1, -1)  # (B, M, 3)

        # Replace points tensor for CylinderEncoder
        # Override the points input with our 10-channel features
        self._lidar_points_10ch = points_padded

        return grid_ind_list, voxels_coarse

    def extract_lidar_tpv(self, points, grid_ind):
        """Override to use pre-computed 10ch features."""
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

    # ------------------------------------------------------------------
    # forward_train / forward_test: reuse OccNet's, which calls extract_feat
    # ------------------------------------------------------------------
