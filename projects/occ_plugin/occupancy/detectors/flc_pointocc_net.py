"""FLCPointOccNet: dual-branch Camera (FLC) + LiDAR (PointOcc TPV) fusion.

Architecture
------------
Camera branch:
    img → img_backbone → img_neck → ViewTransformerLSSFlash
    → cam_bev [B, C*Dz, X, Y]  (e.g. [B, 640, 128, 128])
    → cam_adapter  (Conv2d, optional)
    → [B, cam_out, X, Y]

LiDAR branch:
    points → CylinderEncoder → TPVSwin → TPVFPN → tpv_list
    → TPVFuser (grid_sample fusion)
    → lidar_3d [B, C_tpv, X, Y, Z]  (e.g. [B, 192, 128, 128, 10])
    → flatten Z into channels
    → lidar_bev [B, C_tpv*Z, X, Y]   (e.g. [B, 1920, 128, 128])
    → lidar_adapter (Conv2d, required)
    → [B, lidar_out, X, Y]

Fusion:
    cat([cam_feat, lidar_feat], dim=1)
    → fuse_conv (Conv2d + BN + ReLU, BEVFusion-style 3x3)
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
from .lidar_prep_mixin import LidarPrepMixin


@DETECTORS.register_module()
class FLCPointOccNet(LidarPrepMixin, OccNet):
    """Dual-branch fusion detector: FLC camera BEV + PointOcc LiDAR TPV."""

    def __init__(self,
                 # LiDAR branch
                 lidar_tokenizer=None,
                 lidar_backbone=None,
                 lidar_neck=None,
                 tpv_fuser=None,
                 # Channel bookkeeping (used by lidar_adapter_cfg.in_channels)
                 lidar_proj_in=192,
                 lidar_proj_out=None,  # deprecated: ignored, kept for old-config compat
                 lidar_Dz=10,
                 cam_adapter_cfg=None,
                 lidar_adapter_cfg=None,
                 fuse_conv_cfg=None,
                 fused_channels=256,
                 # Z-flatten strategy:
                 #   True  (default) — flatten-first: reshape [B,C,X,Y,Z]→[B,C*Z,X,Y]
                 #                     then Conv1x1; each Z bin gets own weights (~246K).
                 #   False — project-first: apply a shared Linear C→proj_ch per Z bin
                 #                     then flatten; all Z bins share the same weight (~12K).
                 #                     lidar_adapter_cfg.in_channels must equal
                 #                     lidar_adapter_proj_ch * lidar_Dz.
                 lidar_z_flatten_first=True,
                 # Number of channels to project to per Z bin when
                 # lidar_z_flatten_first=False.  Ignored when True.
                 lidar_z_proj_ch=64,
                 # Cylindrical grid params for data pipeline
                 cyl_grid_size=None,
                 cyl_min_bound=None,
                 cyl_max_bound=None,
                 occ_grid_size=None,
                 occ_coarse_ratio=1,
                 pc_range=None,
                 # Debug flags
                 debug_zero_lidar=False,
                 debug_camera_only_bypass=False,
                 **kwargs):
        super().__init__(**kwargs)

        # debug_zero_lidar: zeros lidar_feat AFTER cam_adapter/lidar_adapter, still
        #   runs both adapters and fuse_conv. Tests "does the fusion path carry
        #   usable camera signal when LiDAR is neutralized."
        # debug_camera_only_bypass: skips cam_adapter / LiDAR branch / fuse_conv
        #   entirely; feeds cam_bev straight to the 2D encoder. True FLC-step2 path.
        #   Tests "is the camera branch itself intact, independent of the
        #   adapters." When both flags are True, bypass takes precedence.
        self.debug_zero_lidar = debug_zero_lidar
        self.debug_camera_only_bypass = debug_camera_only_bypass
        self.lidar_z_flatten_first = lidar_z_flatten_first

        if debug_camera_only_bypass:
            # Skip all fusion-path modules; extract_feat short-circuits
            # cam_bev → occ_encoder. Lets the config omit LiDAR cfgs entirely
            # and frees VRAM so samples_per_gpu can match step2.
            self.lidar_tokenizer = None
            self.lidar_backbone = None
            self.lidar_neck = None
            self.tpv_fuser = None
            self.cam_adapter = None
            self.lidar_adapter = None
            self.fuse_conv = None
        else:
            # ---- LiDAR feature extraction ----
            self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
            self.lidar_backbone = builder.build_backbone(lidar_backbone)
            self.lidar_neck = builder.build_neck(lidar_neck)
            self.tpv_fuser = builder.build_fusion_layer(tpv_fuser)

            # Channel/Z bookkeeping
            self.lidar_proj_in = lidar_proj_in
            self.lidar_Dz = lidar_Dz

            # Project-first path: one Linear shared across all Z bins.
            # lidar_adapter_cfg.in_channels should be lidar_z_proj_ch * lidar_Dz.
            if not lidar_z_flatten_first:
                self.lidar_z_proj = nn.Linear(lidar_proj_in, lidar_z_proj_ch, bias=False)
            else:
                self.lidar_z_proj = None

            # ---- Camera adapter (optional) ----
            if cam_adapter_cfg is not None:
                self.cam_adapter = ConvModule(**cam_adapter_cfg)
            else:
                self.cam_adapter = None

            # ---- LiDAR adapter (required: must collapse 1920ch lidar_bev) ----
            if lidar_adapter_cfg is None:
                raise ValueError(
                    'lidar_adapter_cfg is required when debug_camera_only_bypass=False; '
                    'lidar_bev has C_tpv*Dz channels (e.g. 1920) and must be reduced.')
            self.lidar_adapter = ConvModule(**lidar_adapter_cfg)

            # ---- Fusion conv: cat → project to backbone input channels ----
            if fuse_conv_cfg is None:
                raise ValueError(
                    'fuse_conv_cfg is required when debug_camera_only_bypass=False.')
            self.fuse_conv = ConvModule(**fuse_conv_cfg)

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
        # LidarPrepMixin reads self.pc_range to build voxels_coarse; without
        # this it silently falls back to hardcoded nuScenes defaults and any
        # config-level pc_range change would desync TPV queries from occupancy
        # grid. Mirror PointOccNet's behaviour.
        if pc_range is not None:
            self.pc_range = np.array(pc_range)

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

        # === Full camera-only bypass: exact FLC-step2 path ===
        # Skip cam_adapter / LiDAR branch / fuse_conv entirely. The 2D encoder
        # receives cam_bev with the original 640 channels, identical to what
        # OccNet gives it in the pure-camera config. Used to isolate whether
        # the camera branch itself is intact, without any fusion-path layers.
        # NOTE: unused modules (cam_adapter, fuse_conv, lidar_*) still exist
        # as parameters. For single-GPU `tools/train.py` this is fine. For DDP
        # set `find_unused_parameters=True` and `static_graph=False` in config.
        if self.debug_camera_only_bypass:
            voxel_feats_enc = self.occ_encoder(cam_bev)
            if type(voxel_feats_enc) is not list:
                voxel_feats_enc = [voxel_feats_enc]
            return (voxel_feats_enc, img_feats, None, depth)

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
            # Derive TPV sampling shape from the fuser so voxels_coarse is
            # scaled consistently with the grid_sample normalization inside.
            tpv_norm_shape = [
                int(self.tpv_fuser.tpv_w * self.tpv_fuser.scale_w),
                int(self.tpv_fuser.tpv_h * self.tpv_fuser.scale_h),
                int(self.tpv_fuser.tpv_z * self.tpv_fuser.scale_z),
            ]
            grid_ind, voxels_coarse = self._prepare_lidar_inputs(
                points, tpv_norm_shape=tpv_norm_shape)
            tpv_list = self.extract_lidar_tpv(grid_ind)
            lidar_3d = self.tpv_fuser(tpv_list, voxels_coarse)
            # lidar_3d: [B, C_tpv, X, Y, Z]

            B, C, X, Y, Z = lidar_3d.shape
            if self.lidar_z_flatten_first:
                # Default: flatten Z into channels first, then Conv1x1.
                # Each Z bin has its own projection weights (~246K params).
                lidar_bev = lidar_3d.permute(0, 1, 4, 2, 3).contiguous()  # [B,C,Z,X,Y]
                lidar_bev = lidar_bev.reshape(B, C * Z, X, Y)              # [B,C*Z,X,Y]
            else:
                # Project-first: shared Linear C→proj_ch across all Z bins (~12K params).
                # lidar_3d: [B,C,X,Y,Z] → [B*X*Y*Z, C] → Linear → [B*X*Y*Z, proj_ch]
                # → [B, proj_ch*Z, X, Y]
                lidar_bev = lidar_3d.permute(0, 2, 3, 4, 1).contiguous()  # [B,X,Y,Z,C]
                lidar_bev = self.lidar_z_proj(lidar_bev)                   # [B,X,Y,Z,proj_ch]
                lidar_bev = lidar_bev.permute(0, 4, 3, 1, 2).contiguous() # [B,proj_ch,Z,X,Y]
                lidar_bev = lidar_bev.reshape(B, -1, X, Y)                 # [B,proj_ch*Z,X,Y]

            lidar_feat = self.lidar_adapter(lidar_bev)
        else:
            # Camera-only fallback: produce zeros with lidar_adapter's output
            # channel count (e.g. 128), NOT cam_feat's channel count. cam_feat
            # is 256/640 depending on cam_adapter; cat downstream requires
            # lidar_feat to match fuse_conv.in_channels - cam_feat.shape[1].
            B, _, H, W = cam_feat.shape
            lidar_feat = cam_feat.new_zeros(
                B, self.lidar_adapter.out_channels, H, W)

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
    # forward_train / forward_test: reuse OccNet's, which calls extract_feat
    # ------------------------------------------------------------------
