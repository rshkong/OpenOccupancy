_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
# ============================================================
# FLC-PointOcc Bypass-Only (Experiment B)
#
# Purpose: run FLCPointOccNet with `debug_camera_only_bypass=True`, but
# strip every LiDAR-side module from the config so nothing is built.
# Training recipe is cloned from FLC-step2 (samples_per_gpu=5, 30 epochs,
# img_backbone lr_mult=0.1, etc.). The ONLY structural difference from
# FLC-step2 is that the detector class is FLCPointOccNet rather than OccNet.
#
# Reading the result:
#   - bypass-B ≈ FLC-step2  → camera path through FLCPointOccNet is intact;
#     the earlier bypass gap was caused purely by the bs=1 + SyncBN recipe.
#   - bypass-B still ≪ FLC-step2 → something inside FLCPointOccNet.extract_feat
#     still degrades the camera path even with the cam_bev short-circuit.
#
# Requires detectors/flc_pointocc_net.py to skip LiDAR-branch module build
# when debug_camera_only_bypass=True (so lidar_* / *_cfg=None is accepted).
# ============================================================

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "projects/occ_plugin/"
occ_path = "./data/nuScenes-Occupancy"
depth_gt_path = './data/depth_gt'
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]   # coarse grid: 128x128x10
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
empty_idx = 0
num_cls = 17
visible_mask = False
img_norm_cfg = None

# CONet fine branch — disabled
cascade_ratio = 4
sample_from_voxel = False
sample_from_img = False

dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

numC_Trans = 64
Dz = 10
bev_channels = numC_Trans * Dz          # 640

bev_num_channels = [128, 256, 512]
fpn_in_channels = bev_num_channels[0] + bev_num_channels[2]   # 640
voxel_out_channel = 256

c2h_conv_out_dim = 256
c2h_hidden_dim = 512

model = dict(
    type='FLCPointOccNet',
    loss_norm=True,
    # ---- Bypass switches ----
    # Full camera-only bypass: extract_feat short-circuits cam_bev → occ_encoder.
    # Constructor also skips building every LiDAR / adapter / fuse module.
    debug_zero_lidar=False,
    debug_camera_only_bypass=True,
    # ---- Camera branch (identical to FLC-step2) ----
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLSSFlash',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_depth_weight=3.,
        loss_depth_type='kld',
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False),
    # ---- LiDAR branch / adapters / fuse: all None (not built) ----
    lidar_tokenizer=None,
    lidar_backbone=None,
    lidar_neck=None,
    tpv_fuser=None,
    cam_adapter_cfg=None,
    lidar_adapter_cfg=None,
    fuse_conv_cfg=None,
    # Keep channel bookkeeping consistent even though nothing is built.
    lidar_proj_in=192,
    lidar_proj_out=64,
    lidar_Dz=Dz,
    fused_channels=bev_channels,
    # Cylindrical grid config is unused under bypass; keep defaults for safety.
    cyl_grid_size=[240, 180, 16],
    cyl_min_bound=[0.0, -3.14159265, -5.0],
    cyl_max_bound=[50.0, 3.14159265, 3.0],
    occ_grid_size=[128, 128, 10],
    occ_coarse_ratio=1,
    # ---- 2D Encoder backbone (identical to FLC-step2) ----
    occ_encoder_backbone=dict(
        type='CustomResNet2D',
        numC_input=bev_channels,
        num_layer=[2, 2, 2],
        num_channels=bev_num_channels,
        stride=[2, 2, 2],
        backbone_output_ids=[0, 1, 2],
        norm_cfg=dict(type='BN', requires_grad=True),
        with_cp=True,
    ),
    occ_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=fpn_in_channels,
        out_channels=voxel_out_channel,
        scale_factor=4,
        input_feature_index=(0, 2),
        norm_cfg=dict(type='BN', requires_grad=True),
        extra_upsample=2,
    ),
    # ---- FLC OccHead (identical to FLC-step2) ----
    pts_bbox_head=dict(
        type='FLCOccHead',
        in_channels=voxel_out_channel,
        out_channel=num_cls,
        Dz=Dz,
        conv_out_dim=c2h_conv_out_dim,
        hidden_dim=c2h_hidden_dim,
        norm_cfg_2d=dict(type='BN'),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=1,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    empty_idx=empty_idx,
)

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# Pipeline matches FLC-step2 exactly: no LoadPointsFromFile / MultiSweeps,
# Collect3D keys drop 'points'.
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config,
         sequential=False, aligned=True, trans_only=False, depth_gt_path=depth_gt_path,
         mmlabnorm=True, load_depth=True, img_norm_cfg=img_norm_cfg),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf=bda_aug_conf, classes=class_names, input_modality=input_modality),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path,
         grid_size=occ_size, use_vel=False, unoccupied=empty_idx,
         pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config,
         depth_gt_path=depth_gt_path, sequential=False, aligned=True,
         trans_only=False, mmlabnorm=True, img_norm_cfg=img_norm_cfg),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf=bda_aug_conf, classes=class_names,
         input_modality=input_modality, is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True, occ_path=occ_path,
         grid_size=occ_size, use_vel=False, unoccupied=empty_idx,
         pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]

test_config = dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

train_config = dict(
    type=dataset_type,
    data_root=data_root,
    occ_root=occ_path,
    ann_file=train_ann_file,
    pipeline=train_pipeline,
    classes=class_names,
    modality=input_modality,
    test_mode=False,
    use_valid_flag=True,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    box_type_3d='LiDAR')

# Same batch size as FLC-step2. With LiDAR modules stripped, VRAM matches step2.
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=2,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(
        custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01)

optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=1,
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=30)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]
