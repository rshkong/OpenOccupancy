_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
# ============================================================
# LiDAR-only PointOcc reproduction on RTX 4070 Ti (12 GB).
#
# Pipeline:
#   points → CylinderEncoder → TPVSwin → TPVFPN → TPVAggregator
#           → logits [B, 17, 128, 128, 10]
#           → CE + Lovasz + sem_scal + geo_scal (weights [1,1,1,1])
#
# Differences from PointOcc's `pointtpv_nusc_occ.py`:
#   - cyl_grid_size halved from [480, 360, 32] → [240, 180, 16] for VRAM.
#   - Correspondingly tpv_(h,w,z) = [45, 60, 4] (vs. PointOcc's [90, 120, 8]).
#   - scale_(h,w,z) = 2 keeps aggregator sampling resolution at cyl_grid_size,
#     which matches the voxels_coarse range produced by _prepare_lidar_inputs.
# ============================================================

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "projects/occ_plugin/"
img_norm_cfg = None
occ_path = "./data/nuScenes-Occupancy"
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
empty_idx = 0
num_cls = 17
visible_mask = False

dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# ---- TPV branch dims ----
cyl_grid_size = [240, 180, 16]
tpv_w = cyl_grid_size[0] // 4   # 60  (Swin patch_embed stride 4)
tpv_h = cyl_grid_size[1] // 4   # 45
tpv_z = cyl_grid_size[2] // 4   # 4
tpv_C = 192                     # TPVFPN output channels per plane

# ---- Occupancy grid ----
occ_grid_size = [128, 128, 10]
occ_coarse_ratio = 1

model = dict(
    type='PointOccNet',
    # LiDAR TPV branch (reused from fusion config)
    lidar_tokenizer=dict(
        type='CylinderEncoder',
        grid_size=cyl_grid_size,
        in_channels=10,
        out_channels=128,
        fea_compre=None,
        base_channels=128,
        split=[4, 4, 4],
        track_running_stats=False,
    ),
    lidar_backbone=dict(
        type='TPVSwin',
        in_channels=128,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        patch_size=4,
        strides=[1, 2, 2, 2],
        frozen_stages=-1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=True,
        convert_weights=True,
        # pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
    ),
    lidar_neck=dict(
        type='TPVFPN',
        in_channels=[192, 384, 768],
        out_channels=tpv_C,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False),
    ),
    tpv_aggregator=dict(
        type='TPVAggregator',
        tpv_h=tpv_h,
        tpv_w=tpv_w,
        tpv_z=tpv_z,
        grid_size_occ=occ_grid_size,
        coarse_ratio=occ_coarse_ratio,
        nbr_classes=num_cls,
        in_dims=tpv_C,
        hidden_dims=tpv_C * 2,
        out_dims=tpv_C,
        scale_h=2,
        scale_w=2,
        scale_z=2,
        use_checkpoint=True,
    ),
    # Cylindrical grid config for _prepare_lidar_inputs
    cyl_grid_size=cyl_grid_size,
    cyl_min_bound=[0.0, -3.14159265, -5.0],
    cyl_max_bound=[50.0, 3.14159265, 3.0],
    occ_grid_size=occ_grid_size,
    occ_coarse_ratio=occ_coarse_ratio,
    pc_range=point_cloud_range,
    # PointOcc loss: CE + Lovasz + sem_scal + geo_scal, equal weights
    loss_weight=(1.0, 1.0, 1.0, 1.0),
    ignore_index=255,
    empty_idx=empty_idx,
    num_classes=num_cls,
)

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
         sweeps_num=10),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf=bda_aug_conf, classes=class_names,
         input_modality=input_modality),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True,
         occ_path=occ_path, grid_size=occ_size, use_vel=False,
         unoccupied=empty_idx, pc_range=point_cloud_range,
         cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_occ', 'points']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
         coord_type='LIDAR',
         load_dim=5,
         use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
         sweeps_num=10),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf=bda_aug_conf, classes=class_names,
         input_modality=input_modality, is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True,
         occ_path=occ_path, grid_size=occ_size, use_vel=False,
         unoccupied=empty_idx, pc_range=point_cloud_range,
         cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names,
         with_label=False),
    dict(type='Collect3D', keys=['gt_occ', 'points'],
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

data = dict(
    samples_per_gpu=1,
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

runner = dict(type='EpochBasedRunner', max_epochs=24)
find_unused_parameters = False
static_graph = True
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

custom_hooks = [
    # OccEfficiencyHook deepcopies the model → wastes VRAM; keep disabled.
]
