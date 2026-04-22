# FLC-PointOcc 模型文档

FLC-PointOcc 是一个双模态（Camera + LiDAR）融合架构，结合了 FlashOcc 的 2D BEV 快速预测和 PointOcc 的 Tri-Perspective View (TPV) 几何特征。核心思路是用 LiDAR 分支产生的 TPV 几何特征来补充相机分支 BEV 中深度不准确的缺陷。

配置文件：`projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py`

---

## 1. 模型结构与张量流动

### 1.1 总体架构

```text
┌─────────────────── Camera Branch ────────────────────┐
│                                                       │
│  img [B, 6, 3, 256, 704]                             │
│    → ResNet50 (img_backbone)                          │
│    → SECONDFPN (img_neck)                             │
│    → [B, 6, 512, 16, 44]                             │
│    → ViewTransformerLSSFlash                          │
│    → cam_bev [B, 640, 128, 128]                      │
│    → cam_adapter (Conv2d 1x1 + BN + ReLU)            │
│    → cam_feat [B, 320, 128, 128]                     │
│                                                       │
└────────────────────────┬──────────────────────────────┘
                         │
                         │
┌─────────────────── LiDAR Branch ─────────────────────┐
│                                                       │
│  points [B, N, 5]  (x, y, z, intensity, ring)       │
│    → cart2polar + 10-channel特征构建                   │
│    → CylinderEncoder (lidar_tokenizer)               │
│      稀疏点云 → SparseConvTensor → 三方向MaxPool      │
│    → 3个TPV平面:                                      │
│        tpv_xy [B, 128, 240, 180]                     │
│        tpv_yz [B, 128, 180, 16]                      │
│        tpv_zx [B, 128, 16, 240]                      │
│    → TPVSwin (lidar_backbone)                        │
│      3个平面共享Swin Transformer → 多尺度特征          │
│    → TPVFPN (lidar_neck)                             │
│      每个平面独立FPN → [B, 192, H, W] per plane      │
│    → TPVFuser (tpv_fuser)                            │
│      grid_sample在128x128x10坐标处采样 + 三平面求和    │
│    → lidar_3d [B, 192, 128, 128, 10]                 │
│    → Linear(192→64)  通道压缩                         │
│    → [B, 64, 128, 128, 10]                           │
│    → flatten Z (reshape)                              │
│    → [B, 640, 128, 128]                              │
│    → lidar_adapter (Conv2d 1x1 + BN + ReLU)          │
│    → lidar_feat [B, 320, 128, 128]                   │
│                                                       │
└────────────────────────┬──────────────────────────────┘
                         │
                         ↓
        cat([cam_feat, lidar_feat], dim=1)
                         ↓
              fusion_input [B, 640, 128, 128]
                         │
              fuse_conv (Conv2d 1x1 + BN + ReLU)
                         │
              fused_bev [B, 640, 128, 128]
                         │
              CustomResNet2D (occ_encoder_backbone)
                         │
              FPN_LSS (occ_encoder_neck)
                         │
              [B, 256, 128, 128]
                         │
              FLCOccHead (Conv2d + C2H MLP)
                         │
              output_voxels [B, 17, 128, 128, 10]
              # 最终语义占据 logits，不参与 cat
```

### 1.2 逐阶段详解

#### Camera 分支

| 阶段 | 模块 | 输入 → 输出 | 代码位置 |
|------|------|-------------|---------|
| 图像编码 | `ResNet50` + `SECONDFPN` | `[B,6,3,256,704]` → `[B,6,512,16,44]` | `occnet.py:image_encoder()` |
| 视角变换 | `ViewTransformerLSSFlash` | 2D图像特征 → BEV体素 → collapse Z | `image2bev/ViewTransformerLSSFlash` |
| BEV输出 | Z-collapse | `[B,64,128,128,10]` → `[B,640,128,128]` | ViewTransformer内部 |
| 通道适配 | `cam_adapter` (Conv2d 1x1) | `[B,640,128,128]` → `[B,320,128,128]` | `flc_pointocc_net.py` |

#### LiDAR 分支

| 阶段 | 模块 | 输入 → 输出 | 代码位置 |
|------|------|-------------|---------|
| 点云预处理 | `_prepare_lidar_inputs()` | 原始点 `[N,5]` → 10通道柱坐标特征 + grid_ind | `flc_pointocc_net.py` |
| 稀疏编码 | `CylinderEncoder` | 10ch特征 → 3个TPV平面 `[B,128,H,W]` | `lidar_encoder/cylinder_encoder.py` |
| 多尺度特征 | `TPVSwin` | 3个平面各过共享Swin → 多尺度特征 | `lidar_encoder/tpv_swin.py` |
| 特征融合 | `TPVFPN` | 每个平面的多尺度特征 → 单尺度 `[B,192,H,W]` | `lidar_encoder/tpv_fpn.py` |
| 3D重建 | `TPVFuser` | 3个平面 → grid_sample → 求和 → `[B,192,128,128,10]` | `lidar_encoder/tpv_fuser.py` |
| 通道压缩 | `lidar_voxel_proj` (Linear) | `[B,192,X,Y,Z]` → `[B,64,X,Y,Z]` | `flc_pointocc_net.py` |
| Z-flatten | reshape | `[B,64,128,128,10]` → `[B,640,128,128]` | `flc_pointocc_net.py` |
| 通道适配 | `lidar_adapter` (Conv2d 1x1) | `[B,640,128,128]` → `[B,320,128,128]` | `flc_pointocc_net.py` |

#### 融合 + 解码

| 阶段 | 模块 | 输入 → 输出 | 代码位置 |
|------|------|-------------|---------|
| 通道拼接 | `torch.cat` | `[B,320,X,Y]` + `[B,320,X,Y]` → `[B,640,X,Y]` | `flc_pointocc_net.py` |
| 融合卷积 | `fuse_conv` (Conv2d 1x1 + BN + ReLU) | `[B,640,X,Y]` → `[B,640,X,Y]` | `flc_pointocc_net.py` |
| 2D编码器 | `CustomResNet2D` | `[B,640,128,128]` → 多尺度 | `occnet.py` → `occ_encoder()` |
| 2D FPN | `FPN_LSS` | 多尺度 → `[B,256,128,128]` | `occnet.py` → `occ_encoder()` |
| 占据预测 | `FLCOccHead` (Conv2d + C2H MLP) | `[B,256,128,128]` → `[B,17,128,128,10]` | `dense_heads/flc_occ_head.py` |

### 1.3 10通道点云特征说明

`CylinderEncoder` 的输入是 10 通道的点特征，由 `_prepare_lidar_inputs()` 从原始 5 通道 LiDAR 点 `[x, y, z, intensity, ring]` 构建：

| 通道 | 含义 | 说明 |
|------|------|------|
| 0-2 | centred rho, phi, z | 点的柱坐标减去所在体素中心坐标 |
| 3-5 | absolute rho, phi, z | 点的绝对柱坐标 |
| 6-7 | absolute x, y | 点的笛卡尔 x, y |
| 8 | intensity | 激光反射强度 |
| 9 | ring | 激光线束编号 |

### 1.4 TPV (Tri-Perspective View) 原理

TPV 将 3D 空间分解为三个正交的 2D 平面：
- **XY 平面 (BEV)**：鸟瞰图，擅长捕捉水平布局
- **XZ 平面 (Front)**：正视图，擅长捕捉物体高度
- **YZ 平面 (Side)**：侧视图，补充深度方向信息

3D 空间中任意一个体素的特征 = 三个平面在该位置的投影特征之和。这通过 `grid_sample` 实现：每个体素的笛卡尔坐标先转换为柱坐标，然后在三个平面上采样，最后求和。

---

## 2. 涉及的文件

### 2.1 新建文件

| 文件路径 | 功能 |
|---------|------|
| `projects/occ_plugin/occupancy/lidar_encoder/__init__.py` | 注册所有 LiDAR 模块 |
| `projects/occ_plugin/occupancy/lidar_encoder/cylinder_encoder.py` | `CylinderEncoder`: 点云 → 柱坐标稀疏编码 → 3个TPV平面 |
| `projects/occ_plugin/occupancy/lidar_encoder/tpv_swin.py` | `TPVSwin`: 3个TPV平面共享 Swin Transformer 多尺度特征提取 |
| `projects/occ_plugin/occupancy/lidar_encoder/tpv_fpn.py` | `TPVFPN`: 每个TPV平面的多尺度 → 单尺度 FPN |
| `projects/occ_plugin/occupancy/lidar_encoder/tpv_fuser.py` | `TPVFuser`: grid_sample 从3平面采样 → 求和 → 3D体素特征 |
| `projects/occ_plugin/occupancy/detectors/flc_pointocc_net.py` | `FLCPointOccNet`: 双模态融合主检测器 |
| `projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py` | 完整训练配置 |

### 2.2 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `projects/occ_plugin/occupancy/__init__.py` | 添加 `from .lidar_encoder import *` |
| `projects/occ_plugin/occupancy/detectors/__init__.py` | 添加 `from .flc_pointocc_net import FLCPointOccNet` |

### 2.3 依赖的已有文件（未修改）

| 文件路径 | 功能 |
|---------|------|
| `projects/occ_plugin/occupancy/detectors/occnet.py` | 父类 `OccNet`，提供 camera pipeline 和 loss 计算 |
| `projects/occ_plugin/occupancy/dense_heads/flc_occ_head.py` | `FLCOccHead`，Conv2d + C2H MLP 预测头 |
| `projects/occ_plugin/occupancy/backbones/custom_resnet.py` | `CustomResNet2D`，2D BEV 编码器 |
| `projects/occ_plugin/occupancy/necks/fpn_lss.py` | `FPN_LSS`，2D BEV FPN |

### 2.4 新增外部依赖

| 包名 | 用途 | 安装方式 |
|------|------|---------|
| `torch_scatter` | CylinderEncoder 中的 scatter_max 点云聚合 | `pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html` |

---
## 3. 训练方法

### 3.1 启动训练

```bash
conda activate OpenOccupancy-4070

# 单GPU训练
 PYTHONPATH="./":$PYTHONPATH python tools/train.py projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py --seed 0


# 分布式训练 (如有多GPU)
bash tools/dist_train.sh \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py \
    <GPU数量>
```

### 3.2 从 checkpoint 恢复训练

```bash
python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py \
    --work-dir work_dirs/flc_pointocc \
    --resume-from work_dirs/flc_pointocc/latest.pth
```

### 3.3 测试评估

```bash
bash tools/dist_test.sh \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py \
    work_dirs/flc_pointocc/best_SSC_mean.pth \
    1
```

---

## 4. 训练注意事项

### 4.1 显存管理

- 当前配置 `samples_per_gpu=2`，在 RTX 4070 Ti (12GB) 上预计占用 ~10 GB
- 如果 OOM，优先降低 `samples_per_gpu` 到 1
- 可启用梯度累积来补偿 batch size：修改 `cumulative_iters`（如设为 2，等效 batch_size 翻倍）

### 4.2 loss_norm 必须为 False

配置中 `loss_norm=False` 已设置正确。**绝对不要改为 True**，否则所有 loss 会被自身归一化到 1.0，梯度失去方向信息导致无法学习。

### 4.3 Swin 预训练权重（可选但推荐）

当前 Swin Transformer 从头训练。如果想加速 LiDAR 分支收敛，可以下载 ImageNet 预训练权重：

```bash
mkdir -p pretrain
wget -O pretrain/swin_tiny_patch4_window7_224.pth \
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

然后在配置文件中取消注释：
```python
# 将这行
# pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
# 改为
pretrained='pretrain/swin_tiny_patch4_window7_224.pth',
```

### 4.4 数据要求

本模型同时需要 camera 和 LiDAR 数据：
- `data/nuscenes/` — nuScenes 完整数据集（含 LiDAR sweeps）
- `data/nuScenes-Occupancy/` — 占据标注
- `data/depth_gt/` — 深度 GT（用于 loss_depth）
- `data/nuscenes/nuscenes_occ_infos_train.pkl` / `val.pkl` — 数据索引

确保 `input_modality` 中 `use_lidar=True` 和 `use_camera=True` 同时开启。

### 4.5 训练时长预估

由于 LiDAR 分支（特别是 Swin Transformer 对 3 个 TPV 平面各过一遍）增加了不少计算量，单卡训练 30 epoch 在 4070 Ti 上预计需要 3-5 天。可以先跑 10-15 epoch 观察趋势。

---

## 5. 可调参数参考

所有参数均在配置文件 `CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py` 中修改。

### 5.1 全局空间参数

| 参数 | 默认值 | 含义 | 修改影响 |
|------|--------|------|---------|
| `point_cloud_range` | `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` | 感知范围 [x_min, y_min, z_min, x_max, y_max, z_max] (m) | 改变会影响所有空间分辨率 |
| `occ_size` | `[512, 512, 40]` | GT 占据网格分辨率 (高分辨率) | 通常不改 |
| `lss_downsample` | `[4, 4, 4]` | BEV 下采样倍率 → 实际工作分辨率 128x128x10 | 增大可省显存但降低精度 |

### 5.2 Camera 分支参数

| 参数 | 默认值 | 含义 | 修改影响 |
|------|--------|------|---------|
| `data_config.input_size` | `(256, 704)` | 输入图像分辨率 (H, W) | 增大可提精度但增加显存和时间 |
| `numC_Trans` | `64` | 每个高度 bin 的通道数 | 决定 BEV 通道数 = numC_Trans * Dz |
| `Dz` | `10` | 高度方向 bin 数 | 改变需同时修改 grid_config.zbound |
| `img_backbone.depth` | `50` | ResNet 深度 (18/34/50/101) | 增大提精度但增加计算 |
| `img_backbone.frozen_stages` | `0` | 冻结前 N 个 stage | 设为 1 可省显存，但降低 fine-tune 效果 |

### 5.3 LiDAR 分支参数

| 参数 | 默认值 | 含义 | 修改影响 |
|------|--------|------|---------|
| `cyl_grid_size` | `[240, 180, 16]` | 柱坐标稀疏网格分辨率 [rho, phi, z] | 增大精度更高但显存暴涨 |
| `lidar_tokenizer.split` | `[4, 4, 4]` | CylinderEncoder 各轴池化因子 | 必须能整除 cyl_grid_size |
| `lidar_tokenizer.base_channels` | `128` | CylinderEncoder 输出通道数 | 改变需同步修改 Swin in_channels |
| `lidar_backbone.embed_dims` | `96` | Swin 嵌入维度 | 增大提升容量但增加显存 |
| `lidar_backbone.depths` | `[2,2,6,2]` | Swin 各 stage 的 block 数 | 减少可加速 (如 [2,2,2,2]) |
| `lidar_backbone.with_cp` | `True` | 梯度 checkpoint | True 省显存、慢一点 |
| `tpv_C` | `192` | TPVFPN 输出通道数 (每个平面) | 改变需同步修改 lidar_proj_in |
| `tpv_fuser.scale_h/w/z` | `2, 2, 2` | TPV平面插值上采样因子 | 减小可省显存和计算 |

### 5.4 融合参数

| 参数 | 默认值 | 含义 | 修改影响 |
|------|--------|------|---------|
| `lidar_proj_out` | `64` | LiDAR 通道压缩目标维度 | 决定 lidar_bev 通道 = proj_out * Dz |
| `cam_adapter_out` | `320` | Camera adapter 输出通道 | cam_out + lidar_out 必须 = fused_channels |
| `lidar_adapter_out` | `320` | LiDAR adapter 输出通道 | 同上 |
| `fused_channels` | `640` | 融合后通道数 (= occ_encoder_backbone 输入) | 改变需同步修改 bev_channels |

### 5.5 解码器参数

| 参数 | 默认值 | 含义 | 修改影响 |
|------|--------|------|---------|
| `c2h_conv_out_dim` | `256` | FLCOccHead Conv2d 输出通道 | 增大增加容量 |
| `c2h_hidden_dim` | `512` | FLCOccHead C2H MLP 隐藏层维度 | 增大增加容量 |
| `voxel_out_channel` | `256` | FPN_LSS 输出通道 (= FLCOccHead 输入) | 改变需同步修改 |

### 5.6 训练超参数

| 参数 | 默认值 | 含义 | 调整建议 |
|------|--------|------|---------|
| `samples_per_gpu` | `2` | 单 GPU batch size | OOM 时降为 1 |
| `workers_per_gpu` | `2` | 数据加载线程数 | CPU 充足可增至 4 |
| `optimizer.lr` | `3e-4` | 初始学习率 | batch_size 减半时可等比缩小 |
| `optimizer.weight_decay` | `0.01` | 权重衰减 | 通常不改 |
| `lr_config.warmup_iters` | `500` | warmup 步数 | 通常不改 |
| `runner.max_epochs` | `30` | 总训练轮数 | 可先跑 15 epoch 观察 |
| `cumulative_iters` | `1` | 梯度累积步数 | 设为 2 可在 bs=1 时等效 bs=2 |
| `grad_clip.max_norm` | `35` | 梯度裁剪阈值 | 训练不稳定时可降低 |

### 5.7 Loss 权重

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `loss_voxel_ce_weight` | `1.0` | Cross-Entropy loss 权重 |
| `loss_voxel_sem_scal_weight` | `1.0` | 语义缩放 loss 权重 |
| `loss_voxel_geo_scal_weight` | `1.0` | 几何缩放 loss 权重 |
| `loss_voxel_lovasz_weight` | `1.0` | Lovasz-softmax loss 权重 |
| `loss_depth_weight` | `3.0` | 深度监督 loss 权重 (ViewTransformer) |

### 5.8 数据增强

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `bda_aug_conf.rot_lim` | `(-0, 0)` | BEV 旋转增强范围 (度) |
| `bda_aug_conf.scale_lim` | `(0.95, 1.05)` | 缩放增强范围 |
| `bda_aug_conf.flip_dx_ratio` | `0.5` | X 方向翻转概率 |
| `bda_aug_conf.flip_dy_ratio` | `0.5` | Y 方向翻转概率 |
| `data_config.resize` | `(-0.06, 0.11)` | 图像随机缩放范围 |
| `data_config.rot` | `(-5.4, 5.4)` | 图像随机旋转范围 (度) |
| `sweeps_num` | `10` | LiDAR 多帧聚合数量 |

---

## 6. 常见问题

### Q: CUDA out of memory

降低 `samples_per_gpu` 或减小 `cyl_grid_size`。也可以关闭 LiDAR 分支的 Swin 中的 checkpoint (`with_cp=False` → `True` 已默认开启)。

### Q: 只想测试 camera 分支效果

将 `input_modality.use_lidar` 设为 `False`，模型会自动 fallback 到纯相机模式（LiDAR 特征用零填充）。但这不是最优的 camera-only 模型，建议用专用的 FLC 配置 `CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py`。

### Q: 柱坐标网格分辨率如何选择

`cyl_grid_size` 控制 CylinderEncoder 的稀疏体素化精度：
- `[480, 360, 32]` — PointOcc 原版，适合 24GB+ GPU
- `[240, 180, 16]` — 当前默认，适合 12GB GPU，精度损失有限
- `[120, 90, 8]` — 极度节省，特征质量下降明显

注意修改后必须同步调整 `split`（需能整除 grid_size 的每个维度）和 `tpv_w/h/z`。
