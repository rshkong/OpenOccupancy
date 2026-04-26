# FLC-PointOcc 模型文档

FLC-PointOcc 是一个双模态（Camera + LiDAR）融合架构，结合了 FlashOcc 的 2D BEV 快速预测和 PointOcc 的 Tri-Perspective View (TPV) 几何特征。核心思路是用 LiDAR 分支产生的 TPV 几何特征来补充相机分支 BEV 中深度不准确的缺陷。

当前提供两个融合配置，唯一变量是相机 BEV 是否压缩：

| 配置文件 | 简称 | cam_adapter | fuse_conv 输入 |
|---|---|---|---|
| `projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py` | **Version A** | Conv1x1 640→256 | 384 |
| `projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py` | **Version B** | 无（cam_bev 直接进 cat） | 768 |

两版的 LiDAR 分支、lidar_adapter（1920→128）、fuse_conv 输出（256）、post-fusion 主干完全相同；用于消融"压相机是否更好"。

LiDAR 配方对齐已验证的 LiDAR-only PointOcc（`work_dirs/LiDAR_pointocc_server4090x2/`）：`cyl_grid=[480,360,32]`、`split=[8,8,8]`、`tpv = cyl//2 = [240,180,16]`、Swin-T ImageNet 预训练权重。

> 历史配置 `CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py` 仍存在于仓库中，但已通过 `debug_camera_only_bypass=True` 切换为相机短路路径（等价于 FLC-step2，详见 §6 调试开关）。新训练请使用上面的 A / B 配置。

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
│                                                       │
│    Version A: cam_adapter Conv1x1 640→256             │
│      → cam_feat [B, 256, 128, 128]                   │
│    Version B: 无 cam_adapter                          │
│      → cam_feat = cam_bev [B, 640, 128, 128]         │
│                                                       │
└────────────────────────┬──────────────────────────────┘
                         │
                         │
┌─────────────────── LiDAR Branch ─────────────────────┐
│                                                       │
│  points [B, N, 5]  (x, y, z, intensity, ring)        │
│    → cart2polar + 10-channel 特征构建                 │
│    → CylinderEncoder (lidar_tokenizer)                │
│      cyl_grid=[480,360,32], split=[8,8,8]             │
│    → 3 个 TPV 平面:                                   │
│        tpv_xy [B, 128, 480, 360]                      │
│        tpv_yz [B, 128, 360, 32]                       │
│        tpv_zx [B, 128, 32, 480]                       │
│    → TPVSwin (lidar_backbone, Swin-T ImageNet 预训练) │
│    → TPVFPN (lidar_neck)                              │
│      每个平面 → [B, 192, h, w] (h,w = cyl//2)         │
│    → TPVFuser (tpv_fuser, scale_h/w/z=2)              │
│      grid_sample 在 128×128×10 处采样 + 三平面求和    │
│    → lidar_3d [B, 192, 128, 128, 10]                 │
│                                                       │
│    Flatten Z 先于通道压缩（每个 Z bin 独立投影权重） │
│    → permute/reshape → lidar_bev [B, 1920, 128, 128] │
│    → lidar_adapter Conv1x1 1920→128                   │
│    → lidar_feat [B, 128, 128, 128]                   │
│                                                       │
└────────────────────────┬──────────────────────────────┘
                         │
                         ↓
        cat([cam_feat, lidar_feat], dim=1)
            Version A: [B, 384, 128, 128]   (256 + 128)
            Version B: [B, 768, 128, 128]   (640 + 128)
                         ↓
        fuse_conv Conv3x3 + BN + ReLU → 256
                         ↓
              fused_bev [B, 256, 128, 128]
                         │
              CustomResNet2D (occ_encoder_backbone, numC_input=256)
                         │
              FPN_LSS (occ_encoder_neck)
                         │
              [B, 256, 128, 128]
                         │
              FLCOccHead (Conv2d + C2H MLP)
                         │
              output_voxels [B, 17, 128, 128, 10]
```

设计原则（参考 BEVFusion）：**前轻后重**——adapter 小、fuse 用 Conv3x3、post-fusion 主干在 256 通道下做主要工作。Conv3x3 在融合点提供跨模态空间平滑，比之前的 Conv1x1 更合理。

     A/B 参数对比

     ┌──────────────────┬──────────────┬──────────────┬─────────────┬─────────────┐
     │                  │ A (cam→256)  │ B (cam=640)  │  BEVFusion  │
     ├──────────────────┼──────────────┼──────────────┼─────────────┼─────────────┤
     │ cam_adapter      │ 1x1 640→256  │ 无           │ 无          │
     ├──────────────────┼──────────────┼──────────────┼─────────────┼─────────────┤
     │ lidar_adapter    │ 1x1 1920→128 │ 1x1 1920→128 │ 无          │
     ├──────────────────┼──────────────┼──────────────┼─────────────┼─────────────┤
     │ cat 后通道        │ 384          │ 768          │ 208         │
     ├──────────────────┼──────────────┼──────────────┼─────────────┼─────────────┤
     │ fuse_conv        │ 3x3 384→256  │ 3x3 768→256  │ 3x3 208→256 │
     ├──────────────────┼──────────────┼──────────────┼─────────────┼─────────────┤
     │ post-fusion 输入  │  256          │ 256         │ 256        │
     └──────────────────┴──────────────┴──────────────┴─────────────┴─────────────┘


### 1.2 逐阶段详解

#### Camera 分支

| 阶段 | 模块 | 输入 → 输出 | 代码位置 |
|------|------|-------------|---------|
| 图像编码 | `ResNet50` + `SECONDFPN` | `[B,6,3,256,704]` → `[B,6,512,16,44]` | `occnet.py:image_encoder()` |
| 视角变换 | `ViewTransformerLSSFlash` | 2D 图像特征 → BEV 体素 → collapse Z | `image2bev/ViewTransformerLSSFlash` |
| BEV 输出 | Z-collapse | `[B,64,128,128,10]` → `[B,640,128,128]` | ViewTransformer 内部 |
| 通道适配 | **A**: `cam_adapter` Conv1x1 / **B**: 无 | `[B,640,X,Y]` → A: `[B,256,X,Y]` / B: 透传 | `flc_pointocc_net.py` |

#### LiDAR 分支

| 阶段 | 模块 | 输入 → 输出 | 代码位置 |
|------|------|-------------|---------|
| 点云预处理 | `_prepare_lidar_inputs()` | `[N,5]` → 10 通道柱坐标特征 + grid_ind | `lidar_prep_mixin.py` |
| 稀疏编码 | `CylinderEncoder` (split=`[8,8,8]`) | 10ch → 3 个 TPV 平面 `[B,128,H,W]` | `lidar_encoder/cylinder_encoder.py` |
| 多尺度特征 | `TPVSwin` (Swin-T ImageNet 预训练) | 3 个平面共享 Swin → 多尺度特征 | `lidar_encoder/tpv_swin.py` |
| 特征融合 | `TPVFPN` | 多尺度 → `[B,192,h,w]` | `lidar_encoder/tpv_fpn.py` |
| 3D 重建 | `TPVFuser` (scale=2) | 3 平面 → grid_sample → 求和 → `[B,192,128,128,10]` | `lidar_encoder/tpv_fuser.py` |
| **Z-flatten** | `permute + reshape` | `[B,192,X,Y,Z]` → `[B, 192·Z=1920, X, Y]` | `flc_pointocc_net.py` |
| 通道适配 | `lidar_adapter` Conv1x1 1920→128 | `[B,1920,X,Y]` → `[B,128,X,Y]` | `flc_pointocc_net.py` |

**Z-flatten 设计**：先把 Z 维 flatten 进通道，再用 Conv1x1 压缩。这样每个高度 bin 都拥有自己的投影权重（参数 1920·128 ≈ 246K）。旧实现先 `Linear(192→64)` 再 flatten 会强迫所有 Z 共享同一个 192→64 矩阵（仅 12K 参数），对地面 / 中层 / 上层语义不同的 occupancy 任务过于受限。

#### 融合 + 解码

| 阶段 | 模块 | A 输入 → 输出 | B 输入 → 输出 | 代码位置 |
|------|------|---------------|---------------|---------|
| 通道拼接 | `torch.cat` | `256 + 128 → 384` | `640 + 128 → 768` | `flc_pointocc_net.py` |
| 融合卷积 | `fuse_conv` Conv3x3 + BN + ReLU | `384 → 256` | `768 → 256` | `flc_pointocc_net.py` |
| 2D 编码器 | `CustomResNet2D` (`numC_input=256`) | `[B,256,128,128]` → 多尺度 | 同 A | `occnet.py:occ_encoder()` |
| 2D FPN | `FPN_LSS` | 多尺度 → `[B,256,128,128]` | 同 A | `occnet.py:occ_encoder()` |
| 占据预测 | `FLCOccHead` (Conv2d + C2H MLP) | `[B,256,128,128]` → `[B,17,128,128,10]` | 同 A | `dense_heads/flc_occ_head.py` |

### 1.3 10 通道点云特征说明

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

到 `tpv_fuser` 输出 `[B,192,X,Y,Z]` 这一步，FLC-PointOcc 的 LiDAR 分支与 LiDAR-only PointOcc 数学等价（都是 `tpv_hw + tpv_zh + tpv_wz`）。LiDAR-only 在这之上仅再过一个 MLP+classifier 直接出 17 类 logits；融合版本则把这个 3D 特征 flatten Z 后送入 fuse_conv。

---

## 2. 涉及的文件

### 2.1 核心检测器与 LiDAR 模块

| 文件路径 | 功能 |
|---------|------|
| `projects/occ_plugin/occupancy/detectors/flc_pointocc_net.py` | `FLCPointOccNet`：双模态融合主检测器 |
| `projects/occ_plugin/occupancy/detectors/lidar_prep_mixin.py` | `LidarPrepMixin`：点云 → 柱坐标 / grid_ind |
| `projects/occ_plugin/occupancy/lidar_encoder/__init__.py` | 注册所有 LiDAR 模块 |
| `projects/occ_plugin/occupancy/lidar_encoder/cylinder_encoder.py` | `CylinderEncoder` |
| `projects/occ_plugin/occupancy/lidar_encoder/tpv_swin.py` | `TPVSwin` |
| `projects/occ_plugin/occupancy/lidar_encoder/tpv_fpn.py` | `TPVFPN` |
| `projects/occ_plugin/occupancy/lidar_encoder/tpv_fuser.py` | `TPVFuser` |

### 2.2 配置文件

| 文件路径 | 用途 |
|---------|------|
| `projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py` | **Version A**：cam 压到 256 |
| `projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py` | **Version B**：cam 保持 640 |
| `projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py` | 旧融合配置（当前用 bypass 开关跑相机短路） |
| `projects/configs/baselines/CAM-LiDAR_flc_pointocc_bypass_128x128x10_4070ti.py` | 实验 B：bypass-only 对齐 step2 配方 |

### 2.3 依赖的已有文件（未修改）

| 文件路径 | 功能 |
|---------|------|
| `projects/occ_plugin/occupancy/detectors/occnet.py` | 父类 `OccNet`，提供 camera pipeline 和 loss 计算 |
| `projects/occ_plugin/occupancy/dense_heads/flc_occ_head.py` | `FLCOccHead`，Conv2d + C2H MLP 预测头 |
| `projects/occ_plugin/occupancy/backbones/custom_resnet.py` | `CustomResNet2D`，2D BEV 编码器 |
| `projects/occ_plugin/occupancy/necks/fpn_lss.py` | `FPN_LSS`，2D BEV FPN |

### 2.4 外部依赖

| 包名 | 用途 | 安装方式 |
|------|------|---------|
| `torch_scatter` | CylinderEncoder 中的 scatter_max 点云聚合 | `pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html` |

---

## 3. 训练方法

### 3.1 前置：Swin-T ImageNet 预训练权重（必需）

A/B 两版 config 都加载了 `pretrain/swin_tiny_patch4_window7_224.pth`，没有这个文件训练会失败：

```bash
mkdir -p pretrain
wget -O pretrain/swin_tiny_patch4_window7_224.pth \
    https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

### 3.2 启动训练

两版 config 训练命令一致，只换 config 路径和 work-dir。下面先给**单卡训练**版本，适合当前多卡训练还未稳定的情况。当前 config 默认 `samples_per_gpu=4`，因此单卡时 effective batch = 4；如果显存不足，手动把 `samples_per_gpu` 下调到 1 或 2。

```bash
conda activate OpenOccupancy-4070
cd /home/shkong/MyProject/OpenOccupancy

# Version A：cam 压到 256
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --work-dir work_dirs/CAM-LiDAR_flc_pointocc_camadapt256 \
    --seed 0

# Version B：cam 保持 640
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py \
    --work-dir work_dirs/CAM-LiDAR_flc_pointocc_camfull640 \
    --seed 0
```

后续如果多卡恢复正常，再把命令切回分布式版本，并按 `samples_per_gpu × nproc = 8` 对齐原计划的 effective batch；偏离 8 时按 linear scaling 微调 `lr`（默认 2e-4）。

### 3.3 单卡 smoke test（不依赖完整数据 / 调试用）

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --work-dir work_dirs/_smoke_A --seed 0 --no-validate \
    2>&1 | head -200
```

预期：日志里出现 `load checkpoint from local path: pretrain/swin_tiny...`，无 shape mismatch，第一个 iter loss 落地。单卡 4070 Ti 上 `samples_per_gpu=4` 大概率 OOM，smoke test 时手动改到 1。

### 3.4 从 checkpoint 恢复训练

```bash
PYTHONPATH="./":$PYTHONPATH python -m torch.distributed.launch \
    --nproc_per_node=2 tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --work-dir work_dirs/CAM-LiDAR_flc_pointocc_camadapt256 \
    --launcher pytorch --seed 0 \
    --resume-from work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/latest.pth
```

### 3.5 测试评估

```bash
bash tools/dist_test.sh \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/best_SSC_mean.pth \
    2
```

### 3.6 指标判读

| 现象 | 解释 |
|------|------|
| 第 1 epoch 末 SSC_mean ≥ camera-only step1 (~0.09) | 融合至少没退化 |
| 收敛后 SSC_mean ≥ LiDAR-only PointOcc | 融合真有效 |
| A ≈ B（差距 < 0.01 SSC） | cam 压不压都行；倾向 A（参数少、更接近 BEVFusion） |
| A < B | 256 维 cam_adapter 丢信息，建议放宽 cam 通道 |
| A > B | fuse_conv 在 768 输入下不够宽，可放大或退回 A |

---

## 4. 训练注意事项

### 4.1 显存管理

- 默认 `samples_per_gpu=4` 针对 24GB 卡（4090）。12GB 卡（4070 Ti）下需降到 1，且可能仍 OOM —— 4070 Ti 不是融合训练的目标平台。
- LiDAR 分支已开 `with_cp=True`（CylinderEncoder + Swin），相机 ResNet50 也是 `with_cp=True`。
- `OccEfficiencyHook` 会 deepcopy 整个模型，浪费 ~400MB VRAM，已在 config 中禁用。
- **`workers_per_gpu=2`**（不要随便调到 8）。每个 worker 进程会缓存 prefetch 队列里的 6 摄像头图像 + 10 sweep LiDAR + 512×512×40 occupancy GT，单 worker RES ≈ 4 GiB。`workers=8` 会让总 RAM 飙到 100–200 GiB（参照 LiDAR-only baseline `workers=2` 时稳定在 ~11 GiB）。

### 4.2 `loss_norm` 必须为 `False`

A/B 两版 config 都设了 `loss_norm=False`。**绝对不要改为 `True`**，否则所有 loss 会被自身归一化到 1.0，梯度失去方向信息导致无法学习。

### 4.3 数据要求

- `data/nuscenes/` — nuScenes 完整数据集（含 LiDAR sweeps）
- `data/nuScenes-Occupancy/` — 占据标注
- `data/depth_gt/` — 深度 GT（用于 loss_depth）
- `data/nuscenes/nuscenes_occ_infos_train.pkl` / `val.pkl` — 数据索引
- `pretrain/swin_tiny_patch4_window7_224.pth` — Swin-T ImageNet 权重

确保 `input_modality.use_lidar=True` 和 `use_camera=True` 同时开启。

### 4.4 训练时长预估

LiDAR 分支 cyl_grid 升到 `[480,360,32]` 之后，2×4090（eff bs=8）跑完 24 epoch 大约需要 1.5–2 天（参考 LiDAR-only PointOcc 同卡同 bs 用了约 1 天）。

---

## 5. 可调参数参考

参数都在 `CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py` / `CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py` 中。

### 5.1 全局空间参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `point_cloud_range` | `[-51.2,-51.2,-5.0,51.2,51.2,3.0]` | 感知范围 |
| `occ_size` | `[512, 512, 40]` | GT 占据网格分辨率 |
| `lss_downsample` | `[4, 4, 4]` | BEV 下采样倍率 → 工作分辨率 128×128×10 |

### 5.2 Camera 分支参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `data_config.input_size` | `(256, 704)` | 输入图像分辨率 |
| `numC_Trans` | `64` | LSS 每高度 bin 通道数 |
| `Dz` | `10` | 高度 bin 数（cam_bev = 64×10 = 640） |
| `img_backbone.depth` | `50` | ResNet 深度 |
| `cam_adapter_out` | A: 256 / B: 无 | A 才有 cam_adapter |

### 5.3 LiDAR 分支参数（对齐 LiDAR-only）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `cyl_grid_size` | `[480, 360, 32]` | 柱坐标稀疏网格分辨率 |
| `lidar_tokenizer.split` | `[8, 8, 8]` | CylinderEncoder 各轴池化因子 |
| `tpv_w/h/z` | `cyl // 2` = `(240,180,16)` | TPVFuser 采样网格大小 |
| `lidar_backbone.pretrained` | `pretrain/swin_tiny_patch4_window7_224.pth` | Swin-T ImageNet 权重（必加） |
| `tpv_C` | `192` | TPVFPN 输出通道（每平面） |
| `lidar_proj_in` | `tpv_C = 192` | 仅作记账，已不再用 Linear 压缩 |
| `lidar_adapter.in_channels` | `tpv_C * Dz = 1920` | flatten Z 后输入通道 |
| `lidar_adapter_out` | `128` | LiDAR 通道压缩目标 |

### 5.4 融合参数

| 参数 | A 默认 | B 默认 | 含义 |
|------|--------|--------|------|
| `cam_adapter_cfg` | Conv1x1 640→256 | `None` | cam 是否压 |
| `lidar_adapter_cfg` | Conv1x1 1920→128 | 同 A | LiDAR Z-flatten 后压缩 |
| `fuse_conv_cfg.in_channels` | 384 | 768 | cat 后通道 |
| `fuse_conv_cfg.kernel_size` | 3 | 3 | BEVFusion-style Conv3x3 |
| `fuse_conv_cfg.out_channels` | 256 | 256 | post-fusion 输入通道 |
| `fused_channels` | 256 | 256 | 同 fuse_out |
| `occ_encoder_backbone.numC_input` | 256 | 256 | post-fusion 主干输入 |

### 5.5 训练超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `samples_per_gpu` | `4` | 单 GPU batch（24GB 卡） |
| `optimizer.lr` | `2e-4` | 对齐 LiDAR-only PointOcc |
| `paramwise_cfg` | `img_backbone lr_mult=0.1` | ResNet50 预训练，更小 lr |
| `runner.max_epochs` | `24` | 对齐 PointOcc |
| `find_unused_parameters` | `False` | + `static_graph=True`，DDP 加速 |

### 5.6 Loss 权重

| 参数 | 默认值 |
|------|--------|
| `loss_voxel_ce_weight` | `1.0` |
| `loss_voxel_sem_scal_weight` | `1.0` |
| `loss_voxel_geo_scal_weight` | `1.0` |
| `loss_voxel_lovasz_weight` | `1.0` |
| `loss_depth_weight` | `3.0` |

### 5.7 数据增强

| 参数 | 默认值 |
|------|--------|
| `bda_aug_conf.scale_lim` | `(0.95, 1.05)` |
| `bda_aug_conf.flip_dx_ratio` | `0.5` |
| `bda_aug_conf.flip_dy_ratio` | `0.5` |
| `data_config.resize` | `(-0.06, 0.11)` |
| `data_config.rot` | `(-5.4, 5.4)` |
| `sweeps_num` | `10` |

---

## 6. 调试开关与常见问题

### 6.1 `debug_camera_only_bypass`

`FLCPointOccNet.__init__` 接受两个 debug 开关，A/B 两版生产 config 都设为 `False`：

- **`debug_camera_only_bypass=True`**：完全跳过 cam_adapter / LiDAR 分支 / fuse_conv，`cam_bev` 直接进 occ_encoder。等价于 FLC-step2 的相机路径。LiDAR 模块在 `debug_camera_only_bypass=True` 时**不会被构造**（在 `__init__` 中直接置 `None`），所以 config 里可以省略 LiDAR 字段。
- **`debug_zero_lidar=True`**：cam_adapter / lidar_adapter / fuse_conv 都跑，但把 `lidar_feat` 零化。测"融合路径在 LiDAR 中性时还能不能让相机信号通过"。

A/B 配置都关闭这两个开关，跑真正的双模态融合。

### 6.2 lidar_adapter / fuse_conv 必填

`debug_camera_only_bypass=False` 时，`lidar_adapter_cfg` 和 `fuse_conv_cfg` 都必须在 config 里给出。原本 `lidar_adapter_cfg=None` 会 fallback 到 `nn.Identity()`，但这与 flatten-then-Conv1x1 不兼容（lidar_bev 是 1920ch，Identity 会让 fuse_conv 输入爆炸），现已改为直接 `raise`。

`lidar_proj_out` 是已废弃的参数，构造器会接受并忽略，仅为兼容旧 config。

### 6.3 Q: CUDA out of memory

降低 `samples_per_gpu`；或在 4070 Ti 上跑就只能用 bypass-only config。融合训练目标平台是 24GB 卡。

### 6.4 Q: 只想测试 camera 分支效果

不要用 fusion config + `use_lidar=False`；改用专用 camera-only 配置：
- 干净 step2：`CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py`
- 沿用融合代码、纯相机短路：旧 fusion config + `debug_camera_only_bypass=True`，或 `CAM-LiDAR_flc_pointocc_bypass_128x128x10_4070ti.py`

### 6.5 Q: 柱坐标网格分辨率如何选择

`cyl_grid_size` 控制 CylinderEncoder 的稀疏体素化精度：

- `[480, 360, 32]` — 当前默认（PointOcc 原版），24GB+ GPU 推荐
- `[240, 180, 16]` — 12GB GPU 折中（旧 4070 Ti 配置），精度损失明显
- `[120, 90, 8]` — 极度节省，特征质量下降明显

修改后必须同步调整 `split`（需能整除每个维度）和 `tpv_w/h/z = cyl // 2`。
