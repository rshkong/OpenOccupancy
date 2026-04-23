# Training Cheatsheet

快速查询：目前项目里有哪些可训练模型，配置在哪里，用什么命令训。
所有命令默认在仓库根目录运行，并已激活 `OpenOccupancy-4070` conda 环境。

```bash
conda activate OpenOccupancy-4070
cd /home/shkong/MyProject/OpenOccupancy
```

---

## 1. Camera-only: FlashOcc baseline (FLC step1)

- Config: `projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py`
- 训练入口：`run.sh`（多卡 DDP）

```bash
bash run.sh projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py 1
# 第二个参数是 GPU 数；单卡写 1
```

工作目录会自动落到 `work_dirs/CAM-R50_img256x704_128x128x10_4070ti/`。

---

## 2. Camera-only: FLC step2 (flc_step2 / flc_noNorm)

同样通过 `run.sh`：

```bash
# step2
bash run.sh projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py 1

# noNorm 变体
bash run.sh projects/configs/baselines/CAM-R50_img256x704_flc_noNorm_128x128x10_4070ti.py 1
```

---

## 3. Camera + LiDAR: FLC-PointOcc 融合模型

- Config: `projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py`
- 直接走单卡 `tools/train.py`（融合模型暂时单卡）

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py \
    --work-dir work_dirs/CAM-LiDAR_flc_pointocc_128x128x10_4070ti \
    --seed 0
```

### 3a. FLC-PointOcc Degraded（把 LiDAR 清零，保留 adapter + fuse_conv）

`debug_zero_lidar=True`：cam_adapter、lidar_adapter、fuse_conv **仍然跑**，只是 lidar_feat 被零化。测”融合路径在 LiDAR 中性时还能不能让相机信号通过”。

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py \
    --work-dir work_dirs/CAM-LiDAR_flc_pointocc_degraded \
    --seed 0
```
我发现degraded效果并不好，几乎和fused分支效果差不多，说明两件事1.lidar分支没有用 2.camera分支有问题，甚至不如FLC-step2

### 3b. FLC-PointOcc Camera-Only Bypass（真·FLC-step2 路径，沿用融合 config）

`debug_camera_only_bypass=True`：**完全跳过** cam_adapter / LiDAR 分支 / fuse_conv，`cam_bev` 直接进 2D encoder，和 FLC-step2 的相机路径一模一样。用来判定相机分支本身是否健康（独立于 adapter/fuse_conv）。

改 config：把 `debug_zero_lidar` 改 `False`，加一行 `debug_camera_only_bypass=True`，然后：

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py \
    --work-dir work_dirs/CAM-LiDAR_flc_pointocc_bypass \
    --seed 0
```

注意：此路径下 LiDAR 分支模块**还会被 build**（只是 forward 时被短路），数据 pipeline 仍加载点云，`samples_per_gpu=1`。适合快速验证，但 bs=1 + SyncBN 会让结果仍远逊 FLC-step2。这时应换用 3c。

要跑"真正的融合"版本：把两个 debug 开关都置 `False`，再换 `--work-dir`。

### 3c. FLC-PointOcc Bypass-Only 专用 config（实验 B，对齐 FLC-step2 训练配方）

和 3b 同样走 bypass 短路，但**把 LiDAR-相关模块从 config 里全部剥掉**（`lidar_tokenizer / lidar_backbone / lidar_neck / tpv_fuser / cam_adapter_cfg / lidar_adapter_cfg / fuse_conv_cfg = None`），数据 pipeline 也删除 `LoadPointsFromFile` / `LoadPointsFromMultiSweeps`，`input_modality.use_lidar=False`。

核心区别：释放了点云加载 + LiDAR TPV 分支的 VRAM，`samples_per_gpu` 可以直接拉到 **5**，和 FLC-step2 完全一致（30 epochs、`img_backbone lr_mult=0.1`、`OccEfficiencyHook` 都对齐）。

- Config: `projects/configs/baselines/CAM-LiDAR_flc_pointocc_bypass_128x128x10_4070ti.py`

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_bypass_128x128x10_4070ti.py \
    --work-dir work_dirs/CAM-LiDAR_flc_pointocc_bypass_only \
    --seed 0
```

判读（对齐训练配方后才有意义）：
- 实验 B 结果 ≈ FLC-step2 → `FLCPointOccNet` 的相机路径本身没问题，3a/3b 的下降是 bs=1 + SyncBN 造成的
- 实验 B 结果仍 ≪ FLC-step2 → 问题在 `FLCPointOccNet.extract_feat` 本身（即使短路了也还有路径差异），下一步要 diff OccNet vs FLCPointOccNet 的 forward 细节

---

## 4. LiDAR-only: PointOcc 复现（最新）

- Config: `projects/configs/baselines/LiDAR_pointocc_128x128x10_4070ti.py`
- 单卡 `tools/train.py`

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/LiDAR_pointocc_128x128x10_4070ti.py \
    --work-dir work_dirs/LiDAR_pointocc_128x128x10_4070ti \
    --seed 0
```

用途：验证 LiDAR 分支（CylinderEncoder + TPVSwin + TPVFPN + TPVAggregator）
单独是否能复现 PointOcc 的精度；如果能，融合模型跑不过 FlashOcc 的锅就落在融合
机制上，而不是 LiDAR port。

---

## 5. 断点续训 / 指定 checkpoint

所有命令都接受 `--resume-from` 和 `--load-from`：

```bash
# 从上次的 epoch 继续（保留 optimizer / lr 状态）
PYTHONPATH="./":$PYTHONPATH python tools/train.py <config> \
    --work-dir <dir> --seed 0 \
    --resume-from work_dirs/<dir>/latest.pth

# 只加载权重，重新开始优化
PYTHONPATH="./":$PYTHONPATH python tools/train.py <config> \
    --work-dir <dir> --seed 0 \
    --load-from path/to/ckpt.pth
```

---

## 6. 评估（train 完 / 中途评估）

```bash
PYTHONPATH="./":$PYTHONPATH python tools/test.py \
    <config> work_dirs/<dir>/latest.pth \
    --eval bbox
```

---

## 参考：快速对照

| 模型 | Config | 入口 |
|---|---|---|
| FlashOcc (FLC step1) | `CAM-R50_img256x704_128x128x10_4070ti.py` | `run.sh ... 1` |
| FLC step2 | `CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py` | `run.sh ... 1` |
| FLC noNorm | `CAM-R50_img256x704_flc_noNorm_128x128x10_4070ti.py` | `run.sh ... 1` |
| FLC-PointOcc (fusion) | `CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py` | `tools/train.py` |
| FLC-PointOcc Degraded (zero LiDAR) | 同上 + `debug_zero_lidar=True` | `tools/train.py` |
| FLC-PointOcc Bypass (沿用融合 config) | 同上 + `debug_camera_only_bypass=True` | `tools/train.py` |
| FLC-PointOcc Bypass-Only (实验 B，bs=5 对齐 step2) | `CAM-LiDAR_flc_pointocc_bypass_128x128x10_4070ti.py` | `tools/train.py` |
| LiDAR-only PointOcc | `LiDAR_pointocc_128x128x10_4070ti.py` | `tools/train.py` |
| 原始 LiDAR baseline | `LiDAR_128x128x10.py` | `run.sh ... 1` |
