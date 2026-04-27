# Evaluation Data Collection Cheatsheet

本文档列出收集论文中每个实验数据所需执行的具体命令。所有命令在 `OpenOccupancy` 根目录下运行，执行前激活 conda 环境：

```bash
conda activate OpenOccupancy-4070
cd /home/shkong/MyProject/OpenOccupancy
```

---

## 模型 × 配置文件 × Work-dir 对照表

| 简称 | 配置文件（`projects/configs/baselines/`） | Work-dir（`work_dirs/`） |
|---|---|---|
| **Cam-OO** | `CAM-R50_img256x704_128x128x10_4070ti.py` | `CAM-R50_img256x704_*` |
| **LiDAR-OO** | `LiDAR_128x128x10.py` | `LiDAR_128x128x10_*` |
| **MM-OO** | `Multimodal-R50_img256x704_128x128x10.py` *(需先创建 256×704 版本)* | `Multimodal_*` |
| **PointOcc** | `LiDAR_pointocc_128x128x10_server.py` | `LiDAR_pointocc_server4090*` |
| **FlashOcc** | `CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py` | `CAM-R50_img256x704_flc_step2_*` |
| **FLC-PointOcc A** | `CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py` | `CAM-LiDAR_flc_pointocc_camadapt256` |
| **FLC-PointOcc B** | `CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py` | `CAM-LiDAR_flc_pointocc_camfull640` |

以下命令中的 `[CKPT]` 均为实际的 `.pth` 文件路径，通常为 `work_dirs/<名称>/best_SSC_mean_epoch_XX.pth`。

---

## 1. mIoU / SC / Per-class IoU

**工具**：`tools/dist_test.sh`  
**输出**：`SC_non-empty`、`SSC_mean`（= mIoU）、`SSC_<class>` × 17 类

```bash
# ① Cam-OO
bash tools/dist_test.sh \
    projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py \
    [CKPT] 1 2>&1 | tee eval_logs/cam_oo.log

# ② LiDAR-OO
bash tools/dist_test.sh \
    projects/configs/baselines/LiDAR_128x128x10.py \
    [CKPT] 1 2>&1 | tee eval_logs/lidar_oo.log

# ③ MM-OO
bash tools/dist_test.sh \
    projects/configs/baselines/Multimodal-R50_img256x704_128x128x10.py \
    [CKPT] 1 2>&1 | tee eval_logs/mm_oo.log

# ④ PointOcc
bash tools/dist_test.sh \
    projects/configs/baselines/LiDAR_pointocc_128x128x10_server.py \
    [CKPT] 1 2>&1 | tee eval_logs/pointocc.log

# ⑤ FlashOcc
bash tools/dist_test.sh \
    projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \
    [CKPT] 1 2>&1 | tee eval_logs/flashocc.log

# ⑥ FLC-PointOcc A
bash tools/dist_test.sh \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    [CKPT] 1 2>&1 | tee eval_logs/flc_pointocc_A.log

# ⑦ FLC-PointOcc B
bash tools/dist_test.sh \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py \
    [CKPT] 1 2>&1 | tee eval_logs/flc_pointocc_B.log
```

**记录什么**：从终端/日志最后几行提取：
- `SSC_mean` → 填入论文 mIoU 列
- `SSC_<class>` × 17 → 填入 per-class IoU 表

---

## 2. 参数量（#Params / M）

**工具**：`tools/profile_model.py --params`  
**注意**：`--params` 仅构建模型、打印参数量后立即退出，无需 GPU 推理。

```bash
# ① Cam-OO
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py \
    --checkpoint [CKPT] --params

# ② LiDAR-OO
python tools/profile_model.py \
    --config projects/configs/baselines/LiDAR_128x128x10.py \
    --checkpoint [CKPT] --params

# ③ MM-OO
python tools/profile_model.py \
    --config projects/configs/baselines/Multimodal-R50_img256x704_128x128x10.py \
    --checkpoint [CKPT] --params

# ④ PointOcc
python tools/profile_model.py \
    --config projects/configs/baselines/LiDAR_pointocc_128x128x10_server.py \
    --checkpoint [CKPT] --params

# ⑤ FlashOcc
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \
    --checkpoint [CKPT] --params

# ⑥ FLC-PointOcc A
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --checkpoint [CKPT] --params

# ⑦ FLC-PointOcc B
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py \
    --checkpoint [CKPT] --params
```

**记录什么**：输出中的 `Total params : X.XX M`

---

## 3. GFLOPs

**工具**：`tools/profile_model.py --flops`  
**依赖**：需要安装 `fvcore`（或 `thop`）。安装命令：
```bash
pip install fvcore
# 或
pip install thop
```

**注意**：LiDAR 分支中的 `scatter_max`（CylinderEncoder）等稀疏操作无法被 fvcore 精确统计，会在输出中标注为 `unsupported`。摘录卷积/attention 部分时应注意这一限制，在论文中注明"不含稀疏点云聚合层"。

```bash
# 示例（其余模型同理，替换 --config 和 --checkpoint）
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \
    --checkpoint [CKPT] --flops

python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --checkpoint [CKPT] --flops
```

**记录什么**：输出中的 `GFLOPs (batch=1) : XX.X`

---

## 4. FPS + 推理峰值显存（Infer.Mem）

**工具**：`tools/profile_model.py`（默认模式）  
**建议参数**：`--warmup 10 --measure 50 --batch-size 1`

```bash
# ① Cam-OO
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py \
    --checkpoint [CKPT] \
    --warmup 10 --measure 50 --batch-size 1 \
    2>&1 | tee profile_logs/cam_oo_infer.log

# ② LiDAR-OO
python tools/profile_model.py \
    --config projects/configs/baselines/LiDAR_128x128x10.py \
    --checkpoint [CKPT] \
    --warmup 10 --measure 50 --batch-size 1 \
    2>&1 | tee profile_logs/lidar_oo_infer.log

# ③ MM-OO
python tools/profile_model.py \
    --config projects/configs/baselines/Multimodal-R50_img256x704_128x128x10.py \
    --checkpoint [CKPT] \
    --warmup 10 --measure 50 --batch-size 1 \
    2>&1 | tee profile_logs/mm_oo_infer.log

# ④ PointOcc
python tools/profile_model.py \
    --config projects/configs/baselines/LiDAR_pointocc_128x128x10_server.py \
    --checkpoint [CKPT] \
    --warmup 10 --measure 50 --batch-size 1 \
    2>&1 | tee profile_logs/pointocc_infer.log

# ⑤ FlashOcc
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \
    --checkpoint [CKPT] \
    --warmup 10 --measure 50 --batch-size 1 \
    2>&1 | tee profile_logs/flashocc_infer.log

# ⑥ FLC-PointOcc A
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --checkpoint [CKPT] \
    --warmup 10 --measure 50 --batch-size 1 \
    2>&1 | tee profile_logs/flc_pointocc_A_infer.log

# ⑦ FLC-PointOcc B
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py \
    --checkpoint [CKPT] \
    --warmup 10 --measure 50 --batch-size 1 \
    2>&1 | tee profile_logs/flc_pointocc_B_infer.log
```

**记录什么**：
- `FPS : X.XX` → 填入论文 FPS 列
- `Infer peak : X.XX MiB` → 填入推理显存列

---

## 5. 训练峰值显存（Train.Mem）

**工具**：`tools/profile_model.py --train-mem`  
**重要**：使用论文中的训练 batch size（`--batch-size 4`），确保和实际训练一致。

```bash
# ① Cam-OO
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py \
    --checkpoint [CKPT] \
    --train-mem --batch-size 4

# ② LiDAR-OO
python tools/profile_model.py \
    --config projects/configs/baselines/LiDAR_128x128x10.py \
    --checkpoint [CKPT] \
    --train-mem --batch-size 4

# ③ MM-OO
python tools/profile_model.py \
    --config projects/configs/baselines/Multimodal-R50_img256x704_128x128x10.py \
    --checkpoint [CKPT] \
    --train-mem --batch-size 4

# ④ PointOcc
python tools/profile_model.py \
    --config projects/configs/baselines/LiDAR_pointocc_128x128x10_server.py \
    --checkpoint [CKPT] \
    --train-mem --batch-size 4

# ⑤ FlashOcc
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \
    --checkpoint [CKPT] \
    --train-mem --batch-size 4

# ⑥ FLC-PointOcc A
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --checkpoint [CKPT] \
    --train-mem --batch-size 4

# ⑦ FLC-PointOcc B
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py \
    --checkpoint [CKPT] \
    --train-mem --batch-size 4
```

**记录什么**：输出中的 `Train peak memory : X.XX GiB`

---

## 6. 训练时长（Train Duration / GPU-H）

**工具**：从训练日志手动提取。

```bash
# 查看训练日志里的 epoch 起止时间戳
# 日志通常在 work_dirs/<name>/<timestamp>.log
tail -n 200 work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/*.log | grep -E "Epoch\(train\)|time_cost"

# 或者直接用 grep 找每个 epoch 结束行
grep "Epoch(train)" work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/*.log | head -5
grep "Epoch(train)" work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/*.log | tail -5
```

**计算方法**：
1. 记录第 1 epoch 开始时间戳和最后一个 epoch 结束时间戳
2. `Total wall time (h) = (end_ts - start_ts) / 3600`
3. 单卡训练时：`GPU-H = Total wall time × 1`
4. 双卡训练时：`GPU-H = Total wall time × 2`

也可以直接对比日志里的 `eta`（剩余时间预估）并按已训练时长反推。

---

## 7. 定性可视化（Qualitative Figures）

```bash
# 对指定 checkpoint 生成预测 .npy 文件
bash tools/dist_test.sh \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    [CKPT] 1 \
    --show --show-dir visualization_results/flc_pointocc_A

# 查看某场景的预测结果
python tools/show_npy.py \
    visualization_results/flc_pointocc_A/<scene_token>/<lidar_token>/pred_c.npy
```

场景选择建议（参考 `docs/flc_pointocc_evaluation_draft.md §5.4`）：
- 1 个正面案例：LiDAR 深度补偿明显提升 mIoU 的复杂场景
- 1 个对比案例：FLC-PointOcc vs FlashOcc（相机专注场景）
- 1 个失败案例：远距离小目标或遮挡区域的漏检

---

## 8. 快速汇总：论文效率表所需数据一览

| 数据列 | 来源脚本 | 关键输出字段 |
|---|---|---|
| #Params (M) | `profile_model.py --params` | `Total params` |
| GFLOPs | `profile_model.py --flops` | `GFLOPs (batch=1)` |
| FPS | `profile_model.py` (默认) | `FPS` |
| Infer.Mem (MiB) | `profile_model.py` (默认) | `Infer peak` |
| Train.Mem (GiB) | `profile_model.py --train-mem` | `Train peak memory` |
| Train.Duration (GPU-H) | 训练日志手动计算 | 首尾时间戳差 × GPU 数 |
| mIoU | `dist_test.sh` | `SSC_mean` |
| Per-class IoU × 17 | `dist_test.sh` | `SSC_<class>` |

---

## 附：推荐执行顺序

```text
Step 1: dist_test.sh × 7 个模型  →  获得 mIoU / per-class IoU
Step 2: profile_model.py --params × 7  →  参数量（快，几秒完成）
Step 3: profile_model.py --warmup 10 --measure 50 × 7  →  FPS + 推理显存
Step 4: profile_model.py --train-mem --batch-size 4 × 7  →  训练显存
Step 5: 手动从日志提取训练时长
Step 6: profile_model.py --flops × 7（可选，如 fvcore 安装好）
Step 7: dist_test.sh --show × 选定场景  →  定性可视化 .npy
```

---

## 注意事项

1. **Multimodal MM-OO 需要先创建 256×704 配置**：原始 `Multimodal-R50_img1600_128x128x10.py` 使用 896×1600 分辨率；需从其派生一个 256×704 版本（参见 `flc_pointocc_implementation_draft.md §3.3 [TOBECONFIRMED #5]`）。

2. **FPS 与显存受 CUDA 状态影响**：建议在其他 GPU 任务空闲时测量，`--warmup 10` 足以稳定 CUDA 内核状态。

3. **LiDAR 模型 FLOPs 统计不完整**：CylinderEncoder 使用 `torch_scatter` 的 `scatter_max` 操作，fvcore/thop 无法统计，输出的 GFLOPs 为下界估计。在论文表格中添加脚注说明。

4. **profile_model.py 使用随机 dummy 输入**：FPS 与实际数据的差异通常在 5% 以内（无 DataLoader 开销）；训练显存测量使用随机 GT，与真实训练基本一致。
