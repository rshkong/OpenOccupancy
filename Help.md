

没问题，为了方便你日后随时手动调试和验证，请保存以下两条核心命令。

**执行前请务必先激活环境：**
```bash
conda activate OpenOccupancy-4070
```

---

### 1. 如何调用特定的模型权重生成预测结果 (.npy)

使用官方提供的分布式测试脚本 dist_test.sh 加上 `--show` 相关参数来生成和保存 `.npy` 矩阵文件。

**通用命令格式：**
```bash
bash tools/dist_test.sh <配置文件路径> <权重文件路径> <GPU数量> --show --show-dir <自定义保存输出的目录名>
```

**你的项目实际调用示例：**
```bash
bash tools/dist_test.sh work_dirs/cam_r50_256x704/CAM-R50_img256x704_128x128x10_4070ti.py work_dirs/cam_r50_256x704/latest.pth 1 --show --show-dir visualization_results/latest_epoch1
```

**使用作者模型和原配文件推理并保存较高分辨率的 `.npy` 结果：**
```bash
# 请确保此时你在 OpenOccupancy-4070 环境下
# 这里假设你想跑 1600分辨率 的 baseline
bash tools/dist_test.sh projects/configs/baselines/CAM-R50_img1600_128x128x10.py ckpt/camera-based-baseline.pth 1 --show --show-dir visualization_results/author_baseline
```
*(注意：跑作者的模型如果尺寸激增可能会消耗更长的时间和更多的显存)*



**参数说明：**
*   **配置文件路径**: 训练时使用的 `.py` config 文件，决定了网络结构和生成范围。
*   **权重文件路径**: 你想要测试的 `.pth` checkpoint 路径。
*   **GPU数量**: 本地测试通常填 `1` 即可。
*   `--show`: 激活测试时的可视化/保存钩子逻辑，也就是触发你修改过的 `show_occ.py`。
*   `--show-dir`: 你希望把生成的 `.npy` 保存到哪个根目录下。建议像示例中那样带上 checkpoin 版本信息（如 `latest_epoch1`），这样结果就不会互相覆盖。

**`dist_test.sh` 的作用可以简洁理解为：**
*   不加 `--show` 时：主要用于跑完整个测试集并输出评估结果（如 `SC` / `SSC` 指标）。
*   加上 `--show --show-dir` 时：除了前向推理，还会把预测结果保存出来，便于后续可视化。
*   默认不会像训练那样生成大量 checkpoint 或 TensorBoard 曲线，主要输出是终端文本；只有你显式加保存参数时才会额外落盘。

> 💡 **提示**：如果你发现输出文件夹里只有 `pred_c.npy` 和 `pred_f.npy`，而没有 Ground Truth 真值（`gt.npy`），别忘了打开配置文件（`CAM-R50_img256x704_...py`），在文件接近末尾的 `test_pipeline` 字典列表里，把包含 `type='OccDefaultFormatBundle3D'` 那一行的 `with_label=False` 改为 `with_label=True`。

---

### 2. 如何训练模型

**执行前请务必先激活环境：**
```bash
conda activate OpenOccupancy-4070
```

#### 通用命令格式

训练统一通过根目录的 `run.sh` 调用（它内部会调用 `tools/dist_train.sh`）：

```bash
bash run.sh <配置文件路径> <GPU数量> [可选参数]
```

或者直接调用底层脚本：

```bash
bash tools/dist_train.sh <配置文件路径> <GPU数量> [可选参数]
```

**常用可选参数：**

| 参数 | 说明 |
|---|---|
| `--work-dir <路径>` | 指定保存 checkpoint 和日志的目录（默认自动生成为 `work_dirs/<config名>/`） |
| `--resume-from <pth路径>` | 从某个 checkpoint 继续训练（恢复 epoch 数、optimizer 状态） |
| `--load-from <pth路径>` | 加载权重作为初始化（不恢复 epoch，相当于 finetune） |
| `--no-validate` | 训练时跳过 validation（加快速度） |

**关于 checkpoint 保存：**
- 默认每个 epoch 保存一次（`checkpoint_config = dict(interval=1)`，在 `_base_/default_runtime.py` 中定义）
- 保存在 `work_dirs/<config名>/` 下，命名为 `epoch_1.pth`, `epoch_2.pth`, ... 以及 `latest.pth`
- 日志同目录下的 `.log` 文件和 TensorBoard `tf_events` 文件

---

#### 实际调用示例

**① 训练原版 OpenOccupancy baseline（CAM-R50, 256×704 输入）**
```bash
# 单卡，work_dir 自动为 work_dirs/CAM-R50_img256x704_128x128x10_4070ti/
bash run.sh projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py 1
```

**② 训练我们自己改进的 FLC-Occ（Conv2d + C2H MLP，Step 2）**
```bash
bash run.sh projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py 1
```

**③ 指定 work_dir 并跳过 validation（快速看收敛趋势）**
```bash
bash run.sh projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py 1 \
    --work-dir work_dirs/flc_step2_debug \
    --no-validate
```

**④ 中断后续训（从 epoch 3 的 checkpoint 恢复）**
```bash
bash run.sh projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py 1 \
    --resume-from work_dirs/CAM-R50_img256x704_flc_step2_128x128x10_4070ti/epoch_3.pth
```

**⑤ 只跑少量迭代验证流程是否跑通（改 max_epochs 后用 --no-validate）**
```bash
# 临时在命令行覆盖 max_epochs，mmcv 支持 --cfg-options
bash tools/dist_train.sh projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py 1 \
    --cfg-options runner.max_epochs=1 \
    --no-validate
```

---

#### 训练过程说明

- 默认共训练 **15 个 epoch**（`runner = dict(type='EpochBasedRunner', max_epochs=15)`）
- 日志每 50 个 iteration 打印一次（`log_config = dict(interval=50)`）
- 训练完成后用推理命令（见第 1 节）加载 `latest.pth` 生成预测结果

**`work_dirs/<config名>/` 里常见文件说明：**

- `epoch_1.pth`, `epoch_2.pth`, ...：每个 epoch 保存的 checkpoint，包含模型参数，通常也包含 optimizer / scheduler 状态，可用于继续训练或测试。
- `latest.pth`：软链接，始终指向当前最新的 checkpoint，测试时通常直接用它最方便。
- `best_*.pth`：按评估指标自动保存的最优模型，例如 `best_SSC_mean_epoch_2.pth` 表示在第 2 个 epoch 达到当前最佳 `SSC_mean`。
- `<时间戳>.log`：纯文本训练日志，终端里打印的关键信息基本都会写到这里，适合直接 `tail -f` 查看。
- `<时间戳>.log.json`：结构化日志，每一行通常是一条 JSON 记录，里面有 `epoch / iter / lr / loss` 等字段，适合后处理、画曲线或自己写脚本分析。
- `tf_logs/`：TensorBoard 日志目录，里面是 `events.out.tfevents...` 文件。TensorBoard 应该指向这个目录，而不是某个具体 event 文件。
- `<配置文件名>.py`：训练启动时保存的一份 config 副本，方便之后追溯当时到底用了什么参数。

**怎么理解这些日志文件：**

- 平时快速看训练是否正常：先看 `.log`
- 想画更细的 loss / lr 曲线：看 `.log.json` 或 TensorBoard 的 `tf_logs/`
- 想恢复训练或做推理：看 `latest.pth`、`epoch_*.pth`、`best_*.pth`
- 使用如下方式打开tensorboard
  `tensorboard --logdir_spec baseline:/home/shkong/MyProject/OpenOccupancy/work_dirs/cam_r50_256x704/tf_logs,flashocc:/home/shkong/MyProject/OpenOccupancy/work_dirs/CAM-R50_img256x704_flc_step2_128x128x10_4070ti/tf_logs --port 6006`

---

### 3. 如何使用 Open3D 脚本可视化某个生成的 .npy

使用我刚才帮你编写好的轻量级快捷脚 show_npy.py。

**通用命令格式：**
```bash
python tools/show_npy.py <生成的npy文件路径>
```

**实际调用示例：**
```bash
python tools/show_npy.py visualization_results/e005041f659c47e194cd5b18ea6fc346/af84a45530cf448799aefbfd2a7187ad/pred_c.npy
```

**使用作者权重的调用示例：**
```bash
# 等网络生成对应文件夹后，直接用刚才的脚本传参进去查看即可
python tools/show_npy.py visualization_results/author_baseline/任意scene_token/任意lidar_token/pred_c.npy
```
**参数说明及操作方法：**
*   **生成的npy文件路径**：通常生成在 `--show-dir` 指定的目录下，按 `<scene_token>/<lidar_token>/` 分类保存。常见的文件名有预测结果 `pred_c.npy` （有时也有 `pred_f.npy`）和真实结果 `gt.npy`。
*   **交互操作**：
    *   **鼠标左键拖拽**：旋转视角。
    *   **鼠标滚轮/右键拖拽**：缩放/平移视角。
    *   **ESC 或 Q 键**：关闭并退出当前的 3D 查看器。
*   

---

### 4. 如何验证模型并查看 SC / SSC 指标

使用官方提供的分布式测试脚本 `dist_test.sh`，不给 `--show` 时，它主要会执行评估流程并在终端 / 日志中输出 `SC`、`SSC_mean` 和各类别 `SSC_*` 指标。

**通用命令格式：**
```bash
bash tools/dist_test.sh <配置文件路径> <权重文件路径> <GPU数量>
```

**实际调用示例：**

**① 验证原版 baseline（CAM-R50, 256×704）**
```bash
bash tools/dist_test.sh projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py \
    work_dirs/cam_r50_256x704/latest.pth 1
```

**② 验证我们自己的 FLC-Occ Step 2**
```bash
bash tools/dist_test.sh projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \
    work_dirs/CAM-R50_img256x704_flc_step2_128x128x10_4070ti/latest.pth 1
```

**③ 把评估输出同时保存成文本日志**
```bash
bash tools/dist_test.sh projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py \
    work_dirs/cam_r50_256x704/latest.pth 1 | tee baseline_eval.log
```

**参数说明：**
*   **配置文件路径**：必须和该 checkpoint 训练时使用的 config 对应，否则网络结构或数据处理流程可能不匹配。
*   **权重文件路径**：通常填 `latest.pth`、`best_*.pth` 或某个 `epoch_*.pth`。
*   **GPU数量**：本地单卡验证一般填 `1`。

**你会在输出里看到什么：**
*   `SC_non-empty`：Scene Completion 指标，只看“空 / 非空”，不区分类别。
*   `SSC_mean`：Semantic Scene Completion 的总体平均指标，通常是最重要的主指标。
*   `SSC_car`、`SSC_barrier`、`SSC_pedestrian` 等：各类别的语义占据指标，数值越高越好。

**如何理解这种验证方式：**
*   这是一次性评估，默认主要输出到终端和文本日志。
*   它不像训练过程那样自动生成完整的 TensorBoard `val/*` 曲线卡片。
*   如果只是想补做 baseline 的验证，这种方式最直接、最稳妥。

**建议优先看的指标：**
*   `SC_non-empty`：判断几何 completion 能不能学到。
*   `SSC_mean`：判断整体语义 occupancy 效果。
*   `SSC_car`、`SSC_barrier`、`SSC_pedestrian`：判断关键类别是否真的有提升。

---

### 5. 如何对模型做性能分析（参数量 / FLOPs / FPS / 显存）

使用 `tools/profile_model.py` 脚本，支持项目中全部五种模型路径（OccNet-Camera、OccNet-LiDAR、OccNet-Multimodal、PointOccNet、FLCPointOccNet），用随机 dummy 输入测量推理耗时、显存，以及训练显存。脚本自动识别模型类型，无需手动指定。

**通用命令格式：**
```bash
python tools/profile_model.py [--config <配置文件>] [--checkpoint <权重文件>] [可选参数]
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--config` | FlashOcc step2 config | 模型配置文件路径 |
| `--checkpoint` | `''` | 权重文件路径（留空则用随机初始化） |
| `--mode` | `full` | `extract`：只跑到 occ encoder；`full`：跑到 occ head 输出 |
| `--warmup` | `10` | 预热帧数，不计入统计 |
| `--measure` | `50` | 正式统计帧数 |
| `--batch-size` | `1` | batch size |
| `--num-points` | `20000` | 每帧 dummy LiDAR 点数（LiDAR/融合模型适用） |
| `--seed` | `0` | 随机种子 |
| `--params` | flag | 只打印参数量后退出，不做前向推理 |
| `--flops` | flag | 估算 GFLOPs（需安装 fvcore 或 thop） |
| `--train-mem` | flag | 测量训练峰值显存（forward + backward） |

---

**实际调用示例：**

**① 仅查看参数量（所有模型通用，几秒完成）**
```bash
# FlashOcc
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \
    --checkpoint work_dirs/CAM-R50_img256x704_flc_step2_128x128x10_4070ti/best_SSC_mean.pth \
    --params

# FLC-PointOcc A（双模态融合）
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --checkpoint work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/best_SSC_mean.pth \
    --params
```

**② 推理速度 + 推理显存（标准论文数据）**
```bash
# Cam-OO (相机 baseline)
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py \
    --checkpoint work_dirs/cam_r50_256x704/best_SSC_mean.pth \
    --warmup 10 --measure 50

# LiDAR-OO（无相机输入，自动构建 LiDAR dummy 点云）
python tools/profile_model.py \
    --config projects/configs/baselines/LiDAR_128x128x10.py \
    --checkpoint work_dirs/LiDAR_128x128x10/best_SSC_mean.pth \
    --warmup 10 --measure 50

# PointOcc（LiDAR-only TPV，无 record_time，自动适配）
python tools/profile_model.py \
    --config projects/configs/baselines/LiDAR_pointocc_128x128x10_server.py \
    --checkpoint work_dirs/LiDAR_pointocc_server4090x2/best_SSC_mean.pth \
    --warmup 10 --measure 50

# FLC-PointOcc A（双模态，包含 cam_adapter + lidar_tpv + fusion 各 stage）
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --checkpoint work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/best_SSC_mean.pth \
    --warmup 10 --measure 50
```

**③ 训练显存（batch=4，和论文训练设置一致）**
```bash
python tools/profile_model.py \
    --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
    --checkpoint work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/best_SSC_mean.pth \
    --train-mem --batch-size 4
```

**④ GFLOPs 估算（需安装 fvcore）**
```bash
pip install fvcore   # 首次安装

python tools/profile_model.py \
    --config projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \
    --checkpoint work_dirs/CAM-R50_img256x704_flc_step2_128x128x10_4070ti/best_SSC_mean.pth \
    --flops
```

> **注意**：LiDAR 分支中的 `scatter_max`（CylinderEncoder）为稀疏算子，fvcore 无法统计，GFLOPs 为下界估计，在论文中需加注脚说明。

---

**输出格式说明：**

```
Detected model kind : flc_pointocc  (FLCPointOccNet)
Total params        : XX.XX M
Trainable params    : XX.XX M

========================================================================
Forward Inference Profiling
  avg forward : XX.XX ms
  FPS         : X.XX            ← 论文 FPS 列
  Infer peak  : XXX.X MiB       ← 论文推理显存列
========================================================================

Stage                  Avg Time (ms)   Share
------------------------------------------------
img_encoder                 XX.XX      XX.X%   ← ResNet50 + SECONDFPN
view_transformer            XX.XX      XX.X%   ← LSS + 深度估计
cam_adapter                 XX.XX      XX.X%   ← Conv1x1 640→256 (A版有, B版无)
lidar_tpv                   XX.XX      XX.X%   ← CylinderEnc + Swin + TPVFuser
lidar_adapter               XX.XX      XX.X%   ← Conv1x1 1920→128
fusion                      XX.XX      XX.X%   ← Conv3x3 BN ReLU
occ_encoder                 XX.XX      XX.X%   ← CustomResNet2D + FPN_LSS
occ_head                    XX.XX      XX.X%   ← FLCOccHead (C2H MLP)

Stage                  Alloc Δ    Stage Peak Δ        After
----------------------------------------------------------------
img_encoder            X.XX MiB      X.XX MiB      X.XX MiB
...
```

各 stage 因模型类型不同而有所差异：
- **OccNet-Camera**（Cam-OO, FlashOcc）：`img_encoder → view_transformer → occ_encoder → occ_head`
- **OccNet-LiDAR**（LiDAR-OO）：`pts_encoder → occ_encoder → occ_head`
- **OccNet-Multimodal**（MM-OO）：`img_encoder → view_transformer → pts_encoder → occ_fuser → occ_encoder → occ_head`
- **PointOccNet**（PointOcc）：`lidar_tpv → tpv_aggregator`
- **FLCPointOccNet**（FLC-PointOcc A/B）：`img_encoder → view_transformer → cam_adapter → lidar_tpv → lidar_adapter → fusion → occ_encoder → occ_head`（cam_adapter 在 B 版本中不存在）

完整的论文数据收集流程见 `docs/evaluation_cheatsheet.md`。

