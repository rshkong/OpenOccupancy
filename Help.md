

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




