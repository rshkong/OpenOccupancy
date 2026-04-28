# FLC-Occ

**FLC-Occ** is an efficient semantic occupancy perception project for resource-constrained autonomous driving platforms. The main model in this repository is **FLC-PointOcc**, a Camera + LiDAR fusion architecture that combines a FlashOcc-style 2D BEV camera branch with a PointOcc-style LiDAR TPV geometry branch.

The design goal is practical: use low-resolution surround-view images, lightweight BEV fusion, and mostly 2D BEV decoding to reduce the cost of semantic occupancy prediction while preserving geometric and semantic accuracy.

## Highlights

- **Resource-constrained input**: all camera-based experiments use `256 x 704` multi-view images.
- **Camera branch**: LSS-based image-to-BEV lifting followed by FlashOcc-style height collapse.
- **LiDAR branch**: cylindrical point encoding and TPV representation from PointOcc.
- **Fusion strategy**: camera BEV and LiDAR BEV are fused in BEV space with lightweight adapters and a `3x3` fusion convolution.
- **Efficient decoder**: post-fusion processing uses a 2D BEV backbone, FPN-LSS neck, and a channel-to-height occupancy head.
- **Experiment focus**: compare camera-only, LiDAR-only, and Camera + LiDAR occupancy models under a unified single-GPU, low-resolution setting.

## Method Overview

FLC-PointOcc contains two parallel modality branches.

```text
Multi-view images
  -> ResNet50 + SECONDFPN
  -> LSS / ViewTransformerLSSFlash
  -> camera BEV feature [B, 640, 128, 128]

LiDAR points
  -> cylindrical point feature construction
  -> CylinderEncoder
  -> TPVSwin + TPVFPN
  -> TPVFuser
  -> LiDAR voxel feature [B, 192, 128, 128, 10]
  -> flatten height into channels
  -> LiDAR BEV feature [B, 1920, 128, 128]

Camera BEV + LiDAR BEV
  -> modality adapters
  -> concat in BEV
  -> Conv3x3 fusion
  -> CustomResNet2D + FPN_LSS
  -> FLCOccHead
  -> semantic occupancy logits [B, 17, 128, 128, 10]
```

The main motivation is to avoid a heavy shared 3D voxel decoder. The camera branch collapses height into channels before BEV decoding, while the LiDAR branch uses TPV planes to encode geometry efficiently. Fusion is performed after both modalities have been converted to BEV-compatible features.

## Repository Layout

```text
projects/configs/baselines/      Experiment configs
projects/occ_plugin/             Model, dataset, encoder, head, and hook implementations
tools/                           Training, testing, profiling, and visualization scripts
docs/                            Project notes and thesis drafts
work_dirs/                       Training logs and checkpoints
pretrain/                        External pretrained weights
data/                            Dataset files
```

## Core Implementation Files

```text
projects/occ_plugin/occupancy/detectors/flc_pointocc_net.py
projects/occ_plugin/occupancy/detectors/lidar_prep_mixin.py
projects/occ_plugin/occupancy/detectors/occnet.py
projects/occ_plugin/occupancy/image2bev/ViewTransformerLSSFlash.py
projects/occ_plugin/occupancy/lidar_encoder/cylinder_encoder.py
projects/occ_plugin/occupancy/lidar_encoder/tpv_swin.py
projects/occ_plugin/occupancy/lidar_encoder/tpv_fpn.py
projects/occ_plugin/occupancy/lidar_encoder/tpv_fuser.py
projects/occ_plugin/occupancy/dense_heads/flc_occ_head.py
```

## Main Configs

| Purpose | Config |
|---|---|
| Camera-only baseline | `projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py` |
| FlashOcc-style camera model / FLC-step2 | `projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py` |
| LiDAR-only PointOcc reproduction | `projects/configs/baselines/LiDAR_pointocc_128x128x10_server.py` |
| FLC-PointOcc Version A, camera BEV compressed to 256 channels | `projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py` |
| FLC-PointOcc Version B, camera BEV kept at 640 channels | `projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py` |
| FLC-PointOcc zero-LiDAR ablation | `projects/configs/baselines/CAM-LiDAR_flc_pointocc_zerol_128x128x10.py` |
| FLC-PointOcc 1x1 fusion ablation | `projects/configs/baselines/CAM-LiDAR_flc_pointocc_fuse1x1_128x128x10.py` |
| FLC-PointOcc project-first LiDAR ablation | `projects/configs/baselines/CAM-LiDAR_flc_pointocc_projfirst_128x128x10.py` |

Version A is the default fusion design. It compresses camera BEV from `640` to `256` channels before fusion and uses a `1920 -> 128` LiDAR adapter after height flattening. Version B keeps all `640` camera channels before fusion for the camera-compression ablation.

## Environment

The current experiments are developed under:

```text
Python 3.8
PyTorch 2.0.0
CUDA 11.8
Ubuntu 20.04
Single NVIDIA RTX 4090 24GB
```

Environment snapshots are provided in:

```text
env_conda_export.yml
env_pip_requirements.txt
```

## Data Layout

Expected local dataset layout:

```text
data/nuscenes/
data/nuScenes-Occupancy/
data/depth_gt/
```

Expected annotation files:

```text
data/nuscenes/nuscenes_occ_infos_train.pkl
data/nuscenes/nuscenes_occ_infos_val.pkl
```

## Pretrained Weights

PointOcc and FLC-PointOcc use Swin-T for the LiDAR TPV backbone. Place the checkpoint at:

```text
pretrain/swin_tiny_patch4_window7_224.pth
```

Example:

```bash
mkdir -p pretrain
wget -O pretrain/swin_tiny_patch4_window7_224.pth \
  https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

## Training

FlashOcc-style camera model:

```bash
bash run.sh projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py 1
```

PointOcc LiDAR-only:

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
  projects/configs/baselines/LiDAR_pointocc_128x128x10_server.py \
  --work-dir work_dirs/LiDAR_pointocc_server4090x2 \
  --seed 0
```

FLC-PointOcc Version A:

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
  projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \
  --work-dir work_dirs/CAM-LiDAR_flc_pointocc_camadapt256 \
  --seed 0
```

FLC-PointOcc Version B:

```bash
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
  projects/configs/baselines/CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py \
  --work-dir work_dirs/CAM-LiDAR_flc_pointocc_camfull640 \
  --seed 0
```

More command variants are maintained in:

```text
docs/training_cheatsheet.md
docs/flc_pointocc.md
```

## Evaluation

Use `tools/test.py` with the target config and checkpoint:

```bash
PYTHONPATH="./":$PYTHONPATH python tools/test.py \
  <config.py> \
  <checkpoint.pth> \
  --eval mIoU
```

Experiment notes and result tables are maintained in:

```text
docs/experiments_result.txt
docs/flc_pointocc_evaluation_draft.md
```

## Documentation

| Document | Purpose |
|---|---|
| `docs/flc_pointocc.md` | Engineering notes for FLC-PointOcc |
| `docs/flc_pointocc_methodology_draft.md` | Detailed methodology draft with formulas |
| `docs/flc_pointocc_implementation_draft.md` | Implementation / experimental setup draft |
| `docs/flc_pointocc_evaluation_draft.md` | Evaluation chapter draft and result tables |
| `docs/model_structure_guideline.md` | Model structure walkthrough |
| `docs/training_cheatsheet.md` | Practical training commands |

## Acknowledgements

This project benefits from the following open-source projects and research codebases:

- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy)
- [FlashOcc](https://github.com/Yzichen/FlashOCC)
- [PointOcc](https://github.com/wzzheng/PointOcc)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)

Their datasets, model designs, and engineering practices provide important foundations for this work.
