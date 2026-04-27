# FLC-PointOcc — Implementation / Experimental Setup Draft

本文档是面向论文写作的 **Implementation（实验设置）章节**初稿，刻意详尽，后续再删减精炼。所有 `[TOBECONFIRMED]` 标记表示需要人工核查后再确认，`[XX.X]` 表示需要填入实验数字的占位符。

---

## 1. Dataset

All experiments are conducted on the **nuScenes** dataset [CITATION: Caesar et al., 2020], a large-scale autonomous driving benchmark collected in Boston and Singapore using a vehicle equipped with six surround-view cameras and a 32-beam LiDAR sensor. The dataset comprises [TOBECONFIRMED: 700 training / 150 validation / 150 test] scenes, each approximately 20 seconds in duration at 2 Hz keyframe annotation rate, yielding approximately [TOBECONFIRMED: 28,130 training and 6,019 validation] samples.

For occupancy prediction, we use the **nuScenes-Occupancy** annotation provided by the OpenOccupancy benchmark [CITATION: Wang et al., 2023]. The ground-truth occupancy volume is defined over the ego-vehicle's coordinate frame as a 3D Cartesian grid:

$$
\mathbf{Y}^{gt} \in \{0, 1, \ldots, C-1, 255\}^{512 \times 512 \times 40},
$$

covering the spatial range $[-51.2, 51.2] \times [-51.2, 51.2] \times [-5.0, 3.0]$ metres along $x$, $y$, and $z$ axes respectively, at a voxel resolution of $0.2\ \text{m} \times 0.2\ \text{m} \times 0.2\ \text{m}$. The label value 255 denotes ignored / unlabelled voxels. There are $C = 17$ classes in total, consisting of one free (empty) class and 16 semantic object/surface classes: car, truck, construction vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic cone, bicycle rack [TOBECONFIRMED: verify full class list against OpenOccupancy annotation], driveable surface, other flat, sidewalk, terrain, and manmade / vegetation.

All models predict on a **coarse grid** of $128 \times 128 \times 10$ voxels, obtained by applying a $4 \times 4 \times 4$ spatial downsampling to the full-resolution label. The downsampling follows an occupancy-aware aggregation rule: within each $4 \times 4 \times 4$ block, if any occupied semantic label is present, it is preferentially assigned to the coarse voxel; otherwise the free label is retained. This prevents occupied thin structures from vanishing under naive majority voting.

**LiDAR input.** We accumulate $T=10$ consecutive LiDAR sweeps into a single point cloud per sample, transforming all points into the current ego frame using calibrated sensor poses. Each point is represented by five raw channels: $(x, y, z, \iota, r)$, where $\iota$ is the laser return intensity and $r$ is the laser ring index. The accumulated point cloud typically contains on the order of $10^5$ points per frame [TOBECONFIRMED].

**Camera input.** Six surround-view cameras provide a $360°$ horizontal field of view. The original raw image resolution is $900 \times 1600$ pixels. In this work, all camera-based methods uniformly use an input resolution of $\mathbf{256 \times 704}$ pixels, motivated by the target deployment scenario of **resource-constrained embedded platforms** (e.g., onboard compute units of small autonomous vehicles). This reduction in resolution also enables single-GPU training for all methods, facilitating reproducible comparison.

---

## 2. Evaluation Metrics

We adopt the standard OpenOccupancy evaluation protocol. The primary metric is the **mean Intersection over Union (mIoU)** over all $C = 17$ semantic classes, computed on the coarse $128 \times 128 \times 10$ prediction grid against the downsampled ground-truth. Voxels marked as ignore (value 255) are excluded from evaluation.

For class $c$, the per-class IoU is defined as:

$$
\text{IoU}_c = \frac{|\mathcal{V}_c^{pred} \cap \mathcal{V}_c^{gt}|}{|\mathcal{V}_c^{pred} \cup \mathcal{V}_c^{gt}|},
$$

where $\mathcal{V}_c^{pred}$ and $\mathcal{V}_c^{gt}$ denote the sets of voxels predicted and labelled as class $c$, respectively. The overall mIoU is the unweighted mean across all 17 classes:

$$
\text{mIoU} = \frac{1}{C} \sum_{c=0}^{C-1} \text{IoU}_c.
$$

In the codebase, this is also referred to as `SSC_mean` (Semantic Scene Completion mean IoU), following the scene completion literature convention.

In addition to the overall mIoU, we report **per-class IoU** values to facilitate fine-grained analysis of which object categories benefit most from LiDAR augmentation.

---

## 3. Baselines

We compare FLC-PointOcc against the following six baselines. Crucially, **all baselines are retrained from scratch in our unified experimental environment** — single NVIDIA RTX 4090 GPU, OpenOccupancy evaluation framework — rather than reporting numbers from original publications. This is necessary for two reasons:

1. **Inconsistent evaluation protocols.** FlashOcc and PointOcc were originally trained and evaluated outside the OpenOccupancy benchmark, with different label conventions, voxel resolutions, and metric implementations. Directly comparing across these heterogeneous setups would be unfair.
2. **Unavailable low-resolution configurations.** The OpenOccupancy baselines were originally trained at a higher camera resolution ($896 \times 1600$ or $1600 \times 900$); no published models or results exist for the $256 \times 704$ setting used here.

### 3.1 OpenOccupancy Camera Baseline (Cam-OO)

The camera-only baseline from the OpenOccupancy framework [CITATION]. It follows a standard LSS-based pipeline: a ResNet-50 backbone with SECONDFPN neck extracts per-camera features; a BEVDepth-style view transformer lifts them into 3D voxel space; a 3D ResNet-18 (`CustomResNet3D`) with `FPN3D` neck encodes the occupancy volume; and an `OccHead` decodes class logits. The `OccHead` includes a coarse-to-fine refinement stage, which is disabled in all experiments (`sample_from_voxel=False`, `sample_from_img=False`). The BEV channel width in the view transformer is $C_{trans} = 80$. Camera input resolution is $256 \times 704$ (our adaptation from the original $896 \times 1600$). This baseline uses 3D volumetric convolutions for occupancy decoding and does not apply the Channel-to-Height (C2H) trick.

### 3.2 OpenOccupancy LiDAR Baseline (LiDAR-OO)

The LiDAR-only baseline from the OpenOccupancy framework [CITATION]. Raw LiDAR points are voxelized by a hard voxelization step (`HardSimpleVFE`, voxel size $0.1\ \text{m}$, max 10 points per voxel). A sparse encoder `SparseLiDAREnc8x` (base channels $= 16$, output channels $= 80$) produces a 3D sparse feature volume. A 3D ResNet-18 (`CustomResNet3D`, depth $= 18$) followed by `FPN3D` encodes multi-scale features, and an `OccHead` decodes semantic occupancy. No camera input is used.

### 3.3 OpenOccupancy Multimodal Baseline (MM-OO)

The multi-modal fusion baseline from the OpenOccupancy framework [CITATION]. The camera branch is identical to Cam-OO; the LiDAR branch is identical to LiDAR-OO. The two modalities are fused in 3D voxel space before being passed through a shared 3D encoder and `OccHead`. Camera input resolution is $256 \times 704$ in our experiments [TOBECONFIRMED: confirm that an adapted 256×704 Multimodal config was created and used; original config in `Multimodal-R50_img1600_128x128x10.py` uses 896×1600].

### 3.4 PointOcc (LiDAR-only)

PointOcc [CITATION: TOBECONFIRMED] is a LiDAR-only occupancy prediction method based on the Tri-Perspective View (TPV) representation. Raw points are encoded into cylindrical features by `CylinderEncoder` (cylindrical grid $[480, 360, 32]$, split factor $[8, 8, 8]$, output channels 128). Three TPV planes are extracted by max-pooling along each axis and encoded by a shared Swin Transformer backbone (Swin-T, ImageNet pretrained). A `TPVFPN` neck produces 192-channel features per plane, and a `TPVAggregator` performs query-based sampling at each coarse occupancy voxel center, summing the three plane contributions, and applies a linear classifier to predict 17-class logits directly from the 3D feature volume.

In the original PointOcc paper, training and evaluation used a different framework and metric convention. We port PointOcc into the OpenOccupancy codebase, adapting it to: (i) the same 17-class nuScenes-Occupancy labels, (ii) coarse prediction grid $128 \times 128 \times 10$, (iii) BEV data augmentation (scale and flip) consistent with other methods, and (iv) the unified OpenOccupancy evaluation pipeline. The LiDAR recipe (cylindrical grid size, Swin-T pretraining, optimizer) is otherwise kept identical to the original.

### 3.5 FlashOcc (Camera-only, 256×704)

FlashOcc [CITATION: TOBECONFIRMED] is a camera-only BEV occupancy method that avoids 3D convolutions by collapsing the height axis into BEV channels. Our reproduction follows the FlashOcc design within the OpenOccupancy codebase, referred to internally as **FLC-step2**. The pipeline consists of:

- ResNet-50 backbone + SECONDFPN neck → per-camera features $[B, 6, 512, 16, 44]$,
- `ViewTransformerLSSFlash` (BEVDepth-style depth net, $C_{trans} = 64$, $D_z = 10$ height bins) → BEV feature map $[B, 640, 128, 128]$,
- `CustomResNet2D` + `FPN_LSS` → multi-scale 2D BEV features,
- `FLCOccHead` (Conv2D + Channel-to-Height MLP) → coarse occupancy logits $[B, 17, 128, 128, 10]$.

All processing after the Z-collapse is purely 2D, making this method computationally efficient. Camera input is $256 \times 704$. Depth supervision is applied with weight $\lambda_{depth} = 3$.

### 3.6 FLC-PointOcc (Ours)

Our proposed dual-modal method, described in Section~\ref{sec:method}. The default configuration is **Version A** (`camadapt256`), in which the camera BEV is compressed from 640 to 256 channels before fusion. An ablation variant **Version B** (`camfull640`) retains the full 640-channel camera BEV. See Section~\ref{sec:ablation} for the comparison.

---

## 4. Implementation Details

### 4.1 Hardware and Software Environment

All experiments are conducted on a single **NVIDIA GeForce RTX 4090** (24 GB VRAM). The host machine is equipped with a **32-core Intel Xeon Platinum 8358P @ 2.60 GHz** CPU and runs **Ubuntu 20.04**. The software stack is:

| Component | Version |
|---|---|
| Python | 3.8 |
| PyTorch | 2.0.0 |
| CUDA | 11.8 |
| mmdetection3d | [TOBECONFIRMED: check exact version used in the project] |
| torch_scatter | [TOBECONFIRMED: check exact version; required for CylinderEncoder] |

The OpenOccupancy codebase [CITATION] serves as the training and evaluation framework for all methods.

We note that the original OpenOccupancy baselines were trained on **multiple GPUs** (typically 4–8 GPUs, effective batch size 16–32). In contrast, **all methods in this paper are trained on a single GPU** with a batch size of 4. While this reduces the effective batch size relative to the original setting, the unified hardware setup ensures that all methods are compared under identical computational constraints, which is the relevant criterion for our target deployment scenario.

### 4.2 Optimizer and Learning Rate Schedule

All models are trained with the **AdamW** optimizer [CITATION: Loshchilov & Hutter, 2019]:

$$
\theta_{t+1} = \theta_t - \alpha_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha_t \lambda \theta_t,
$$

with base learning rate $\alpha = 2 \times 10^{-4}$, weight decay $\lambda = 0.01$, and default momentum parameters $(\beta_1, \beta_2) = (0.9, 0.999)$.

For camera-based backbones (ResNet-50), which are initialized from ImageNet pretrained weights, a **learning rate multiplier of $0.1$** is applied, giving an effective learning rate of $2 \times 10^{-5}$ for the image backbone parameters. All other parameters use the base learning rate.

Gradient clipping is applied with an $\ell_2$ norm threshold of 35.

The learning rate follows a **Cosine Annealing** schedule:

$$
\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha - \alpha_{min})\left(1 + \cos\frac{\pi t}{T}\right),
$$

with minimum learning rate $\alpha_{min} = \alpha \cdot r_{min}$, where $r_{min} = 10^{-3}$. A **linear warmup** is applied for the first 500 iterations, ramping the learning rate from $\frac{\alpha}{3}$ to $\alpha$.

All models are trained for **24 epochs** with a batch size of 4 (single GPU, no gradient accumulation). Mixed-precision training (FP16) is enabled via `GradientCumulativeFp16OptimizerHook`.

### 4.3 Data Augmentation

**Camera augmentation** is applied consistently across all camera-based methods:

| Augmentation | Parameter |
|---|---|
| Random resize | ratio uniformly sampled from $[-0.06, +0.11]$ relative to base scale |
| Random rotation | angle uniformly sampled from $[-5.4°, +5.4°]$ |
| Random horizontal flip | applied with probability 0.5 |
| Random crop height | $(0.0, 0.0)$ (no vertical crop) |

**BEV augmentation** is applied to the BEV frame for methods that produce BEV features (all camera-based and multi-modal methods):

| Augmentation | Parameter |
|---|---|
| Random BEV scale | scale factor uniformly sampled from $[0.95, 1.05]$ |
| Random flip along $x$ axis | applied with probability 0.5 |
| Random flip along $y$ axis | applied with probability 0.5 |
| Random BEV rotation | $[0°, 0°]$ (disabled) |

LiDAR-only methods (LiDAR-OO) do not use camera augmentation. The LiDAR-only PointOcc also applies BEV augmentation with identical parameters (scale $[0.95, 1.05]$, flip_dx/flip_dy $p=0.5$, no rotation).

### 4.4 Pre-trained Weights

**ResNet-50** (camera backbone): initialized from the ImageNet pretrained weights provided by `torchvision` (`torchvision://resnet50`), used in all camera-based methods.

**Swin Transformer Tiny** (LiDAR backbone, TPVSwin): initialized from the ImageNet pretrained checkpoint `swin_tiny_patch4_window7_224.pth` (from the official Swin Transformer repository), used in PointOcc and FLC-PointOcc.

No other pre-trained weights are used. The fusion modules (cam_adapter, lidar_adapter, fuse_conv) in FLC-PointOcc are initialized randomly.

### 4.5 Training Details per Method

The table below summarizes the key architectural and training differences across methods. Unless otherwise noted, all methods share the same optimizer, LR schedule, epochs, and evaluation setup described above.

| Method | Backbone | View Transform | BEV Encoder | Occ Head | Image Res. | Loss Norm |
|---|---|---|---|---|---|---|
| Cam-OO | ResNet50 + SECONDFPN | LSS (BEVDepth), $C_t=80$ | CustomResNet3D + FPN3D | OccHead (3D) | 256×704 | True |
| LiDAR-OO | — | HardVoxel + SparseLiDAREnc8x | CustomResNet3D + FPN3D | OccHead (3D) | — | True |
| MM-OO | ResNet50 + SECONDFPN | LSS (BEVDepth), $C_t=80$ | CustomResNet3D + FPN3D | OccHead (3D) | 256×704 [TOBECONFIRMED] | True |
| PointOcc | — | CylinderEnc + TPVSwin + TPVFPN | TPVAggregator (direct) | Linear classifier | — | — |
| FlashOcc | ResNet50 + SECONDFPN | LSSFlash (C2H), $C_t=64$ | CustomResNet2D + FPN_LSS | FLCOccHead (C2H) | 256×704 | False |
| **FLC-PointOcc (A)** | ResNet50 + TPVSwin | LSSFlash + TPV fusion | CustomResNet2D + FPN_LSS | FLCOccHead (C2H) | 256×704 | False |

**Note on `loss_norm`:** The OpenOccupancy baselines (Cam-OO, LiDAR-OO, MM-OO) use `loss_norm=True`, which normalises each loss term by its own value before summing. FlashOcc and FLC-PointOcc use `loss_norm=False`, which is the standard multi-term weighted sum. This difference originates from the respective original codebases and is preserved in our reproduction [TOBECONFIRMED: confirm whether this difference is intentional or should be unified].

### 4.6 FLC-PointOcc Architecture Configuration

For completeness, the precise channel widths and spatial resolutions at each stage of **FLC-PointOcc Version A** are:

**Camera branch:**

| Stage | Module | Output Shape |
|---|---|---|
| Image encoding | ResNet-50 + SECONDFPN | $[B, 6, 512, 16, 44]$ |
| Depth & context | BEVDepth DepthNet | depth: $[B, 6, 112, 16, 44]$; context: $[B, 6, 64, 16, 44]$ |
| View transform | ViewTransformerLSSFlash | $[B, 64, 128, 128, 10]$ |
| Z-collapse | Concat along $z$ → channels | $[B, 640, 128, 128]$ |
| Camera adapter | Conv1×1, BN, ReLU | $[B, 256, 128, 128]$ |

**LiDAR branch:**

| Stage | Module | Output Shape |
|---|---|---|
| Point preprocessing | 10-ch cylindrical features + cart2polar | $[M, 10]$ |
| Cylindrical encoding | CylinderEncoder (split=[8,8,8]) | TPV planes: $[B, 128, H_i, W_i]$ |
| Swin backbone | TPVSwin (shared weights, Swin-T) | Multi-scale per plane |
| TPV FPN | TPVFPN | $\tilde{T}_{xy/yz/zx} \in \mathbb{R}^{192 \times h \times w}$, $(h, w) = (240, 180), (180, 16), (16, 240)$ |
| 3D lifting | TPVFuser (grid_sample + sum) | $[B, 192, 128, 128, 10]$ |
| Z-flatten | Permute + reshape | $[B, 1920, 128, 128]$ |
| LiDAR adapter | Conv1×1, BN, ReLU | $[B, 128, 128, 128]$ |

**Fusion and decoding:**

| Stage | Module | Output Shape |
|---|---|---|
| Concatenation | torch.cat | $[B, 384, 128, 128]$ |
| Fusion conv | Conv3×3, BN, ReLU | $[B, 256, 128, 128]$ |
| BEV encoder | CustomResNet2D + FPN_LSS | $[B, 256, 128, 128]$ |
| Occ head | FLCOccHead (Conv2D + C2H MLP) | $[B, 17, 128, 128, 10]$ |

### 4.7 Loss Configuration

FLC-PointOcc and FlashOcc are trained with a combined loss:

$$
\mathcal{L} = \lambda_{d} \mathcal{L}_{depth} + \lambda_{ce} \mathcal{L}_{ce} + \lambda_{sem} \mathcal{L}_{sem} + \lambda_{geo} \mathcal{L}_{geo} + \lambda_{lov} \mathcal{L}_{lovasz},
$$

with $\lambda_d = 3$ and $\lambda_{ce} = \lambda_{sem} = \lambda_{geo} = \lambda_{lov} = 1.0$. Depth loss is applied only to the camera branch.

The OpenOccupancy baselines (Cam-OO, LiDAR-OO, MM-OO) use an equivalent set of occupancy losses ($\mathcal{L}_{ce}$, $\mathcal{L}_{sem}$, $\mathcal{L}_{geo}$, $\mathcal{L}_{lovasz}$) following the original OpenOccupancy framework. The camera baseline additionally uses depth supervision. PointOcc uses $\mathcal{L}_{ce} + \mathcal{L}_{lovasz} + \mathcal{L}_{sem} + \mathcal{L}_{geo}$ with all weights equal to 1.0, without depth supervision.

### 4.8 Dataloader Configuration

`workers_per_gpu = 2` for all fusion methods. A larger value (e.g., 8) would cause excessive RAM usage because each worker pre-fetches 6-camera images, 10-sweep LiDAR point clouds, and $512 \times 512 \times 40$ occupancy GT simultaneously; empirically, 2 workers stabilise RAM around 11 GiB whereas 8 workers can consume 100–200 GiB.

Camera-only methods (Cam-OO, FlashOcc) also use `workers_per_gpu = 2` in our experiments, consistent with all other methods.

---

## 5. Items to Be Confirmed Before Finalising

Below is a consolidated checklist of all [TOBECONFIRMED] items from the above sections:

| # | Location | Item to Confirm |
|---|---|---|
| 1 | §1 Dataset | Exact nuScenes train/val/test split: 700/150/150 scenes? |
| 2 | §1 Dataset | Approximate keyframe count: ~28,130 train / ~6,019 val? |
| 3 | §1 Dataset | Full 16 semantic class names against OpenOccupancy annotation |
| 4 | §1 Dataset | Typical point count per accumulated 10-sweep frame (~10^5?) |
| 5 | §3.3 MM-OO | Confirm an adapted 256×704 Multimodal config exists and was/will be run (`Multimodal-R50_img1600_128x128x10.py` currently uses 896×1600) |
| 6 | §4.1 Software | Exact version of mmdetection3d used |
| 7 | §4.1 Software | Exact version of torch_scatter used |
| 8 | §4.5 Loss Norm | Decide whether to unify `loss_norm` setting across all baselines for full fairness, or keep as-is and explicitly note the difference |
