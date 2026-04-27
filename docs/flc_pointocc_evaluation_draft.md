# FLC-PointOcc — Evaluation / Experiments Chapter Draft

本文档是面向论文写作的 **Evaluation（实验分析与结果）章节**初稿，刻意详尽，后续再删减精炼。`[TOBECONFIRMED]` 表示需要人工核查，`[XX.X]` 是实验数字占位符，`[REF]` 是引用占位符。

---

## 5. Experiments

In this section, we present a comprehensive evaluation of FLC-PointOcc against state-of-the-art baselines on the nuScenes-Occupancy benchmark. We first compare quantitative semantic occupancy accuracy (Section~\ref{sec:comparison}), then analyse resource efficiency from multiple dimensions (Section~\ref{sec:efficiency}), and finally conduct ablation studies to verify each key design decision (Section~\ref{sec:ablation}). Qualitative visualisations are provided in Section~\ref{sec:qualitative}.

---

## 5.1 Main Quantitative Comparison

### 5.1.1 Results

Table~\ref{tab:main} reports the semantic occupancy performance of all methods on the nuScenes-Occupancy validation set. We report the **geometric IoU** (binary occupied vs. free), the **semantic mIoU** over all 17 classes, and the **per-class IoU** for each of the 16 foreground categories.

> **[PLACEHOLDER TABLE — fill in actual numbers after training completes]**

| Method | Modality | Image Res. | IoU | mIoU | barrier | bicycle | bus | car | cons. veh. | motorcycle | pedestrian | traffic cone | trailer | truck | drive. sur. | other flat | sidewalk | terrain | manmade | vegetation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cam-OO [REF] | C | 256×704 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| LiDAR-OO [REF] | L | — | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| MM-OO [REF] | C+L | 256×704 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| PointOcc [REF] | L | — | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| FlashOcc [REF] | C | 256×704 | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| **FLC-PointOcc (Ours)** | **C+L** | **256×704** | **[XX.X]** | **[XX.X]** | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |

*Table 1. Semantic occupancy performance on nuScenes-Occupancy validation set. C: camera, L: LiDAR, C+L: multi-modal. All methods are retrained under a unified single-GPU setup with identical evaluation protocol (see Section~\ref{sec:impl}). Image resolution 256×704 is used for all camera-based methods, targeting resource-constrained deployment scenarios.*

### 5.1.2 Analysis

**Overall performance.** FLC-PointOcc achieves [XX.X] mIoU, outperforming the camera-only baseline FlashOcc by [+XX.X] mIoU ([+XX.X]%) and the camera-only OpenOccupancy baseline by [+XX.X] mIoU. [FILL IN after results are available]

**Modality complementarity.** Consistent with prior work [REF: OpenOccupancy], camera-based and LiDAR-based representations exhibit complementary strengths across semantic categories. As shown in Table~\ref{tab:main}:

- **LiDAR-dominant categories**: large flat structures such as *driveable surface*, *sidewalk*, and *terrain* are more accurately predicted by LiDAR-based and multi-modal methods, owing to the dense and precise geometric measurements of the spinning LiDAR. [FILL IN with specific IoU deltas after results.]
- **Camera-dominant categories**: small and visually distinctive objects such as *bicycle*, *pedestrian*, *motorcycle*, and *traffic cone* benefit disproportionately from camera semantics, since the surround-view images provide dense RGB texture cues that are sparse in LiDAR point clouds. [FILL IN with specific IoU deltas after results.]
- FLC-PointOcc, by fusing both modalities in BEV, captures the advantages of both, achieving strong performance across both category groups. [FILL IN.]

**Comparison within multi-modal methods.** FLC-PointOcc outperforms/[TOBECONFIRMED: compare with] the OpenOccupancy multi-modal baseline (MM-OO) by [+XX.X] mIoU ([+XX.X]%), while being significantly more computationally efficient (see Section~\ref{sec:efficiency}). This improvement stems from the stronger per-modality encoders: MM-OO uses a conventional voxel-based 3D convolution pipeline for both branches, whereas FLC-PointOcc replaces them with the more efficient FlashOcc-style 2D BEV camera encoder and the TPV-based LiDAR encoder, both avoiding expensive 3D convolutions in the shared decoding stage.

**Comparison with LiDAR-only methods.** FLC-PointOcc [outperforms/matches] PointOcc by [+XX.X] mIoU [FILL IN], demonstrating that camera features provide complementary semantic information beyond what LiDAR geometry alone can offer, particularly for small objects and visually complex surfaces. The improvement on categories such as *bicycle* (+[XX.X]) and *pedestrian* (+[XX.X]) is especially notable. [FILL IN.]

---

## 5.2 Resource Efficiency Analysis

A primary motivation for FLC-PointOcc is efficient deployment on resource-constrained platforms, such as the embedded compute units of small autonomous vehicles. We therefore conduct a multi-dimensional efficiency analysis, encompassing model size, computational cost, inference speed, memory footprint, and training cost.

### 5.2.1 Efficiency Metrics

All efficiency measurements are conducted under the following conditions:
- **Hardware**: single NVIDIA RTX 4090 (24 GB VRAM), identical to training hardware.
- **Input**: single-sample inference (batch size = 1), using the standard 256×704 camera resolution and 10-sweep LiDAR input.
- **FPS and latency**: measured in PyTorch with [TOBECONFIRMED: FP32 / FP16 mixed precision — specify], averaged over [TOBECONFIRMED: 100 / 200] validation samples after [TOBECONFIRMED: 50-sample] warm-up.
- **GFLOPs**: computed using [TOBECONFIRMED: `thop` / `fvcore` / `ptflops` — specify the tool used] at batch size 1.
- **Inference GPU memory**: peak GPU memory consumption during a single forward pass, measured with `torch.cuda.max_memory_allocated()`.
- **Training GPU memory**: peak GPU memory during a training step (forward + backward) at batch size 4.
- **Training duration**: total wall-clock time in GPU-Hours (GPU × hours) to train for 24 epochs on the full nuScenes-Occupancy training set with a single RTX 4090.

### 5.2.2 Efficiency Table

> **[PLACEHOLDER TABLE — fill in actual numbers after measurements]**

| Method | Modality | #Params (M) | GFLOPs | FPS (Hz) | Infer. Mem. (MiB) | Train. Mem. (GB) | Train. Duration (GPU-H) | mIoU |
|---|---|---|---|---|---|---|---|---|
| Cam-OO | C | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| LiDAR-OO | L | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| MM-OO | C+L | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| PointOcc | L | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| FlashOcc | C | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] | [XX.X] |
| **FLC-PointOcc (Ours)** | **C+L** | **[XX.X]** | **[XX.X]** | **[XX.X]** | **[XX.X]** | **[XX.X]** | **[XX.X]** | **[XX.X]** |

*Table 2. Efficiency comparison on nuScenes-Occupancy. FPS is measured in PyTorch [FP32/FP16] on a single RTX 4090. "Infer. Mem." is peak GPU memory during a single forward pass. "Train. Mem." is peak GPU memory during training at batch size 4. "Train. Duration" is total training cost in GPU×Hours for 24 epochs.*

### 5.2.3 Analysis

**Accuracy–efficiency trade-off.** Figure~\ref{fig:tradeoff} plots mIoU against inference FPS and training GPU-H for all methods, visualising the accuracy–efficiency frontier. FLC-PointOcc [FILL IN: describe where it sits on the Pareto front relative to baselines]. Compared to the OpenOccupancy multi-modal baseline (MM-OO), which uses full 3D voxel convolutions, FLC-PointOcc achieves a superior mIoU while reducing GFLOPs by [XX.X]× and inference memory by [XX.X]%.

> **[TODO: create Figure — scatter plot of mIoU vs FPS, with bubble size encoding #Params or Training GPU-H. Each method is one bubble. Separate curves for camera-only, LiDAR-only, and multi-modal methods.]**

**Why FLC-PointOcc is efficient despite being multi-modal.** The key efficiency advantage comes from the design principle of performing all shared computation in 2D BEV rather than 3D voxel space:

1. The camera branch follows FlashOcc, collapsing the height axis into channels before any BEV encoding, eliminating 3D convolutions entirely on the camera side.
2. The LiDAR branch uses the TPV representation, avoiding a full 3D dense tensor: the three TPV planes are encoded independently (2D), and the 3D tensor is only reconstructed at the final TPV-to-voxel lifting step — which is a lightweight `grid_sample` operation with no learned parameters.
3. The Z-flatten + Conv1×1 LiDAR adapter (1920→128 channels) replaces what would otherwise require a 3D convolutional stack to reduce spatial resolution.
4. The post-fusion decoder (CustomResNet2D + FPN_LSS + FLCOccHead) operates entirely in 2D BEV.

As a result, [FILL IN: describe the specific FLOPs / memory savings compared to MM-OO after numbers are available].

**Parameter count.** FLC-PointOcc has [XX.X] M parameters in total, compared to [XX.X] M (Cam-OO), [XX.X] M (LiDAR-OO), [XX.X] M (MM-OO), [XX.X] M (PointOcc), and [XX.X] M (FlashOcc). [FILL IN and analyse.]

**Training cost.** Training FLC-PointOcc for 24 epochs on a single RTX 4090 takes approximately [XX.X] GPU-H. In contrast, MM-OO requires [XX.X] GPU-H under the same single-GPU setting [FILL IN]. This reduction in training cost, combined with the single-GPU requirement, makes FLC-PointOcc significantly more accessible to researchers and developers without access to large GPU clusters.

---

## 5.3 Ablation Studies

We conduct ablation experiments to verify the contribution of each major design choice in FLC-PointOcc. Unless otherwise noted, all ablations use Version A as the default and are evaluated on the nuScenes-Occupancy validation set.

### 5.3.1 Effect of Camera Channel Compression (Version A vs. Version B)

The first ablation examines whether compressing the camera BEV feature from 640 to 256 channels before fusion (Version A) loses information compared to retaining the full 640 channels (Version B).

| Variant | cam_adapter | Fusion Input | #Params (M) | GFLOPs | mIoU |
|---|---|---|---|---|---|
| Version A (default) | Conv1×1, 640→256 | 384 ch | [XX.X] | [XX.X] | [XX.X] |
| Version B | None | 768 ch | [XX.X] | [XX.X] | [XX.X] |

*Table 3. Ablation: camera channel compression. Both variants share identical LiDAR branch, fusion conv (3×3), post-fusion BEV encoder, and occupancy head.*

[FILL IN analysis after results. Expected narrative: if A ≈ B, conclude that the camera adapter introduces minimal information loss while saving parameters and compute. If B > A by a margin, note the trade-off and discuss whether the cost is justified. If A > B, the compression acts as a beneficial regulariser.]

**Discussion.** Version A follows the "front-light/back-heavy" principle of BEVFusion [REF], keeping modality adapters shallow and concentrating capacity in the shared post-fusion encoder. The cam_adapter Conv1×1 has [XX.X] K parameters, a negligible cost. Version B requires a wider fuse_conv (768→256 instead of 384→256), increasing parameters and FLOPs for the fusion stage but providing the post-fusion encoder with all original camera channels uncompressed.

### 5.3.2 LiDAR Contribution Verification

To quantify the benefit of the LiDAR branch, we compare FLC-PointOcc against two reference points: (i) FlashOcc (our FLC-step2), which is architecturally equivalent to FLC-PointOcc with the LiDAR branch entirely bypassed, and (ii) a `debug_zero_lidar` variant in which the LiDAR branch is present but its output is zeroed out before fusion, verifying that the camera signal passes through the fusion pathway without obstruction.

| Method | LiDAR Branch | LiDAR Contribution | mIoU | $\Delta$ vs FlashOcc |
|---|---|---|---|---|
| FlashOcc (FLC-step2) | Off (bypassed) | None | [XX.X] | — |
| FLC-PointOcc (zero LiDAR) | On, output zeroed | None (control) | [XX.X] | [+XX.X] |
| **FLC-PointOcc (Ours, Version A)** | **On, full** | **Full** | **[XX.X]** | **[+XX.X]** |

*Table 4. LiDAR contribution ablation. "FLC-step2" is the FlashOcc-equivalent camera-only model. "Zero LiDAR" verifies that the fusion path does not degrade camera performance in the absence of LiDAR signal.*

[FILL IN analysis. Expected narrative: FLC-PointOcc > FlashOcc by a meaningful margin, confirming the LiDAR branch adds genuine value. Zero-LiDAR ≈ FlashOcc confirms that fuse_conv does not damage the camera signal pathway — the architecture degrades gracefully when LiDAR is absent.]

### 5.3.3 Z-Flatten Strategy: Flatten-First vs. Project-First

A key LiDAR branch design choice is the order in which Z (height) is collapsed and channel dimensionality is reduced. We compare two strategies:

- **Flatten-first** (default): $[B, 192, X, Y, Z] \xrightarrow{\text{reshape}} [B, 1920, X, Y] \xrightarrow{\text{Conv1×1}} [B, 128, X, Y]$. The projection matrix $\mathbf{W} \in \mathbb{R}^{128 \times 1920}$ assigns **independent weights** to each height bin.
- **Project-first**: $[B, 192, X, Y, Z] \xrightarrow{\text{Linear}_{Z}} [B, C', X, Y, Z] \xrightarrow{\text{reshape}} [B, C' Z, X, Y]$. All height bins share the same per-channel linear projection, which is more restrictive.

| Strategy | Projection Params | mIoU |
|---|---|---|
| Project-first (Linear 192→64 per voxel) | ~12 K | [XX.X] |
| **Flatten-first (Conv1×1 1920→128, default)** | **~246 K** | **[XX.X]** |

*Table 5. Ablation: Z-flatten order in the LiDAR BEV projection.*

[FILL IN analysis. Expected narrative: flatten-first is better because ground / mid-height / upper semantics differ substantially in occupancy tasks, and height-specific weights capture this variation. The 246K parameter cost is negligible relative to the full model.]

### 5.3.4 Fusion Convolution: 3×3 vs. 1×1

The fusion convolution (fuse_conv) merges camera and LiDAR BEV features. We compare a spatial 3×3 Conv-BN-ReLU (default, BEVFusion-style) against a channel-mixing-only 1×1 Conv-BN-ReLU.

| Fusion Conv | Kernel | mIoU |
|---|---|---|
| Point-wise | 1×1 | [XX.X] |
| **Spatial (default)** | **3×3** | **[XX.X]** |

*Table 6. Ablation: fusion convolution kernel size.*

[FILL IN analysis. Expected narrative: 3×3 provides local spatial smoothing across modalities, beneficial because camera BEV and LiDAR BEV features may not align perfectly at the sub-voxel level due to calibration noise and depth estimation errors. The 1×1 alternative lacks this spatial context.]

---

## 5.4 Qualitative Results

Figure~\ref{fig:qual} presents side-by-side occupancy predictions for representative validation scenes, comparing the camera-only FlashOcc baseline, LiDAR-only PointOcc, and our proposed FLC-PointOcc (Version A).

> **[TODO: generate visualisations for 3–4 representative scenes showing:]**
> - (a) Surround-view RGB input (6-camera mosaic or front+rear pair)
> - (b) Ground-truth occupancy (bird's-eye view, coloured by semantic class)
> - (c) FlashOcc prediction
> - (d) PointOcc prediction
> - (e) FLC-PointOcc prediction
>
> **Recommend selecting scenes that highlight the complementary strengths:**
> - Scene 1: urban intersection with cyclists and pedestrians → camera advantage (small objects)
> - Scene 2: highway or parking lot with clear flat surfaces → LiDAR advantage (driveable surface geometry)
> - Scene 3: occlusion or camera saturation scenario → LiDAR robustness
> - Scene 4: LiDAR sparsity at range → camera filling in distant regions

**Observation 1: Small object recovery.** Camera features provide rich semantic cues for small objects such as cyclists, pedestrians, and traffic cones. As shown in Figure~\ref{fig:qual}(c) vs. (d), FlashOcc produces more complete predictions for these categories than PointOcc, where LiDAR returns are too sparse to form a dense prediction. FLC-PointOcc inherits the camera's semantic awareness while maintaining geometric accuracy. [FILL IN with specific observations from actual visualisations.]

**Observation 2: Large surface accuracy.** LiDAR features provide precise geometric boundaries for planar structures such as driveable surfaces, sidewalks, and terrain. PointOcc and FLC-PointOcc generate sharper and more geometrically accurate predictions for these classes compared to FlashOcc, where depth estimation errors in the LSS view transformer introduce boundary artefacts. [FILL IN.]

**Observation 3: Fusion coherence.** FLC-PointOcc predictions exhibit greater spatial coherence than either unimodal method: the geometric precision of LiDAR suppresses the blurring artefacts common in camera-only BEV predictions, while the semantic richness of camera features fills in the sparse regions of LiDAR-only predictions. This is particularly visible at medium-to-long ranges (>20 m from the ego vehicle). [FILL IN.]

---

## 5.5 Summary

Table~\ref{tab:summary} provides a consolidated comparison of all methods across both accuracy and efficiency dimensions. [FILL IN once all results are available.]

| Method | Modality | mIoU | FPS | GFLOPs | Train GPU-H | Advantage |
|---|---|---|---|---|---|---|
| Cam-OO | C | [XX.X] | [XX.X] | [XX.X] | [XX.X] | Pure camera, no LiDAR |
| LiDAR-OO | L | [XX.X] | [XX.X] | [XX.X] | [XX.X] | No camera hardware |
| MM-OO | C+L | [XX.X] | [XX.X] | [XX.X] | [XX.X] | OO baseline fusion |
| PointOcc | L | [XX.X] | [XX.X] | [XX.X] | [XX.X] | Strong geometry |
| FlashOcc | C | [XX.X] | [XX.X] | [XX.X] | [XX.X] | Fast camera-only |
| **FLC-PointOcc (A)** | **C+L** | **[XX.X]** | **[XX.X]** | **[XX.X]** | **[XX.X]** | **Efficient fusion** |

*Table 7. Summary comparison.*

The results demonstrate that FLC-PointOcc achieves the best mIoU among all compared methods while maintaining a computational profile substantially lighter than the OpenOccupancy multi-modal baseline. This validates the design thesis that fusing camera semantics and LiDAR geometry in BEV, rather than in full 3D voxel space, provides an efficient and effective pathway towards deployment-friendly surrounding occupancy perception. [FILL IN final statement after numbers are available.]

---

## Appendix: Measurement Protocols for Efficiency Metrics

To ensure reproducibility, we describe the exact procedures for measuring each efficiency metric.

**GFLOPs.** Computed using [TOBECONFIRMED: specify tool] with a single input sample: one 6-camera image tensor `[1, 6, 3, 256, 704]` and one point cloud of [TOBECONFIRMED: N] points. GFLOPs are computed as multiply-add operations (MACs × 2). [TOBECONFIRMED: clarify whether this includes the LiDAR preprocessing and grid_sample operations, which are often excluded from standard FLOP counters.]

**FPS.** Each method is warmed up for [TOBECONFIRMED: 50] inference passes, then FPS is measured as the reciprocal of the mean forward-pass latency averaged over [TOBECONFIRMED: 100] passes. Measurements are taken with `torch.cuda.synchronize()` before and after each pass to ensure GPU-CPU synchronisation. [TOBECONFIRMED: specify FP32 vs. FP16/AMP mode.]

**Inference memory.** `torch.cuda.reset_peak_memory_stats()` is called before each forward pass; `torch.cuda.max_memory_allocated()` is recorded after. Reported as peak delta in MiB.

**Training memory.** Measured at the first complete training step (forward + backward + optimiser step) with batch size 4. Activation checkpointing (`with_cp=True`) is active in both the camera ResNet-50 and LiDAR Swin-T, consistent with training conditions.

**Training duration.** Measured as total wall-clock training time in hours, reported in GPU-H (1 GPU × elapsed hours). Includes data loading time with `workers_per_gpu=2` but excludes validation.

**Parameter count.** Counted using `sum(p.numel() for p in model.parameters())` in millions (M). Pretrained backbone weights are included.

---

## Checklist of Experiments to Run

| # | Experiment | Config / Command | Status |
|---|---|---|---|
| E1 | FLC-PointOcc Version A training | `CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py` | [TODO] |
| E2 | FLC-PointOcc Version B training | `CAM-LiDAR_flc_pointocc_camfull640_128x128x10.py` | [TODO] |
| E3 | FlashOcc (FLC-step2) training | `CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py` | [TODO] |
| E4 | Camera-OO training (256×704) | `CAM-R50_img256x704_128x128x10_4070ti.py` | [TODO] |
| E5 | LiDAR-OO training | `LiDAR_128x128x10.py` | [TODO] |
| E6 | PointOcc training | `LiDAR_pointocc_128x128x10_server.py` | [TODO] |
| E7 | MM-OO training (256×704) | [TOBECONFIRMED: adapted config needed] | [TODO] |
| E8 | Ablation: Version A vs B | E1 + E2 | depends on E1/E2 |
| E9 | Ablation: Zero-LiDAR | Add `debug_zero_lidar=True` variant | [TODO] |
| E10 | Ablation: Z-flatten order | Requires code change + retrain | [TODO] |
| E11 | Ablation: fuse_conv 1×1 | Requires config change + retrain | [TODO] |
| E12 | Efficiency measurement | All trained models, measure FPS/GFLOPs/mem | after E1–E7 |
| E13 | Qualitative visualisation | All trained models, select representative scenes | after E1–E7 |
