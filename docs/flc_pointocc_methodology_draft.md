# FLC-PointOcc Methodology Draft

本文档是一份面向论文写作的 `FLC-PointOcc` 方法部分初稿，依据当前代码实现整理而成。它刻意写得比正式论文更详细，目的是先把设计动机、张量流动、训练目标和数学表达完整铺开，后续再做删减、压缩和英文润色。

对应实现主要来自：

- `projects/occ_plugin/occupancy/detectors/flc_pointocc_net.py`
- `projects/occ_plugin/occupancy/detectors/occnet.py`
- `projects/occ_plugin/occupancy/image2bev/ViewTransformerLSSFlash.py`
- `projects/occ_plugin/occupancy/image2bev/ViewTransformerLSSBEVDepth.py`
- `projects/occ_plugin/occupancy/detectors/lidar_prep_mixin.py`
- `projects/occ_plugin/occupancy/lidar_encoder/cylinder_encoder.py`
- `projects/occ_plugin/occupancy/lidar_encoder/tpv_swin.py`
- `projects/occ_plugin/occupancy/lidar_encoder/tpv_fpn.py`
- `projects/occ_plugin/occupancy/lidar_encoder/tpv_fuser.py`
- `projects/occ_plugin/occupancy/dense_heads/flc_occ_head.py`

---

## 1. Overview

FLC-PointOcc is a dual-modal occupancy prediction framework that fuses:

- a **camera branch** built on the FlashOcc-style 2D BEV pipeline, and
- a **LiDAR branch** built on the PointOcc-style tri-perspective-view (TPV) representation.

The core motivation is straightforward:

1. The camera branch offers dense semantic context over the full surround-view field, but its depth reasoning is indirect.
2. The LiDAR branch offers strong geometric evidence, but its point support is sparse and less semantically expressive than image features.
3. Instead of fusing the two modalities in full 3D voxel space, we fuse them in **BEV after camera Z-collapse and LiDAR TPV-to-voxel lifting**, which preserves most of the geometric benefit while keeping the downstream encoder purely 2D and computationally efficient.

At a high level, the pipeline is:

$$
\{\mathbf I_n\}_{n=1}^N
\xrightarrow{\text{LSS + Z-collapse}}
\mathbf B^{cam}
$$

$$
\mathcal P
\xrightarrow{\text{CylinderEncoder + TPVSwin + TPVFPN + TPVFuser}}
\mathbf F^{lidar}_{3D}
\xrightarrow{\text{flatten } Z}
\mathbf B^{lidar}
$$

$$
[\mathbf B^{cam} \| \mathbf B^{lidar}]
\xrightarrow{\text{BEV fusion}}
\mathbf B^{fuse}
\xrightarrow{\text{2D BEV encoder}}
\mathbf H
\xrightarrow{\text{C2H head}}
\hat{\mathbf Y}
$$

where \(\hat{\mathbf Y}\in \mathbb R^{C\times X\times Y\times Z}\) denotes the coarse semantic occupancy logits, \(C\) is the number of semantic classes, and \((X,Y,Z)\) is the coarse occupancy grid size.

In the current implementation:

- number of cameras: \(N=6\)
- image input size: \(256\times 704\)
- final occupancy label size: \(512\times 512\times 40\)
- coarse occupancy prediction size: \(128\times 128\times 10\)
- number of semantic classes: \(C=17\)

---

## 2. Problem Formulation and Notation

Let the surround-view image set be

$$
\mathcal I=\{\mathbf I_n\}_{n=1}^{N}, \qquad \mathbf I_n\in\mathbb R^{3\times H_{img}\times W_{img}},
$$

and let the accumulated LiDAR point cloud be

$$
\mathcal P=\{\mathbf p_i\}_{i=1}^{M}, \qquad
\mathbf p_i=[x_i,y_i,z_i,\iota_i,r_i],
$$

where \(\iota_i\) and \(r_i\) denote intensity and ring index, respectively.

The goal is to predict a semantic occupancy tensor

$$
\hat{\mathbf Y}\in \mathbb R^{C\times X\times Y\times Z},
$$

where each voxel stores class logits over \(C=17\) categories, including the empty class.

The full-resolution ground-truth occupancy label is

$$
\mathbf Y^{gt}\in \{0,\dots,C-1,255\}^{512\times 512\times 40},
$$

where 255 denotes ignore labels. Since the current model predicts on a coarse grid \((X,Y,Z)=(128,128,10)\), training uses an occupancy-aware downsampled target

$$
\bar{\mathbf Y}^{gt}=\mathcal D_{occ}(\mathbf Y^{gt}),
$$

where \(\mathcal D_{occ}\) aggregates each \(4\times 4\times 4\) local block into one coarse voxel, preferring occupied semantic labels over empty voxels whenever occupied labels are present.

---

## 3. Camera Branch: FlashOcc-Style 2D BEV Encoding

### 3.1 Multi-view image encoding

Each image \(\mathbf I_n\) is first passed through a 2D image backbone and neck:

$$
\mathbf F_n^{img}=\mathcal N_{img}(\mathcal B_{img}(\mathbf I_n)),
$$

where \(\mathcal B_{img}\) is a ResNet-50 backbone and \(\mathcal N_{img}\) is a SECONDFPN neck. In the current configuration, the fused per-camera feature has shape

$$
\mathbf F_n^{img}\in\mathbb R^{512\times H_f\times W_f},\qquad (H_f,W_f)=(16,44).
$$

Stacking all cameras yields

$$
\mathbf F^{img}\in \mathbb R^{B\times N\times 512\times 16\times 44}.
$$

### 3.2 Camera-aware depth and context prediction

The view transformer adopts the BEVDepth-style `DepthNet`, which jointly predicts:

- a depth distribution over discretized depth bins, and
- a context feature that will be lifted into 3D.

For each camera feature \(\mathbf F_n^{img}\), we first compute an intermediate feature

$$
\tilde{\mathbf F}_n = \mathrm{Conv}_{3\times 3}(\mathbf F_n^{img}).
$$

The model also constructs a camera metadata vector \(\mathbf m_n\), concatenating:

- intrinsic parameters,
- post-augmentation parameters,
- BEV augmentation parameters, and
- sensor-to-ego pose terms.

After normalization, \(\mathbf m_n\) is fed into two MLP-SE branches:

$$
\mathbf z_n^{ctx} = \mathrm{MLP}_{ctx}(\mathbf m_n), \qquad
\mathbf z_n^{dep} = \mathrm{MLP}_{dep}(\mathbf m_n).
$$

These vectors modulate the shared image feature through squeeze-excitation:

$$
\mathbf F_n^{ctx} = \mathrm{Conv}_{1\times 1}\big(\mathrm{SE}_{ctx}(\tilde{\mathbf F}_n,\mathbf z_n^{ctx})\big),
$$

$$
\mathbf D_n = \mathrm{DepthHead}\big(\mathrm{SE}_{dep}(\tilde{\mathbf F}_n,\mathbf z_n^{dep})\big),
$$

where \(\mathbf D_n\in\mathbb R^{D_d\times H_f\times W_f}\) are depth logits, and \(\mathbf F_n^{ctx}\in\mathbb R^{C_t\times H_f\times W_f}\) are context features with \(C_t=64\).

The discrete depth distribution is

$$
\mathbf P_n(d,u,v)=\mathrm{softmax}_d(\mathbf D_n(d,u,v)),
$$

where \(d\in\{1,\dots,D_d\}\) indexes depth bins. In the current config,

$$
D_d=\frac{58.0-2.0}{0.5}=112.
$$

### 3.3 Lift-Splat-Shoot view transformation

Given context feature \(\mathbf F_n^{ctx}\) and depth probability \(\mathbf P_n\), we construct a lifted frustum feature volume by an outer-product style factorization:

$$
\mathbf V_n(d,u,v,:) = \mathbf P_n(d,u,v)\cdot \mathbf F_n^{ctx}(u,v,:).
$$

This produces a camera frustum tensor

$$
\mathbf V_n\in\mathbb R^{D_d\times H_f\times W_f\times C_t}.
$$

For each depth bin \(d\) and image pixel \((u,v)\), the corresponding 3D point is obtained by inverse projection and camera-to-ego transformation. Abstractly,

$$
\mathbf x_{n,d,u,v}^{ego}
=
\mathcal T_{bda}
\Big(
\mathbf R_n \mathbf K_n^{-1}\,\tilde{\mathbf u}_{n,d,u,v}
+ \mathbf t_n
\Big),
$$

where:

- \(\tilde{\mathbf u}_{n,d,u,v}\) denotes the post-augmentation-corrected image ray multiplied by depth \(d\),
- \(\mathbf K_n\) is the camera intrinsic matrix,
- \((\mathbf R_n,\mathbf t_n)\) are camera-to-ego extrinsics,
- \(\mathcal T_{bda}\) is the BEV data augmentation transform.

The LSS splat stage accumulates frustum features into a Cartesian voxel grid:

$$
\mathbf F^{cam}_{3D}(x,y,z)
=
\sum_{n,d,u,v}
\mathbb 1\!\left[\Pi(\mathbf x_{n,d,u,v}^{ego})=(x,y,z)\right]
\mathbf V_n(d,u,v),
$$

where \(\Pi(\cdot)\) denotes discretization to the BEV voxel grid. The implementation uses voxel pooling / cumulative summation over points assigned to the same voxel.

### 3.4 Z-collapse to 2D BEV

Unlike OpenOccupancy’s original camera branch, FLC-PointOcc follows FlashOcc and collapses the height axis into channels:

$$
\mathbf B^{cam}(x,y)
=
\mathrm{Concat}_{z=1}^{Z_c}
\mathbf F^{cam}_{3D}(x,y,z),
$$

thus producing a 2D BEV feature map

$$
\mathbf B^{cam}\in\mathbb R^{(C_t Z_c)\times X\times Y}.
$$

For the current setup:

- \(C_t=64\)
- \(Z_c=10\)
- therefore \(\mathbf B^{cam}\in\mathbb R^{640\times 128\times 128}\)

This step is the key FlashOcc-style efficiency trick: it preserves height-conditioned information in channel space while allowing all subsequent BEV processing to remain purely 2D.

---

## 4. LiDAR Branch: PointOcc-Style TPV Geometry Encoding

### 4.1 Cylindrical coordinate conversion

Each raw LiDAR point

$$
\mathbf p_i=[x_i,y_i,z_i,\iota_i,r_i]
$$

is converted into cylindrical coordinates:

$$
\rho_i = \sqrt{x_i^2+y_i^2}, \qquad
\phi_i = \mathrm{atan2}(y_i,x_i).
$$

Let the cylindrical range be

$$
[\rho,\phi,z]\in
[0,50]\times[-\pi,\pi]\times[-5,3],
$$

with grid size

$$
(G_\rho,G_\phi,G_z)=(480,360,32).
$$

The cylindrical voxel index is

$$
\mathbf g_i=
\left\lfloor
\frac{[\rho_i,\phi_i,z_i]-\mathbf b_{min}}{\Delta \mathbf b}
\right\rfloor,
$$

where \(\mathbf b_{min}\) is the lower bound and \(\Delta \mathbf b\) is the per-axis cylindrical voxel size.

### 4.2 Point feature construction

For each point, the model builds a 10-dimensional feature vector:

$$
\mathbf f_i =
\Big[
\rho_i-\bar\rho_i,\;
\phi_i-\bar\phi_i,\;
z_i-\bar z_i,\;
\rho_i,\phi_i,z_i,\;
x_i,y_i,\;
\iota_i,\;
r_i
\Big],
$$

where \((\bar\rho_i,\bar\phi_i,\bar z_i)\) is the center of the cylindrical voxel indexed by \(\mathbf g_i\).

This design combines:

- local offset information inside the voxel,
- absolute cylindrical geometry,
- absolute Cartesian planar position, and
- LiDAR intensity / ring attributes.

### 4.3 Sparse cylindrical voxelization and scatter-max encoding

Each point feature is passed through a point-wise MLP:

$$
\mathbf h_i = \mathrm{MLP}_{pt}(\mathbf f_i).
$$

Then points falling into the same cylindrical voxel are aggregated by max pooling:

$$
\mathbf V(\rho,\phi,z)
=
\max_{i:\mathbf g_i=(\rho,\phi,z)} \mathbf h_i.
$$

The resulting sparse cylindrical tensor is stored as a `SparseConvTensor` and serves as the input to the TPV construction stage.

### 4.4 Tri-Perspective View (TPV) decomposition

PointOcc avoids dense 3D convolutions by decomposing the cylindrical volume into three orthogonal 2D planes:

- \(xy\) plane (here \(\rho\)-\(\phi\) plane),
- \(yz\) plane (here \(\phi\)-\(z\) plane),
- \(zx\) plane (here \(z\)-\(\rho\) plane).

The current implementation uses sparse max pooling along one axis followed by a small per-plane MLP:

$$
\mathbf T_{xy} = \mathrm{MLP}_{xy}\big(\mathrm{Pool}_{z}(\mathbf V)\big),
$$

$$
\mathbf T_{yz} = \mathrm{MLP}_{yz}\big(\mathrm{Pool}_{\rho}(\mathbf V)\big),
$$

$$
\mathbf T_{zx} = \mathrm{MLP}_{zx}\big(\mathrm{Pool}_{\phi}(\mathbf V)\big).
$$

The split factor is \([8,8,8]\), meaning the pooled axis is chunked before being flattened and projected. Intuitively, this compresses each 3D line of voxels into a 2D plane feature while retaining coarse structure along the pooled axis.

The resulting three TPV planes are:

$$
\mathbf T_{xy}\in\mathbb R^{128\times 480\times 360},
\quad
\mathbf T_{yz}\in\mathbb R^{128\times 360\times 32},
\quad
\mathbf T_{zx}\in\mathbb R^{128\times 32\times 480}.
$$

### 4.5 Shared TPV backbone and TPV FPN

The three planes are processed by a shared Swin Transformer backbone:

$$
\{\mathbf T_{xy}^{(s)}\}_{s=1}^S = \mathcal B_{tpv}(\mathbf T_{xy}),
\quad
\{\mathbf T_{yz}^{(s)}\}_{s=1}^S = \mathcal B_{tpv}(\mathbf T_{yz}),
\quad
\{\mathbf T_{zx}^{(s)}\}_{s=1}^S = \mathcal B_{tpv}(\mathbf T_{zx}),
$$

where the same TPV Swin parameters are shared across all three planes.

Each plane’s multi-scale features are then fused by a dedicated top-down FPN:

$$
\tilde{\mathbf T}_{xy} = \mathcal N_{tpv}\big(\{\mathbf T_{xy}^{(s)}\}\big),
\quad
\tilde{\mathbf T}_{yz} = \mathcal N_{tpv}\big(\{\mathbf T_{yz}^{(s)}\}\big),
\quad
\tilde{\mathbf T}_{zx} = \mathcal N_{tpv}\big(\{\mathbf T_{zx}^{(s)}\}\big).
$$

In the current configuration, each TPV plane is projected to 192 channels:

$$
\tilde{\mathbf T}_{xy},\tilde{\mathbf T}_{yz},\tilde{\mathbf T}_{zx}
\in \mathbb R^{192\times h\times w}.
$$

### 4.6 TPV-to-voxel lifting by query sampling

To obtain dense 3D LiDAR features, FLC-PointOcc reuses the PointOcc-style query-based TPV fusion idea, but stops before classification.

Let \(\mathbf q_j=[\rho_j,\phi_j,z_j]\) denote the cylindrical coordinate of a coarse occupancy voxel center. The current code constructs these queries from the Cartesian coarse grid and maps them into cylindrical coordinates.

They are normalized to the range \([-1,1]\):

$$
\bar{\mathbf q}_j=
\left[
2\rho_j/W_\rho -1,\;
2\phi_j/H_\phi -1,\;
2z_j/Z_z -1
\right].
$$

Then the three TPV planes are sampled at the corresponding projected coordinates:

$$
\mathbf f_j^{xy}=\mathcal G(\tilde{\mathbf T}_{xy},[\bar\rho_j,\bar\phi_j]),
$$

$$
\mathbf f_j^{yz}=\mathcal G(\tilde{\mathbf T}_{yz},[\bar\phi_j,\bar z_j]),
$$

$$
\mathbf f_j^{zx}=\mathcal G(\tilde{\mathbf T}_{zx},[\bar z_j,\bar\rho_j]),
$$

where \(\mathcal G(\cdot)\) is bilinear `grid_sample`.

The three views are fused by element-wise summation:

$$
\mathbf f_j^{lidar}=
\mathbf f_j^{xy}+\mathbf f_j^{yz}+\mathbf f_j^{zx}.
$$

Reshaping all queried features yields a dense coarse 3D tensor:

$$
\mathbf F^{lidar}_{3D}\in\mathbb R^{C_{tpv}\times X\times Y\times Z},
\qquad C_{tpv}=192.
$$

For the current implementation:

$$
\mathbf F^{lidar}_{3D}\in\mathbb R^{192\times 128\times 128\times 10}.
$$

---

## 5. Height-Aware LiDAR BEV Projection

A key design choice in FLC-PointOcc is how to transform the LiDAR 3D tensor into a 2D BEV representation for fusion.

Instead of first projecting channels and then flattening height, we **flatten height first** and only then apply a \(1\times 1\) projection:

$$
\mathbf B^{lidar}(x,y)
=
\mathrm{Concat}_{z=1}^{Z_c}
\mathbf F^{lidar}_{3D}(x,y,z),
$$

so that

$$
\mathbf B^{lidar}\in\mathbb R^{(C_{tpv}Z_c)\times X\times Y}
=
\mathbb R^{1920\times 128\times 128}.
$$

Then a \(1\times 1\) convolution reduces channel dimensionality:

$$
\tilde{\mathbf B}^{lidar}
=
\phi_{\ell}\big(\mathbf B^{lidar}\big),
\qquad
\phi_{\ell}:\mathbb R^{1920}\rightarrow\mathbb R^{128}.
$$

Equivalently, for each BEV location \((x,y)\),

$$
\tilde{\mathbf B}^{lidar}(x,y)
=
\mathbf W_{\ell}
\big[
\mathbf F^{lidar}_{3D}(x,y,1)\|
\cdots\|
\mathbf F^{lidar}_{3D}(x,y,Z_c)
\big]
+\mathbf b_{\ell}.
$$

This is important because the projection matrix \(\mathbf W_{\ell}\in\mathbb R^{128\times (192\cdot Z_c)}\) can assign different weights to different height bins. In contrast, a shared per-voxel linear projection before flattening would force all heights to share the same channel projection, which is less expressive for semantic occupancy.

---

## 6. Cross-Modal Fusion in BEV

### 6.1 Camera feature adaptation

Let \(\mathbf B^{cam}\in\mathbb R^{640\times 128\times 128}\) be the FlashOcc-style camera BEV feature.

FLC-PointOcc provides two fusion variants:

**Version A (camadapt256).**

$$
\tilde{\mathbf B}^{cam}=\phi_c(\mathbf B^{cam}),
\qquad
\phi_c:\mathbb R^{640}\rightarrow\mathbb R^{256}.
$$

**Version B (camfull640).**

$$
\tilde{\mathbf B}^{cam}=\mathbf B^{cam}.
$$

The LiDAR branch is identical in both versions:

$$
\tilde{\mathbf B}^{lidar}\in\mathbb R^{128\times 128\times 128}.
$$

### 6.2 BEVFusion-style concatenation and smoothing

The two modalities are fused by channel concatenation followed by a \(3\times 3\) convolution:

$$
\mathbf B^{cat}
=
\tilde{\mathbf B}^{cam}\|
\tilde{\mathbf B}^{lidar},
$$

$$
\mathbf B^{fuse}
=
\phi_f(\mathbf B^{cat}),
$$

where \(\phi_f\) is a \(3\times 3\) Conv-BN-ReLU block.

The channel dimensions are:

- Version A: \(256 + 128 = 384 \rightarrow 256\)
- Version B: \(640 + 128 = 768 \rightarrow 256\)

This design intentionally follows a BEVFusion-style “front-light/back-heavy” principle:

- the modality-specific adapters are shallow and cheap,
- the main semantic reasoning is deferred to the post-fusion BEV encoder.

Using a \(3\times 3\) fusion conv instead of a pure \(1\times 1\) projection allows local spatial smoothing across modalities, which is especially useful because camera semantics and LiDAR geometry may not align perfectly at individual cells.

---

## 7. 2D BEV Decoder and Channel-to-Height Occupancy Head

### 7.1 Post-fusion BEV encoder

The fused BEV feature is processed by a 2D BEV encoder:

$$
\{\mathbf F_1,\mathbf F_2,\mathbf F_3\}
=
\mathcal B_{bev}(\mathbf B^{fuse}),
$$

where \(\mathcal B_{bev}\) is a `CustomResNet2D`.

These multi-scale features are fused by `FPN_LSS`:

$$
\mathbf H
=
\mathcal N_{bev}(\mathbf F_1,\mathbf F_2,\mathbf F_3).
$$

In the current setup:

$$
\mathbf H\in\mathbb R^{256\times 128\times 128}.
$$

### 7.2 Channel-to-Height (C2H) occupancy prediction

FLC-PointOcc adopts a FlashOcc-style coarse occupancy head (`FLCOccHead`) instead of a 3D convolutional occupancy head.

First, a 2D convolution aggregates local BEV context:

$$
\mathbf U = \psi_{2D}(\mathbf H),
\qquad
\mathbf U\in\mathbb R^{256\times 128\times 128}.
$$

Then, for each BEV pillar \((x,y)\), an MLP lifts the 2D feature vector into 3D semantic logits:

$$
\mathbf o_{x,y}
=
\mathrm{MLP}_{c2h}\big(\mathbf U(:,x,y)\big),
\qquad
\mathbf o_{x,y}\in\mathbb R^{C\cdot Z_c}.
$$

Finally, \(\mathbf o_{x,y}\) is reshaped to voxel logits:

$$
\hat{\mathbf Y}(:,x,y,:)
=
\mathrm{reshape}\big(\mathbf o_{x,y}, C, Z_c\big).
$$

Aggregating all pillars yields:

$$
\hat{\mathbf Y}\in\mathbb R^{17\times 128\times 128\times 10}.
$$

This head preserves the FlashOcc philosophy: height information is recovered through a per-pillar channel-to-height mapping instead of explicit 3D convolutions.

### 7.3 About the fine branch

The inherited OpenOccupancy head interface still contains a coarse-to-fine refinement branch. However, in the current FLC-PointOcc instantiation:

- `sample_from_voxel=False`
- `sample_from_img=False`

therefore the active model is a **coarse-only occupancy predictor**. The fine query refinement path is present in the codebase for future extensions, but it is not used by the current A/B experiments.

---

## 8. Training Objective

The final training objective combines:

1. a depth supervision loss for the camera branch, and
2. multiple occupancy losses on the coarse voxel logits.

### 8.1 Depth supervision

Ground-truth depth is discretized into one-hot depth bins:

$$
\mathbf p_u^{*}\in\{0,1\}^{D_d}.
$$

The predicted depth probability is \(\hat{\mathbf p}_u\in[0,1]^{D_d}\). The depth loss is binary cross-entropy over foreground depth pixels:

$$
\mathcal L_{depth}
=
\lambda_d
\frac{1}{|\Omega|}
\sum_{u\in\Omega}
\mathrm{BCE}(\hat{\mathbf p}_u,\mathbf p_u^{*}),
$$

where \(\Omega\) denotes valid depth locations and \(\lambda_d=3\) in the current configuration.

### 8.2 Weighted coarse occupancy cross-entropy

Let \(\bar{\mathbf Y}^{gt}\) be the occupancy-aware coarse label and \(\hat{\mathbf Y}\) the coarse logits. The weighted voxel-wise cross-entropy is:

$$
\mathcal L_{ce}
=
\frac{1}{|\mathcal V|}
\sum_{v\in\mathcal V}
w_{y_v}\,
\mathrm{CE}(\hat{\mathbf Y}_v,\bar{\mathbf Y}^{gt}_v),
$$

where \(w_c\) is the class weight for class \(c\), computed from class frequency \(f_c\) as

$$
w_c = \frac{1}{\log(f_c + 10^{-3})}.
$$

### 8.3 Semantic scaling loss

Following OpenOccupancy’s occupancy objective, a class-wise semantic scaling loss is used to encourage good precision, recall, and specificity for every semantic class.

For class \(c\), let

$$
p_c(v)=\mathrm{softmax}(\hat{\mathbf Y}_v)_c,
\qquad
t_c(v)=\mathbb 1[\bar{\mathbf Y}^{gt}_v=c].
$$

Then class-wise precision, recall, and specificity are computed as:

$$
\mathrm{Prec}_c=
\frac{\sum_v p_c(v)t_c(v)}{\sum_v p_c(v)},
\qquad
\mathrm{Rec}_c=
\frac{\sum_v p_c(v)t_c(v)}{\sum_v t_c(v)},
$$

$$
\mathrm{Spec}_c=
\frac{\sum_v (1-p_c(v))(1-t_c(v))}{\sum_v (1-t_c(v))}.
$$

The semantic scaling loss averages BCE-to-one over valid classes:

$$
\mathcal L_{sem}
=
\frac{1}{|\mathcal C^{+}|}
\sum_{c\in\mathcal C^{+}}
\Big(
\mathrm{BCE}(\mathrm{Prec}_c,1)
+
\mathrm{BCE}(\mathrm{Rec}_c,1)
+
\mathrm{BCE}(\mathrm{Spec}_c,1)
\Big),
$$

where \(\mathcal C^{+}\) denotes classes present in the valid target voxels.

### 8.4 Geometric scaling loss

The geometric scaling loss reduces occupancy to a binary non-empty vs. empty problem. Let

$$
p_{emp}(v)=\mathrm{softmax}(\hat{\mathbf Y}_v)_{empty},
\qquad
p_{occ}(v)=1-p_{emp}(v),
$$

and

$$
t_{occ}(v)=\mathbb 1[\bar{\mathbf Y}^{gt}_v \neq empty].
$$

Binary precision, recall, and specificity are then:

$$
\mathrm{Prec}_{occ}=
\frac{\sum_v p_{occ}(v)t_{occ}(v)}{\sum_v p_{occ}(v)},
$$

$$
\mathrm{Rec}_{occ}=
\frac{\sum_v p_{occ}(v)t_{occ}(v)}{\sum_v t_{occ}(v)},
$$

$$
\mathrm{Spec}_{occ}=
\frac{\sum_v (1-t_{occ}(v))p_{emp}(v)}{\sum_v (1-t_{occ}(v))}.
$$

The geometric scaling loss is:

$$
\mathcal L_{geo}
=
\mathrm{BCE}(\mathrm{Prec}_{occ},1)
+
\mathrm{BCE}(\mathrm{Rec}_{occ},1)
+
\mathrm{BCE}(\mathrm{Spec}_{occ},1).
$$

### 8.5 Lovasz-Softmax loss

To better align optimization with IoU-style occupancy metrics, we also use Lovasz-Softmax:

$$
\mathcal L_{lovasz}
=
\mathrm{LovaszSoftmax}
\big(
\mathrm{softmax}(\hat{\mathbf Y}),
\bar{\mathbf Y}^{gt}
\big).
$$

### 8.6 Total loss

The complete training objective is:

$$
\mathcal L
=
\mathcal L_{depth}
+
\lambda_{ce}\mathcal L_{ce}
+
\lambda_{sem}\mathcal L_{sem}
+
\lambda_{geo}\mathcal L_{geo}
+
\lambda_{lov}\mathcal L_{lovasz},
$$

with

$$
\lambda_{ce}=\lambda_{sem}=\lambda_{geo}=\lambda_{lov}=1.
$$

Since the fine branch is disabled in the current model, no fine occupancy refinement loss is used.

---

## 9. Current Variants and Their Only Difference

The current paper draft should explicitly state that the main A/B ablation changes only the camera-side BEV compression:

### Version A: `camadapt256`

$$
\mathbf B^{cam}\in\mathbb R^{640\times 128\times 128}
\xrightarrow{\phi_c}
\tilde{\mathbf B}^{cam}\in\mathbb R^{256\times 128\times 128}.
$$

Then:

$$
[256 \| 128] \rightarrow 384 \xrightarrow{3\times 3} 256.
$$

### Version B: `camfull640`

$$
\tilde{\mathbf B}^{cam}=\mathbf B^{cam}\in\mathbb R^{640\times 128\times 128},
$$

then:

$$
[640 \| 128] \rightarrow 768 \xrightarrow{3\times 3} 256.
$$

Therefore, A and B differ only in whether the camera BEV is compressed before fusion. The LiDAR branch, fusion output width, BEV decoder, occupancy head, and loss functions remain identical.

---

## 10. Method Summary in One Paragraph

In summary, FLC-PointOcc combines a FlashOcc-style camera BEV encoder with a PointOcc-style LiDAR TPV geometry encoder. The camera branch predicts a depth-aware BEV representation through LSS and collapses the height axis into channels, while the LiDAR branch transforms raw points into cylindrical features, aggregates them into TPV planes, lifts them back to a dense coarse 3D tensor by query-based TPV sampling, and then flattens height into a BEV representation with height-specific projection weights. The two modalities are fused in BEV through lightweight modality adapters and a BEVFusion-style \(3\times 3\) smoothing convolution, after which a pure 2D BEV encoder and a Channel-to-Height occupancy head generate coarse semantic occupancy logits. This design preserves the geometric strength of LiDAR and the semantic density of multi-view images, while avoiding costly 3D convolutions in the shared fusion-decoding stage.
