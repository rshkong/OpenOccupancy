# Model Structure Guideline

本文档对照说明 3 条模型链路：

1. OpenOccupancy camera baseline  
   配置：[projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py](/home/shkong/MyProject/OpenOccupancy/projects/configs/baselines/CAM-R50_img256x704_128x128x10_4070ti.py)
2. 当前 FLC-Step2  
   配置：[projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py](/home/shkong/MyProject/OpenOccupancy/projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py)
3. PointOcc LiDAR occupancy  
   配置：[/home/shkong/MyProject/PointOcc/config/pointtpv_nusc_occ.py](/home/shkong/MyProject/PointOcc/config/pointtpv_nusc_occ.py)

本文档先给出三个模型的总体流程图和张量流动，再逐阶段解释每一步做了什么。

---

## 1. 三个模型的总体流程图

### 1.1 OpenOccupancy camera baseline

```text
输入图像
[B, 6, 3, 256, 704]

-> image_encoder
[B, 6, C_img, 16, 44]

-> ViewTransformerLiftSplatShootVoxel
  - DepthNet
  - depth_prob
  - Lift
  - get_geometry
  - voxel_pooling

-> 3D voxel feature
[B, 80, 128, 128, 10]

-> occ_encoder_backbone (CustomResNet3D)
-> multi-level 3D features

-> occ_encoder_neck (FPN3D)
-> voxel_feats list
  典型每层形状接近:
  [B, 256, X_i, Y_i, Z_i]

-> pts_bbox_head = OccHead
  - forward_coarse_voxel()
  - occ_convs (3D conv)
  - voxel_soft_weights
  - occ_pred_conv (3D conv)

-> out_voxel_feats
[B, 128, X, Y, Z]  # 中间融合后的 3D feature volume

-> output_voxels
[B, 17, X, Y, Z]   # coarse occupancy logits
```

---

### 1.2 FLC-Step2

```text
输入图像
[B, 6, 3, 256, 704]

-> image_encoder
[B, 6, C_img, 16, 44]

-> ViewTransformerLSSFlash
  - DepthNet
  - depth_prob
  - Lift
  - get_geometry
  - voxel_pooling with Z-collapse

-> 2D BEV feature
[B, 640, 128, 128]

-> occ_encoder_backbone (CustomResNet2D)
-> multi-scale 2D features
  典型:
  [B, 128, 64, 64]
  [B, 256, 32, 32]
  [B, 512, 16, 16]

-> occ_encoder_neck (FPN_LSS)

-> final BEV feature
[B, 256, 128, 128]

-> pts_bbox_head = FLCOccHead
  - final_conv (2D conv)
  - c2h_predicter (MLP)

-> output_voxels
[B, 17, 128, 128, 10]   # coarse occupancy logits

-> out_voxel_feats = None
```

---

### 1.3 PointOcc

```text
输入点云
points: [B, N_pts, 10]          # return_feat
grid_ind: list / [N_pts, 3]     # 柱坐标网格索引
grid_ind_vox_coarse:
  [B, (Wc*Hc*Dc), 3]            # coarse voxel query in cylindrical grid

-> CylinderEncoder_Occ
  - point_mlp
  - scatter_max
  - SparseConvTensor
  - SparseMaxPool3d on 3 axes

-> TPV 3-view features
  tpv_xy: [B, C_tpv, W_tpv, H_tpv]
  tpv_yz: [B, C_tpv, H_tpv, Z_tpv]
  tpv_zx: [B, C_tpv, Z_tpv, W_tpv]

-> lidar_backbone (Swin)
-> lidar_neck (GeneralizedLSSFPN)

-> processed TPV feature list
[tpv_xy, tpv_yz, tpv_zx]

-> TPVAggregator_Occ
  - interpolate TPV planes
  - normalize coarse voxel queries
  - project queries to 3 TPV planes
  - grid_sample from each plane
  - fuse 3 sampled features
  - decoder (MLP)
  - classifier

-> coarse logits
[B, 17, 256, 256, 20]   # for coarse_ratio=2 in original config

-> if eval:
   interpolate to final occupancy size
[B, 17, 512, 512, 40]
```

---

## 2. 三个模型共同的“总体思想差异”

### 2.1 baseline

思路是：

- 相机图像先通过 LSS 变成 3D voxel feature
- 后续持续在显式 3D 体素空间里计算
- head 也使用 3D conv

关键词：

- `dense 3D feature volume`
- `3D backbone + 3D neck + 3D head`

### 2.2 FLC-Step2

思路是：

- 相机图像仍然先通过 LSS 建立空间信息
- 但在 `view_transform` 末端就把 `Z` 折叠进 channel
- 后续在 2D BEV 上算完，再用 `C2H` 恢复 coarse 3D logits

关键词：

- `Z-collapse`
- `2D BEV encoder`
- `C2H (channel-to-height)`

### 2.3 PointOcc

思路是：

- LiDAR 不走 dense voxel 3D conv
- 先在柱坐标系下构建三张 TPV plane
- 用 2D backbone 处理 TPV plane
- 再对 coarse voxel query 做 `grid_sample`
- 用 MLP/classifier 输出 occupancy

关键词：

- `TPV (tri-perspective view)`
- `2D planes + query sampling`
- `LiDAR query-based occupancy`

---

## 3. 逐阶段解释

### 3.1 阶段一：输入与预处理

#### 3.1.1 baseline / FLC-Step2

入口都是 [OccNet](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/detectors/occnet.py)：

- [OccNet.extract_img_feat()](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/detectors/occnet.py:61)
- [OccNet.extract_feat()](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/detectors/occnet.py:107)

图像输入：

```text
[B, 6, 3, 256, 704]
```

`image_encoder()` 处理后：

```text
[B, 6, C_img, 16, 44]
```

同时保存：

```python
img_feats = [x.clone()]
```

这份 `img_feats` 是后续 image-only CONet fine branch 的采样来源。

#### 3.1.2 PointOcc

PointOcc 的 occupancy 数据包装在：

- [/home/shkong/MyProject/PointOcc/dataloader/dataset_wrapper.py](/home/shkong/MyProject/PointOcc/dataloader/dataset_wrapper.py)

关键输入包括：

- `return_feat`
  点级输入特征，送入 `CylinderEncoder_Occ`
- `grid_ind`
  点在柱坐标网格中的索引
- `voxel_position_grid_coarse`
  coarse occupancy query 的柱坐标位置

对应 `train_occ.py / eval_occ.py` 里的：

```python
(voxel_position_coarse, points, voxel_label, grid_ind)
```

---

### 3.2 阶段二：前端表征构建

#### 3.2.1 baseline / FLC-Step2：image_encoder

函数：

- [OccNet.image_encoder()](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/detectors/occnet.py:33)

流程：

```text
[B, 6, 3, 256, 704]
-> reshape
[B*6, 3, 256, 704]
-> img_backbone (ResNet50)
-> img_neck (SECONDFPN)
-> reshape back
[B, 6, C_img, 16, 44]
```

#### 3.2.2 PointOcc：CylinderEncoder_Occ

函数：

- [/home/shkong/MyProject/PointOcc/model/cylinder_encoder.py](/home/shkong/MyProject/PointOcc/model/cylinder_encoder.py)

处理步骤：

1. `point_mlp`
   对每个点做 point-wise MLP
2. `scatter_max`
   对落在同一柱坐标网格的点做聚合
3. 构造 `SparseConvTensor`
4. 沿三个方向做 `SparseMaxPool3d`

最后得到三张 TPV plane：

```text
tpv_xy
tpv_yz
tpv_zx
```

这一步可以理解成：

**PointOcc 不直接构建 dense 3D voxel volume，而是先把 LiDAR 压缩成三个 2D plane 表示。**

---

### 3.3 阶段三：空间建模

#### 3.3.1 baseline / FLC-Step2：view_transform

共同核心类：

- [ViewTransformerLiftSplatShootVoxel](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/image2bev/ViewTransformerLSSVoxel.py:16)
- [ViewTransformerLSSFlash](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/image2bev/ViewTransformerLSSFlash.py:12)
- [DepthNet](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/image2bev/ViewTransformerLSSBEVDepth.py:441)

共同流程：

```text
[B, 6, C_img, 16, 44]
-> get_mlp_input
[B, 6, 27]
-> DepthNet
-> depth_digit [B*6, D, 16, 44]
-> img_feat    [B*6, C_trans, 16, 44]
-> depth_prob = softmax(depth_digit)
-> Lift
-> [B, 6, D, 16, 44, C_trans]
-> get_geometry
-> voxel_pooling
```

当前配置里：

- `dbound = [2.0, 58.0, 0.5]`
- `D = 112`

##### baseline 的输出

```text
[B, 80, 128, 128, 10]
```

##### FLC-Step2 的输出

`ViewTransformerLSSFlash` 在 pooling 末端做：

```python
final = torch.cat(final.unbind(dim=2), 1)
```

所以输出从：

```text
[B, C, X, Y, Z]
```

变成：

```text
[B, C*Z, X, Y]
```

在当前配置下：

```text
[B, 640, 128, 128]
```

#### 3.3.2 PointOcc：TPV backbone + neck

在 [PointTPV_Occ.extract_lidar_feat()](</home/shkong/MyProject/PointOcc/model/pointtpv_occ.py>) 里：

```text
TPV 3-view
-> lidar_backbone (Swin)
-> lidar_neck (GeneralizedLSSFPN)
-> processed TPV feature list
```

也就是说 PointOcc 的“空间建模”不是 3D conv，而是：

**在 3 张 TPV plane 上做 2D backbone/neck。**

---

### 3.4 阶段四：occupancy 头

#### 3.4.1 baseline：OccHead

代码：

- [OccHead.forward_coarse_voxel()](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/dense_heads/occ_head.py:129)

流程：

```text
voxel_feats list
-> occ_convs (3D conv for each level)
-> output_occs
-> voxel_soft_weights
-> weighted fusion
-> out_voxel_feats [B, C, X, Y, Z]
-> occ_pred_conv
-> output_voxels [B, 17, X, Y, Z]
```

这里最重要的是：

**baseline 除了输出 coarse logits，还保留了一个真正的 3D 中间 feature volume：**

```text
out_voxel_feats: [B, C, X, Y, Z]
```

这也是原版 CONet `sample_from_voxel=True` 的依赖。

#### 3.4.2 FLC-Step2：FLCOccHead

代码：

- [FLCOccHead.forward_coarse_voxel()](/home/shkong/MyProject/OpenOccupancy/projects/occ_plugin/occupancy/dense_heads/flc_occ_head.py:79)

流程：

```text
[B, 256, 128, 128]
-> final_conv (2D conv)
[B, 256, 128, 128]
-> flatten each (x, y) pillar
[B*X*Y, 256]
-> c2h_predicter
[B*X*Y, Dz * num_cls]
-> reshape
[B, 17, 128, 128, 10]
```

关键区别：

- baseline 恢复的是：`3D feature + 3D logits`
- FLC-Step2 目前只恢复的是：`3D logits`

当前代码里直接写了：

```python
'out_voxel_feats': [None]
```

所以 FLC-Step2 现在没有可供 voxel refinement 使用的 dense 3D feature volume。

#### 3.4.3 PointOcc：TPVAggregator_Occ

代码：

- [/home/shkong/MyProject/PointOcc/model/tpv_aggregator.py](/home/shkong/MyProject/PointOcc/model/tpv_aggregator.py)

核心逻辑：

1. 对三张 TPV plane 做尺度对齐 `F.interpolate`
2. 对 coarse voxel query `voxels_coarse` 做归一化
3. 将 query 分别投到：
   - `tpv_hw`
   - `tpv_zh`
   - `tpv_wz`
4. 从三张 plane 上分别 `grid_sample`
5. 三路相加融合：

```python
fused = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox
```

6. 再过：
   - `decoder` (MLP)
   - `classifier`

7. reshape 成 coarse occupancy logits：

```text
[B, 17, W, H, D]
```

在原始 PointOcc 配置里：

- `grid_size_occ = [512, 512, 40]`
- `coarse_ratio = 2`

所以 coarse logits 是：

```text
[B, 17, 256, 256, 20]
```

eval 时再插值回：

```text
[B, 17, 512, 512, 40]
```

PointOcc 的关键点在于：

**它也没有持续保留 dense 3D conv backbone，而是直接在 coarse voxel queries 上从 TPV planes 采样。**

---

## 4. `out_voxel_feats: [B, C, X, Y, Z]` 在三个模型里的地位

### baseline

有，而且是原生的中间 3D feature volume：

```text
out_voxel_feats: [B, C, X, Y, Z]
```

来源：

- `OccHead.forward_coarse_voxel()`

用途：

- 可作为 voxel refinement 的采样源

### FLC-Step2

目前没有。

原因：

- `view_transform` 末端已经把 `Z` 折叠进了 channel
- `FLCOccHead` 只把 2D BEV 特征恢复成了 3D **分类 logits**
- 没有额外恢复 3D **中间 feature volume**

所以：

```text
output_voxels:   [B, 17, X, Y, Z]
out_voxel_feats: None
```

### PointOcc

严格说也不是 baseline 那种现成的 dense 3D middle feature。  
它更接近：

- 有 3 张 TPV plane feature
- 有 coarse voxel query
- query 时直接从 TPV planes 采样

所以它是：

**query-based coarse feature extraction**

而不是：

**dense 3D voxel feature volume**

---

## 5. 最简总结

### baseline

```text
Image
-> LSS
-> 3D voxel feature
-> 3D encoder
-> 3D head
-> coarse logits + dense 3D feature volume
```

### FLC-Step2

```text
Image
-> LSSFlash
-> 2D BEV feature
-> 2D encoder
-> C2H head
-> coarse logits only
```

### PointOcc

```text
LiDAR
-> cylindrical TPV
-> 2D TPV backbone/neck
-> coarse voxel queries
-> sample from TPV planes
-> coarse logits
```

### 最重要的一句

**baseline 的 coarse head 天然保留了可供后续 3D 插值采样的 `out_voxel_feats`；FLC-Step2 当前没有；PointOcc 则走的是“TPV plane + voxel query sampling”路线。**

---

## 6. 分辨率与 coarse ratio 说明

这几个数字最容易混：

- `256 x 704`
- `128 x 128 x 10`
- `256 x 256 x 20`
- `512 x 512 x 40`
- `coarse_ratio`

它们分别属于不同空间。

### 6.1 `256 x 704` 是图像输入分辨率

对应 camera baseline 和 FLC-Step2 的 `data_config['input_size']`：

```text
image size = [H_img, W_img] = [256, 704]
```

这是 2D 图像平面分辨率，不是 occupancy 网格分辨率。

经过 `image_encoder` 之后，通常会被 backbone/neck 下采样到：

```text
[16, 44]
```

因为当前 `downsample = 16`。

所以：

```text
256 x 704  ->  16 x 44
```

是 **图像特征图** 的变化。

---

### 6.2 `512 x 512 x 40` 是最终 occupancy 标注网格大小

在 OpenOccupancy 和 PointOcc 里，最终的 GT occupancy 都是：

```text
occ_size / grid_size_occ = [512, 512, 40]
```

对应物理范围：

```text
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
```

所以最终每个体素大小大致是：

```text
dx = 102.4 / 512 = 0.2 m
dy = 102.4 / 512 = 0.2 m
dz = 8.0   / 40  = 0.2 m
```

也就是最终标签分辨率是：

```text
0.2m x 0.2m x 0.2m
```

---

### 6.3 `128 x 128 x 10` 是 OpenOccupancy / FLC 的 coarse occupancy grid

OpenOccupancy camera baseline 和 FLC-Step2 都不是直接在 `512 x 512 x 40` 上做 coarse 预测，  
而是先在更粗的网格上工作：

```text
[128, 128, 10]
```

这个数字来自：

```text
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]
```

所以：

```text
512 / 4 = 128
512 / 4 = 128
40  / 4 = 10
```

也就是：

```text
coarse grid = final grid / lss_downsample
```

对应更粗的物理体素大小：

```text
0.8m x 0.8m x 0.8m
```

这就是为什么 camera baseline 和 FLC-Step2 的 coarse output 是：

```text
[B, 17, 128, 128, 10]
```

---

### 6.4 `256 x 256 x 20` 是 PointOcc 原始配置里的 coarse occupancy grid

PointOcc 原始配置：

```python
grid_size_occ = [512, 512, 40]
coarse_ratio = 2
```

所以 coarse grid 变成：

```text
[512/2, 512/2, 40/2] = [256, 256, 20]
```

对应物理体素大小：

```text
0.4m x 0.4m x 0.4m
```

所以 PointOcc 原始 coarse logits 是：

```text
[B, 17, 256, 256, 20]
```

然后在 eval 阶段再插值回：

```text
[B, 17, 512, 512, 40]
```

---

### 6.5 `coarse_ratio` 的意义

`coarse_ratio` 是：

**PointOcc 从最终 occupancy 网格降到 coarse occupancy 网格的倍率。**

公式：

```text
coarse_grid = final_grid / coarse_ratio
```

例如：

- `coarse_ratio = 2`
  ```text
  [512, 512, 40] -> [256, 256, 20]
  ```

- `coarse_ratio = 4`
  ```text
  [512, 512, 40] -> [128, 128, 10]
  ```

如果你未来要把 PointOcc 和当前 FLC 粗网格对齐，最自然的做法就是把 PointOcc 的 coarse grid 也改成：

```text
[128, 128, 10]
```

也就是设置：

```text
coarse_ratio = 4
```

---

### 6.6 为什么同一份最终标签，会出现多个“分辨率”

因为不同阶段的任务不同：

- `256 x 704`
  是相机输入图像分辨率
- `16 x 44`
  是图像 backbone 输出特征图分辨率
- `128 x 128 x 10`
  是 OpenOccupancy / FLC 的 coarse occupancy 分辨率
- `256 x 256 x 20`
  是 PointOcc 原始 coarse occupancy 分辨率
- `512 x 512 x 40`
  是最终监督与评估分辨率

可以把它们理解成两条不同坐标链：

#### Camera 链

```text
图像分辨率:
256 x 704
-> 图像特征图:
16 x 44
-> coarse occupancy:
128 x 128 x 10
-> final occupancy:
512 x 512 x 40
```

#### PointOcc 链

```text
LiDAR TPV plane / cylindrical grid
-> coarse occupancy:
256 x 256 x 20   # 原始配置
-> final occupancy:
512 x 512 x 40
```

所以不要把：

- 图像分辨率
- 特征图分辨率
- coarse occupancy 分辨率
- final occupancy 分辨率

混成同一种“分辨率”。

---

### 6.7 PointOcc 的 TPV 平面尺寸是什么意思

PointOcc 原始配置里：

```python
tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2
```

所以 TPVAggregator 里做完插值后的平面尺寸分别是：

- `tpv_hw`:
  ```text
  [B, C, 360, 480]
  ```
- `tpv_zh`:
  ```text
  [B, C, 32, 360]
  ```
- `tpv_wz`:
  ```text
  [B, C, 480, 32]
  ```

这里的 480/360/32 对应的是 PointOcc 内部使用的 **柱坐标 TPV 网格分辨率**，  
不是最终 occupancy 分辨率。

它们只是提供一个可供 coarse voxel query 去 `grid_sample` 的 2D feature space。

---

## 7. PointOcc 的 `TPVAggregator_Occ` 更细的张量流

下面专门展开 PointOcc 文档中这一段：

```text
-> processed TPV feature list
[tpv_xy, tpv_yz, tpv_zx]

-> TPVAggregator_Occ
  - interpolate TPV planes
  - project coarse voxel queries to 3 TPV planes
  - grid_sample from each plane
  - fuse 3 sampled features
  - decoder (MLP)
  - classifier
```

### 7.1 输入到 `TPVAggregator_Occ` 的张量

来自：

- [/home/shkong/MyProject/PointOcc/model/pointtpv_occ.py](/home/shkong/MyProject/PointOcc/model/pointtpv_occ.py)

输入主要有两类：

#### A. TPV feature list

经过 `CylinderEncoder_Occ -> Swin -> GeneralizedLSSFPN` 后：

```text
tpv_xy: [B, C, W_tpv, H_tpv]
tpv_yz: [B, C, H_tpv, Z_tpv]
tpv_zx: [B, C, Z_tpv, W_tpv]
```

配置里 `_dim_ = 192`，所以这里的 `C` 通常是 192。

#### B. coarse voxel queries

来自 dataloader 的：

```text
voxels_coarse: [B, N, 3]
```

其中：

```text
N = Wc * Hc * Dc
```

在原始 PointOcc 配置里：

```text
Wc, Hc, Dc = 256, 256, 20
N = 256 * 256 * 20
```

这些 query 不是图像像素，也不是最终 dense voxel feature，  
而是“我要在哪些 coarse occupancy 网格位置上取特征并分类”。

---

### 7.2 TPV plane 重排和插值

代码在：

- [/home/shkong/MyProject/PointOcc/model/tpv_aggregator.py](/home/shkong/MyProject/PointOcc/model/tpv_aggregator.py)

首先把三张 plane 重排成统一习惯：

```python
tpv_xy -> tpv_hw
tpv_yz -> tpv_zh
tpv_zx -> tpv_wz
```

然后根据 `scale_h / scale_w / scale_z` 做 `F.interpolate`。

原始配置下，插值后大致为：

```text
tpv_hw: [B, 192, 360, 480]
tpv_zh: [B, 192,  32, 360]
tpv_wz: [B, 192, 480,  32]
```

这一步的意义是：

**把三张 TPV plane 放到一个更细、更统一的 2D采样分辨率上。**

---

### 7.3 coarse voxel queries 归一化

接着把 `voxels_coarse` reshape 成：

```text
[B, 1, N, 3]
```

然后每个坐标分量按 TPV 尺寸归一化到 `[-1, 1]`：

```python
voxels_coarse[..., 0] = voxels_coarse[..., 0] / (tpv_w * scale_w) * 2 - 1
voxels_coarse[..., 1] = voxels_coarse[..., 1] / (tpv_h * scale_h) * 2 - 1
voxels_coarse[..., 2] = voxels_coarse[..., 2] / (tpv_z * scale_z) * 2 - 1
```

这样它们就能被 `F.grid_sample` 当作采样坐标使用。

---

### 7.4 从三张 TPV plane 上分别采样

三次采样分别对应三种二维投影：

#### 在 `tpv_hw` 上采样

用：

```text
[w, h]
```

得到：

```text
tpv_hw_vox: [B, C, N]
```

#### 在 `tpv_zh` 上采样

用：

```text
[h, z]
```

得到：

```text
tpv_zh_vox: [B, C, N]
```

#### 在 `tpv_wz` 上采样

用：

```text
[z, w]
```

得到：

```text
tpv_wz_vox: [B, C, N]
```

每个 coarse voxel query 都会从三张 plane 拿到一份特征。

---

### 7.5 三张 plane 特征融合

代码：

```python
fused = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox
```

所以：

```text
fused: [B, C, N]
```

然后 permute 成：

```text
[B, N, C]
```

这里的 `fused` 非常关键：

- 它不是 dense 3D volume
- 它是“每个 coarse voxel query 对应一条 feature vector”

所以更准确地说，它是：

**coarse voxel query feature table**

---

### 7.6 decoder 和 classifier

接着：

```python
fused -> decoder -> classifier
```

形状变化：

```text
[B, N, 192]
-> decoder
[B, N, 192]
-> classifier
[B, N, 17]
```

再转置、reshape 成：

```text
[B, 17, Wc, Hc, Dc]
```

在原始 PointOcc 配置里就是：

```text
[B, 17, 256, 256, 20]
```

---

### 7.7 eval 时为什么又变回 `512 x 512 x 40`

这是因为 PointOcc 的 coarse logits 只是 coarse 分辨率结果。  
在 eval 分支里，会做：

```python
F.interpolate(logits, size=[W_, H_, D_], mode='trilinear')
```

其中：

```text
W_, H_, D_ = voxel_label.shape = [512, 512, 40]
```

所以最终评估时，PointOcc 仍然要和 OpenOccupancy 一样，在：

```text
512 x 512 x 40
```

这个最终 occupancy 网格上比较指标。
