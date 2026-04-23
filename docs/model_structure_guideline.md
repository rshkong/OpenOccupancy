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

---

## 8. PointOcc 原版源码阅读指南

这一节只针对 `~/MyProject/PointOcc` 原版 PointOcc occupancy 网络，按“配置 -> dataloader -> model forward -> CylinderEncoder -> Swin/FPN -> TPVAggregator -> loss/eval”的真实代码顺序阅读。

核心文件：

- [/home/shkong/MyProject/PointOcc/config/pointtpv_nusc_occ.py](/home/shkong/MyProject/PointOcc/config/pointtpv_nusc_occ.py:31)
- [/home/shkong/MyProject/PointOcc/dataloader/dataset.py](/home/shkong/MyProject/PointOcc/dataloader/dataset.py:40)
- [/home/shkong/MyProject/PointOcc/dataloader/dataset_wrapper.py](/home/shkong/MyProject/PointOcc/dataloader/dataset_wrapper.py:188)
- [/home/shkong/MyProject/PointOcc/model/pointtpv_occ.py](/home/shkong/MyProject/PointOcc/model/pointtpv_occ.py:7)
- [/home/shkong/MyProject/PointOcc/model/cylinder_encoder.py](/home/shkong/MyProject/PointOcc/model/cylinder_encoder.py:106)
- [/home/shkong/MyProject/PointOcc/model/swin.py](/home/shkong/MyProject/PointOcc/model/swin.py:746)
- [/home/shkong/MyProject/PointOcc/model/fpn.py](/home/shkong/MyProject/PointOcc/model/fpn.py:80)
- [/home/shkong/MyProject/PointOcc/model/tpv_aggregator.py](/home/shkong/MyProject/PointOcc/model/tpv_aggregator.py:157)

### 8.1 原版 PointOcc 的完整流程图

```text
nuScenes LiDAR .bin + occupancy .npy

-> Occ_Point_NuScenes.__getitem__()
   points: [N_pts, 5]
   pcd:    [N_occ, 4]  # [z, y, x, cls]

-> Occ_DatasetWrapper_Point_NuScenes.__getitem__()
   cartesian xyz -> cylindrical xyz_pol = [rho, phi, z]
   grid_ind = floor((clip(xyz_pol) - min_bound) / intervals)
   return_feat = concat(
       xyz_pol - voxel_center,
       xyz_pol,
       xyz[:2],
       points[:, 3:]
   )

   return:
   voxel_position_grid_coarse: [N_coarse, 3]
   processed_label:            [512, 512, 40]
   grid_ind:                   [N_pts, 3]
   return_feat:                [N_pts, 10]

-> occ_custom_collate_fn
   voxel_position_coarse: [B, N_coarse, 3]
   points:                [B, N_pts, 10]
   voxel_label:           [B, 512, 512, 40]
   grid_ind:              [B, N_pts, 3]

-> PointTPV_Occ.forward()
   extract_lidar_feat(points, grid_ind)

-> CylinderEncoder_Occ
   point_mlp:    [N_pts, 10] -> [N_pts, 128]
   scatter_max:  per cylindrical voxel -> sparse voxel features
   SparseConvTensor over [480, 360, 32]
   SparseMaxPool3d along z / rho / phi

   output 3 TPV planes:
   tpv_xy: [B, 128, 480, 360]
   tpv_yz: [B, 128, 360, 32]
   tpv_zx: [B, 128, 32, 480]

-> Swin backbone
   each TPV plane independently goes through patch embedding + Swin stages
   output per plane: multi-scale features

-> GeneralizedLSSFPN
   top-down upsample + concat + ConvModule
   output per plane: processed TPV feature, C=192

-> TPVAggregator_Occ
   interpolate TPV planes to sampling resolution
   normalize coarse voxel queries to [-1, 1]
   grid_sample from 3 planes:
     [w, h] from tpv_hw
     [h, z] from tpv_zh
     [z, w] from tpv_wz
   fused = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox

-> decoder + classifier
   fused:  [B, N_coarse, 192]
   logits: [B, N_coarse, 17]
   reshape:
   [B, 17, 256, 256, 20]

-> training:
   compute CE + Lovasz + semantic scaling + geometric scaling loss

-> eval:
   trilinear interpolate to [B, 17, 512, 512, 40]
```

### 8.2 配置里的关键尺寸

原版配置在 [/home/shkong/MyProject/PointOcc/config/pointtpv_nusc_occ.py](/home/shkong/MyProject/PointOcc/config/pointtpv_nusc_occ.py:31)：

```python
_dim_ = 192

tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2

grid_size = [480, 360, 32]
grid_size_occ = [512, 512, 40]
coarse_ratio = 2
nbr_class = 17
```

含义：

```text
grid_size = [480, 360, 32]
```

这是 LiDAR 点云在柱坐标系下的离散网格，三个维度对应：

```text
[rho, phi, z]
```

```text
grid_size_occ = [512, 512, 40]
```

这是最终 occupancy 标签网格。

```text
coarse_ratio = 2
```

表示 PointOcc 不直接预测 `[512,512,40]`，而是先预测：

```text
[512/2, 512/2, 40/2] = [256, 256, 20]
```

```text
tpv_w/h/z * scale_w/h/z = [480, 360, 32]
```

这是 `TPVAggregator_Occ` 采样时使用的逻辑 TPV 坐标范围，和 `grid_size` 对齐。

### 8.3 dataloader 如何把点云变成 PointOcc 输入

原始点云读取在 [/home/shkong/MyProject/PointOcc/dataloader/dataset.py](/home/shkong/MyProject/PointOcc/dataloader/dataset.py:59)：

```python
points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])
```

如果使用 multi-sweep，历史 sweep 会被变换到当前 LiDAR 坐标系，并且代码会写入时间差：

```python
points_sweep[:, :3] = points_sweep[:, :3] @ R.T + t
points_sweep[:, 4] = ts - sweep_ts
```

所以 PointOcc 的 `points[:, 3:]` 更准确地说是 LiDAR 的附加特征，不应该简单理解成固定的 `[intensity, ring]`。多 sweep 下最后一维经常承担 time lag。

包装器在 [/home/shkong/MyProject/PointOcc/dataloader/dataset_wrapper.py](/home/shkong/MyProject/PointOcc/dataloader/dataset_wrapper.py:215) 做三件事。

第一，笛卡尔坐标转柱坐标：

```text
rho = sqrt(x^2 + y^2)
phi = atan2(y, x)
z   = z
xyz_pol = [rho, phi, z]
```

第二，计算每个点的柱坐标网格索引：

```python
intervals = (max_bound - min_bound) / grid_size
grid_ind = floor((clip(xyz_pol) - min_bound) / intervals)
```

也就是：

```text
grid_ind[i] = floor((xyz_pol[i] - min_bound) / voxel_size_cyl)
```

第三，构建 10 维点特征：

```python
voxel_centers = (grid_ind + 0.5) * intervals + min_bound
return_xyz = xyz_pol - voxel_centers
return_feat = concat(return_xyz, xyz_pol, xyz[:, :2], feat)
```

形状：

```text
return_xyz: [N_pts, 3]  # 点相对所在柱坐标 voxel 中心的偏移
xyz_pol:    [N_pts, 3]  # 点的绝对柱坐标
xyz[:, :2]: [N_pts, 2]  # 原始 x,y
feat:       [N_pts, 2]  # 原始点云附加特征

return_feat: [N_pts, 10]
```

同时，dataloader 还会生成所有 coarse occupancy query 的柱坐标连续索引：

```python
voxel_position_coarse = centers of [256, 256, 20] Cartesian coarse grid
voxel_position_grid_coarse = (cart2polar(voxel_position_coarse) - min_bound) / intervals_vox
```

这里的 `voxel_position_grid_coarse` 不是点云，也不是特征，而是后面用于 `grid_sample` 的 query 坐标表。

### 8.4 PointTPV_Occ.forward 的主调用链

入口在 [/home/shkong/MyProject/PointOcc/model/pointtpv_occ.py](/home/shkong/MyProject/PointOcc/model/pointtpv_occ.py:43)：

```python
x_lidar_tpv = self.extract_lidar_feat(points=points, grid_ind=grid_ind)
outs = self.tpv_aggregator(
    x_lidar_tpv,
    voxels=grid_ind_vox,
    voxels_coarse=grid_ind_vox_coarse,
    voxel_label=voxel_label,
    return_loss=return_loss)
```

`extract_lidar_feat()` 的顺序是：

```python
x_3view = self.lidar_tokenizer(points, grid_ind)
x_tpv = self.lidar_backbone(x_3view)
for x in x_tpv:
    x = self.lidar_neck(x)
    tpv_list.append(x[0])
```

对应流程：

```text
CylinderEncoder_Occ -> Swin -> GeneralizedLSSFPN -> TPVAggregator_Occ
```

### 8.5 CylinderEncoder_Occ 的核心实现

代码在 [/home/shkong/MyProject/PointOcc/model/cylinder_encoder.py](/home/shkong/MyProject/PointOcc/model/cylinder_encoder.py:158)。

第一步，把 batch id 拼到每个点的网格索引前面：

```python
cat_pt_ind.append(F.pad(grid_ind[i_batch], (1, 0), value=i_batch))
```

原来：

```text
grid_ind: [N_pts, 3] = [rho_idx, phi_idx, z_idx]
```

变成：

```text
cat_pt_ind: [N_pts, 4] = [batch_idx, rho_idx, phi_idx, z_idx]
```

第二步，point-wise MLP：

```python
processed_cat_pt_fea = self.point_mlp(cat_pt_fea)
```

形状：

```text
[N_pts, 10] -> [N_pts, 128]
```

第三步，同一个柱坐标 voxel 内的点做 max 聚合：

```python
unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]
```

公式可以理解成：

```text
voxel_feat[v] = max_{i | grid_ind_i = v} point_mlp(point_feat_i)
```

第四步，构造稀疏柱坐标体：

```python
ret = SparseConvTensor(processed_pooled_data, coors, np.array(self.grid_size), batch_size)
```

此时逻辑空间是：

```text
[rho, phi, z] = [480, 360, 32]
```

第五步，用三种 `SparseMaxPool3d` 得到三张 TPV plane：

```python
tpv_xy = pool along z
tpv_yz = pool along rho
tpv_zx = pool along phi
```

原版配置 `split=[8,8,8]`，因此：

```text
pool_xy: kernel [1, 1, 4]   # 32 / 8 = 4，压缩 z
pool_yz: kernel [60, 1, 1]  # 480 / 8 = 60，压缩 rho
pool_zx: kernel [1, 45, 1]  # 360 / 8 = 45，压缩 phi
```

压缩后的 `split` 维度会被 flatten 到 channel，再用一个小 MLP 压回 `128` 通道。

输出形状：

```text
tpv_xy: [B, 128, 480, 360]
tpv_yz: [B, 128, 360, 32]
tpv_zx: [B, 128, 32, 480]
```

这一步是 PointOcc 的关键：它没有把所有空间都留在一个 dense 3D tensor 中，而是用三张 2D plane 分别保留三组投影视角。

### 8.6 Swin + GeneralizedLSSFPN 如何处理三张 TPV plane

Swin 在 [/home/shkong/MyProject/PointOcc/model/swin.py](/home/shkong/MyProject/PointOcc/model/swin.py:746) 里对三张 plane 使用同一套逻辑：

```python
for x in x_3view:
    x, hw_shape = self.patch_embed_lidar(x)
...
for stage in self.stages:
    for j in range(len(x_tpv)):
        x_tpv[j], hw_shape_tpv[j], out, out_hw_shape = stage(...)
        if i in self.out_indices:
            outs[j].append(out)
```

注意这里不是把三张 plane concat 后一起过 Swin，而是：

```text
tpv_xy -> Swin -> multi-scale xy features
tpv_yz -> Swin -> multi-scale yz features
tpv_zx -> Swin -> multi-scale zx features
```

`GeneralizedLSSFPN` 在 [/home/shkong/MyProject/PointOcc/model/fpn.py](/home/shkong/MyProject/PointOcc/model/fpn.py:80) 做 top-down 融合：

```python
x = F.interpolate(laterals[i + 1], size=laterals[i].shape[2:])
laterals[i] = torch.cat([laterals[i], x], dim=1)
laterals[i] = self.lateral_convs[i](laterals[i])
laterals[i] = self.fpn_convs[i](laterals[i])
```

输出被 `PointTPV_Occ.extract_lidar_feat()` 取第一个尺度：

```python
if not isinstance(x, torch.Tensor):
    x = x[0]
```

因此传给 `TPVAggregator_Occ` 的是三张已经过 Swin/FPN 处理的 TPV feature plane，通道数通常是：

```text
C = _dim_ = 192
```

### 8.7 TPVAggregator_Occ 的公式化理解

代码在 [/home/shkong/MyProject/PointOcc/model/tpv_aggregator.py](/home/shkong/MyProject/PointOcc/model/tpv_aggregator.py:188)。

输入：

```text
tpv_xy:        [B, C, W_tpv, H_tpv]
tpv_yz:        [B, C, H_tpv, Z_tpv]
tpv_zx:        [B, C, Z_tpv, W_tpv]
voxels_coarse: [B, N, 3]
```

先重排为 PyTorch `grid_sample` 更习惯的二维图格式：

```python
tpv_hw = tpv_xy.permute(0, 1, 3, 2)
tpv_wz = tpv_zx.permute(0, 1, 3, 2)
tpv_zh = tpv_yz.permute(0, 1, 3, 2)
```

然后插值到逻辑 TPV 采样分辨率：

```text
tpv_hw -> [B, C, 360, 480]
tpv_zh -> [B, C,  32, 360]
tpv_wz -> [B, C, 480,  32]
```

把 query 坐标归一化：

```text
w_norm = w / (tpv_w * scale_w) * 2 - 1
h_norm = h / (tpv_h * scale_h) * 2 - 1
z_norm = z / (tpv_z * scale_z) * 2 - 1
```

然后分别采样：

```python
F_hw(q) = grid_sample(tpv_hw, [w_norm, h_norm])
F_zh(q) = grid_sample(tpv_zh, [h_norm, z_norm])
F_wz(q) = grid_sample(tpv_wz, [z_norm, w_norm])
```

融合公式：

```text
F(q) = F_hw(q) + F_zh(q) + F_wz(q)
```

其中 `q` 是一个 coarse voxel query。

形状变化：

```text
F_hw(q), F_zh(q), F_wz(q): [B, C, N]
fused:                    [B, C, N]
permute:                  [B, N, C]
```

最后逐 query 分类：

```python
fused = decoder(fused)
logits = classifier(fused)
```

形状：

```text
[B, N, 192]
-> [B, N, 192]
-> [B, N, 17]
-> [B, 17, 256, 256, 20]
```

### 8.8 训练 loss 和评估输出

训练入口在 [/home/shkong/MyProject/PointOcc/train_occ.py](/home/shkong/MyProject/PointOcc/train_occ.py:187)：

```python
loss = my_model(
    points=points,
    grid_ind=train_grid,
    grid_ind_vox_coarse=train_grid_vox_coarse,
    voxel_label=voxel_label,
    return_loss=True)
```

`TPVAggregator_Occ` 里如果 `return_loss=True`，会计算：

```text
loss = CE
     + Lovasz
     + semantic scaling loss
     + geometric scaling loss
```

如果 coarse logits 是 `[256,256,20]`，而 GT 是 `[512,512,40]`，代码会先把 GT 按 `ratio=2` 聚合到 coarse 分辨率：

```python
voxel_label_coarse = voxel_label.reshape(
    B, W, ratio, H, ratio, D, ratio
).permute(...).reshape(B, W, H, D, ratio**3)
```

这里的逻辑是：一个 coarse voxel 覆盖 `ratio^3` 个 final voxel，然后用众数作为 coarse label；空类 `0` 会被特殊处理，避免大面积 empty 直接压过非空类别。

评估入口在 [/home/shkong/MyProject/PointOcc/eval_occ.py](/home/shkong/MyProject/PointOcc/eval_occ.py:121)：

```python
predict_labels_vox = my_model(..., return_loss=False)
predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
```

`return_loss=False` 时，`TPVAggregator_Occ` 会把 coarse logits 插值回 GT 尺寸：

```python
pred = F.interpolate(
    logits,
    size=[512, 512, 40],
    mode='trilinear',
    align_corners=False)
```

所以原版 PointOcc 的实际训练监督发生在 coarse 分辨率 `[256,256,20]`，但最终评估在 `[512,512,40]`。

### 8.9 对 FLC-PointOcc 融合的直接启发

原版 PointOcc 最适合迁移的不是最终 `classifier` 输出，而是 `TPVAggregator_Occ` 中这一步之前或这一步得到的 query feature：

```text
fused: [B, N, 192]
```

如果你要和 FLC 的 BEV 对齐，有两条路线：

```text
路线 A：先 query，再 reshape
fused [B, N, C]
-> reshape [B, X, Y, Z, C]
-> permute [B, C, X, Y, Z]
-> flatten Z
-> [B, C*Z, X, Y]
-> 与 FLC BEV feature 融合
```

```text
路线 B：保留 TPV plane，不急着 query
tpv_xy / tpv_yz / tpv_zx
-> 为 FLC 的 [X,Y,Z] 网格生成 query
-> grid_sample 得到 [B, C, X, Y, Z]
-> flatten Z
-> 与 FLC BEV feature 融合
```

当前 FLC-PointOcc 更接近路线 B 的变体：先用 TPV fuser 得到 `[B,C,X,Y,Z]`，再把 `Z` 压进 channel 和 FLC BEV 做融合。检查这类融合时，最重要的是确认三件事：

```text
1. PointOcc 的 query 坐标是否和 FLC 的 [128,128,10] 网格严格对齐
2. TPV 采样坐标归一化分母是否等于真实 TPV 逻辑尺寸
3. flatten Z 后的 channel 顺序是否和 FLCOccHead 的 C2H 预期一致
```
