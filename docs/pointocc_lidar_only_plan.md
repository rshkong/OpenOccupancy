# LiDAR-only PointOcc Reproduction Plan

## Context

The `FLCPointOccNet` fusion model (Camera FlashOcc + LiDAR PointOcc-TPV)
currently underperforms **both** baselines it should dominate:

- Pure FlashOcc (camera-only) — our intended lower bound
- PointOcc (LiDAR-only) — the reference for the LiDAR branch

Naively, if the encoder can learn to ignore the LiDAR branch, the fusion
model should never be worse than FlashOcc. The fact that it is tells us
the LiDAR branch actively injects noise into the fusion path.

Two plausible causes:

1. **Bug in the LiDAR branch port** (CylinderEncoder / TPVSwin / TPVFPN /
   TPVFuser / `_prepare_lidar_inputs`) — code was moved from `PointOcc`
   into `OpenOccupancy` with registry / import / convention edits, and
   any silent mismatch breaks the downstream signal.
2. **Fusion mechanics** — scale mismatch between branches, poor init,
   gradient interference, etc.

To disentangle these, this plan builds a **LiDAR-only baseline** that
uses the exact same ported LiDAR modules (so a successful run proves
the port is correct) and reproduces PointOcc's reference numbers. If
it matches PointOcc, cause (1) is ruled out and we focus on (2). If it
diverges, we fix the port — and FLC-PointOcc benefits automatically.

During code review for the port, one bug was already identified that
alone could explain the degradation (see **Phase A.1**). Fixing it is a
prerequisite for either baseline to train well.

## Critical bug: `voxels_coarse` normalization mismatch

`TPVFuser.forward` normalizes grid-sample coordinates by
`tpv_w * scale_w` etc. It expects `voxels_coarse` in range
`[0, tpv_w*scale_w)` so that the normalized result is `[-1, 1)`.

`FLCPointOccNet._prepare_lidar_inputs` produces
`vc_grid = (vc_pol - min_bound) / intervals` where
`intervals = (max_bound - min_bound) / cyl_grid_size`, giving a range
of `[0, cyl_grid_size)`.

With the current config:

| Quantity | Value |
|---|---|
| `cyl_grid_size` | `[240, 180, 16]` |
| `tpv_w * scale_w` | `60 * 2 = 120` |
| `tpv_h * scale_h` | `45 * 2 = 90` |
| `tpv_z * scale_z` | `4  * 2 = 8` |

`voxels_coarse` range: `[0, 240) × [0, 180) × [0, 16)`.
After `TPVFuser`'s `coord / (tpv_dim * scale) * 2 - 1`:
`[-1, 3) × [-1, 3) × [-1, 3)`.

With `padding_mode='border'`, every coordinate > 1 clamps to the TPV
plane's border — so a large fraction of voxels sample the same border
value. The LiDAR branch effectively delivers **near-constant noise**
for those voxels, which is exactly what would make fusion strictly
worse than camera-only.

In PointOcc's original config, `grid_size = [480, 360, 32]` is set
equal to `[tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z]` precisely
so that this normalization round-trips to `[-1, 1)`.

**Fix options:**

- (a) In `_prepare_lidar_inputs`, rescale `vc_grid` so that it lands in
      `[0, tpv_w*scale_w)` regardless of `cyl_grid_size`. Minimal
      config change; preserves `cyl_grid_size` for CylinderEncoder
      (VRAM-friendly setting).
- (b) Set `cyl_grid_size = [tpv_w*scale_w, tpv_h*scale_h, tpv_z*scale_z]`
      in the config (i.e. `[120, 90, 8]`). Mirrors PointOcc's structure
      but changes the point-voxelization resolution, which has its own
      effects.

We choose **(a)**. Rationale:

- No change to CylinderEncoder input resolution, so point-feature
  fidelity stays the same.
- Normalization is a pure scaling, decoupled from voxelization — the
  right place to fix it is the coordinate producer.
- The fix can be expressed as one-line change in `_prepare_lidar_inputs`
  with an explicit argument for the target range.

## Architecture (LiDAR-only baseline)

```
points  →  _prepare_lidar_inputs  →  (10ch points, grid_ind, voxels_coarse)
                                                  ↓
                                          CylinderEncoder
                                                  ↓
                                     TPV planes [tpv_xy, tpv_yz, tpv_zx]
                                                  ↓
                                             TPVSwin
                                                  ↓
                                              TPVFPN
                                                  ↓
                                 TPVAggregator (NEW; port of TPVAggregator_Occ)
                                   • grid_sample × 3 planes
                                   • elem-wise sum
                                   • Linear → Softplus → Linear  (decoder MLP)
                                   • Linear → num_classes        (classifier)
                                                  ↓
                                  logits [B, 17, W_c, H_c, D_c]
                                                  ↓
                        CE + Lovasz_softmax + sem_scal + geo_scal  (weights [1,1,1,1])
```

No camera branch, no 2D occ encoder, no FLCOccHead.

## Design choices (confirmed with user)

1. **Detector**: new class `PointOccNet` (separate file), NOT a subclass of
   `FLCPointOccNet`. `_prepare_lidar_inputs` is extracted to a mixin to
   avoid duplication.
2. **Loss**: strict PointOcc reproduction — CE + lovasz_softmax + sem_scal +
   geo_scal, weights `[1,1,1,1]`, `ignore_index=255`. These functions are
   ported from PointOcc into `projects/occ_plugin/utils/`.
3. **Bug fix**: the `voxels_coarse` normalization bug is fixed in
   `_prepare_lidar_inputs` as a prerequisite — both the LiDAR-only
   baseline and the FLC-PointOcc fusion model benefit.

## File plan

### New files

| Path | Purpose |
|---|---|
| `projects/occ_plugin/utils/lovasz_losses.py` | Ported from PointOcc |
| `projects/occ_plugin/utils/sem_geo_loss.py` | Ported from PointOcc (sem_scal + geo_scal) |
| `projects/occ_plugin/occupancy/lidar_encoder/tpv_aggregator.py` | New `TPVAggregator` (grid-sample + decoder + classifier, logits only) |
| `projects/occ_plugin/occupancy/detectors/lidar_prep_mixin.py` | `LidarPrepMixin` — `_prepare_lidar_inputs`, `_cart2polar`, `extract_lidar_tpv` |
| `projects/occ_plugin/occupancy/detectors/pointocc_net.py` | `PointOccNet` detector |
| `projects/configs/baselines/LiDAR_pointocc_128x128x10_4070ti.py` | LiDAR-only config |

### Modified files

| Path | Change |
|---|---|
| `projects/occ_plugin/occupancy/lidar_encoder/__init__.py` | Register `TPVAggregator` |
| `projects/occ_plugin/occupancy/detectors/__init__.py` | Register `PointOccNet` |
| `projects/occ_plugin/occupancy/detectors/flc_pointocc_net.py` | Use `LidarPrepMixin`; fix `voxels_coarse` scaling |

### Reused as-is

- `projects/occ_plugin/occupancy/lidar_encoder/cylinder_encoder.py`
- `projects/occ_plugin/occupancy/lidar_encoder/tpv_swin.py`
- `projects/occ_plugin/occupancy/lidar_encoder/tpv_fpn.py`

## Implementation phases

### Phase A — Prerequisites (bug fix + loss port)

1. **Verify the bug** by reading `CylinderEncoder` and `TPVSwin` to
   confirm actual TPV feature-map sizes vs. config assumptions. Quantify
   the fraction of `voxels_coarse` that would fall out of `[-1, 1]`.
2. **Fix `_prepare_lidar_inputs`** in `flc_pointocc_net.py` so
   `voxels_coarse[..., i] ∈ [0, tpv_dim_i * scale_i)`. New signature accepts
   the target range as parameters, with a default matching current config.
3. **Port loss functions**: copy `lovasz_losses.py` and `sem_geo_loss.py`
   from `/home/shkong/MyProject/PointOcc/utils/` to
   `projects/occ_plugin/utils/`, adjusting imports.

### Phase B — LiDAR-only baseline implementation

4. **`TPVAggregator`**: replicates `TPVAggregator_Occ.forward` up to the
   `logits` tensor (drops the built-in loss block — detector owns loss).
   Returns logits at coarse occupancy resolution `[B, C, W_c, H_c, D_c]`.
5. **`LidarPrepMixin`**: extract `_prepare_lidar_inputs`, `_cart2polar`,
   `extract_lidar_tpv` out of `FLCPointOccNet`. Both detectors inherit.
6. **`PointOccNet`**:
    - `__init__`: build `lidar_tokenizer`, `lidar_backbone`, `lidar_neck`,
      `tpv_aggregator`. No camera modules.
    - `forward_train(points, gt_occ, ...)`:
        - `_prepare_lidar_inputs(points)` → `(points_10ch, grid_ind, voxels_coarse)`
        - `extract_lidar_tpv(...)` → `tpv_list`
        - `tpv_aggregator(tpv_list, voxels_coarse)` → `logits`
        - Coarsen `gt_occ` to match logits grid
        - Compute `loss = Σ w_i · loss_i(logits, voxel_label)`
    - `forward_test`: same up to `logits`, trilinear upsample to
      `gt_occ` resolution, return per-voxel prediction for eval hook.
7. **Config `LiDAR_pointocc_128x128x10_4070ti.py`**:
    - `input_modality.use_camera = False`
    - Drop `LoadMultiViewImageFromFiles_BEVDet`, `depth_gt_path`, etc.
    - Use `PointOccNet` as `model.type`
    - Mirror PointOcc `pointtpv_nusc_occ.py` numerics (Swin init, FPN,
      aggregator dims), except keep the halved `cyl_grid_size` for 4070 Ti VRAM.
    - Match PointOcc's `samples_per_gpu=1`, AdamW, cosine schedule.
    - Load Swin pretrained (`pretrain/swin_tiny_patch4_window7_224.pth`)
      if available.
8. **Register modules** in the two `__init__.py` files.

### Phase C — Validation

9. **Smoke train** 1–2 epochs on the LiDAR-only config. Check:
    - No crashes through training + evaluation.
    - SC/SSC mIoU trending up (not flat at near-zero).
    - Rough comparison vs. PointOcc's reported numbers
      (`/home/shkong/MyProject/PointOcc/README.md`).
10. **Re-baseline FLC-PointOcc**: with the `voxels_coarse` bug fixed,
    re-train the fusion model (with `debug_zero_lidar=False`) and
    confirm it's no worse than FlashOcc.

## Verification plan

Two training commands, two work directories:

```bash
# LiDAR-only baseline
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/LiDAR_pointocc_128x128x10_4070ti.py \
    --work-dir work_dirs/LiDAR_pointocc_128x128x10_4070ti \
    --seed 0

# Fusion with bug fix (debug_zero_lidar=False)
PYTHONPATH="./":$PYTHONPATH python tools/train.py \
    projects/configs/baselines/CAM-LiDAR_flc_pointocc_128x128x10_4070ti.py \
    --work-dir work_dirs/CAM-LiDAR_flc_pointocc_128x128x10_4070ti_bugfix \
    --seed 0
```

Decision matrix:

| LiDAR-only result | Fusion result (bug-fixed) | Conclusion |
|---|---|---|
| ≈ PointOcc reported | ≥ FlashOcc | Port verified, fusion working |
| ≈ PointOcc reported | < FlashOcc | Port fine, fusion mechanics broken (proceed to Step 3–6 of the fusion improvement plan) |
| ≪ PointOcc reported | — | More bugs in LiDAR port, diagnose further |

## Version control

Tags carved along the way:

- `flc-pointocc-v1` — already set, initial fusion baseline
- `flc-pointocc-v2-bugfix` — after Phase A.2 (bug fix landed)
- `pointocc-lidar-only-v1` — after Phase B complete (baseline config runs)

## Out of scope (for later)

- Swin pretrained loading behavior beyond a direct path (no checkpoint conversion).
- Camera-branch improvements (depth loss tuning, etc.).
- Replacing the 2D occ encoder in FLC-PointOcc — we first want the fusion
  mechanics to work with the existing 2D path.
