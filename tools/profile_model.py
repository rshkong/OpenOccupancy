"""Profile inference/training efficiency of OpenOccupancy models.

Supports all model types:
  - OccNet (camera-only): Cam-OO, FlashOcc
  - OccNet (lidar-only): LiDAR-OO
  - OccNet (multimodal): MM-OO
  - PointOccNet: PointOcc (LiDAR-only TPV)
  - FLCPointOccNet: FLC-PointOcc A/B (Camera + LiDAR fusion)

Usage examples:
  # FPS + memory for FlashOcc
  python tools/profile_model.py \\
      --config projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py \\
      --checkpoint work_dirs/flc_step2/best_SSC_mean.pth

  # Parameter count only
  python tools/profile_model.py --config ... --checkpoint ... --params

  # FLOPs estimate
  python tools/profile_model.py --config ... --checkpoint ... --flops

  # Training memory (forward + backward, batch=4)
  python tools/profile_model.py --config ... --checkpoint ... --train-mem --batch-size 4

  # LiDAR-only PointOcc
  python tools/profile_model.py \\
      --config projects/configs/baselines/LiDAR_pointocc_128x128x10_server.py \\
      --checkpoint work_dirs/LiDAR_pointocc/best_SSC_mean.pth

  # FLC-PointOcc dual-modal
  python tools/profile_model.py \\
      --config projects/configs/baselines/CAM-LiDAR_flc_pointocc_camadapt256_128x128x10.py \\
      --checkpoint work_dirs/CAM-LiDAR_flc_pointocc_camadapt256/best_SSC_mean.pth
"""
import argparse
import importlib
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile OpenOccupancy models (latency / memory / FLOPs / params)')
    parser.add_argument(
        '--config',
        default='projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py',
        help='config file path')
    parser.add_argument(
        '--checkpoint',
        default='',
        help='checkpoint (.pth) path; leave empty to profile randomly initialised model')
    parser.add_argument(
        '--mode',
        choices=['extract', 'full'],
        default='full',
        help='extract: stop after occ encoder; full: include occ head')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--measure', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-cams', type=int, default=None,
                        help='override number of cameras from config')
    parser.add_argument('--num-points', type=int, default=20000,
                        help='dummy LiDAR points per sample')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--params', action='store_true',
                        help='print parameter count and exit (no forward pass)')
    parser.add_argument('--flops', action='store_true',
                        help='estimate GFLOPs (requires fvcore or thop)')
    parser.add_argument('--train-mem', action='store_true',
                        help='measure peak memory during forward+backward (training mode)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plugin import
# ---------------------------------------------------------------------------

def import_plugins(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    plugin_dir = getattr(cfg, 'plugin_dir', os.path.dirname(config_path)).rstrip('/')
    module_path = '.'.join(p for p in plugin_dir.split('/') if p)
    importlib.import_module(module_path)


# ---------------------------------------------------------------------------
# Model-type detection
# ---------------------------------------------------------------------------

def detect_model_kind(model):
    """Return a string tag identifying the model pipeline."""
    cls = type(model).__name__
    if cls == 'FLCPointOccNet':
        return 'flc_pointocc'
    if cls == 'PointOccNet':
        return 'pointocc'
    # OccNet family: differentiate by what sub-modules are present
    has_camera = (hasattr(model, 'img_backbone') and model.img_backbone is not None)
    has_lidar_vox = (hasattr(model, 'pts_voxel_encoder')
                     and model.pts_voxel_encoder is not None)
    if has_camera and has_lidar_vox:
        return 'occnet_mm'
    if has_camera:
        return 'occnet_cam'
    return 'occnet_lidar'


# ---------------------------------------------------------------------------
# Dummy input builders
# ---------------------------------------------------------------------------

def build_cam_inputs(cfg, device, batch_size, num_cams_override=None):
    """Build dummy camera tensor list [img, rots, trans, intrins, ...].

    Returns (img_inputs, img_metas).
    """
    if not hasattr(cfg, 'data_config'):
        raise ValueError(
            'Config has no data_config — cannot build camera dummy inputs. '
            'Use a camera-capable config or skip camera inputs.')
    dc = cfg.data_config
    input_size = dc['input_size']
    num_cams = num_cams_override or dc.get('Ncams') or len(dc.get('cams', [])) or 6
    h, w = input_size
    b, n = batch_size, num_cams

    img = torch.randn(b, n, 3, h, w, device=device)
    rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
    trans = torch.zeros(b, n, 3, device=device)
    intrins = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
    post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
    post_trans = torch.zeros(b, n, 3, device=device)
    bda = torch.eye(3, device=device).view(1, 3, 3).repeat(b, 1, 1)
    img_hw = torch.stack([
        torch.full((b,), h, device=device, dtype=torch.float32),
        torch.full((b,), w, device=device, dtype=torch.float32),
    ], dim=0)

    img_inputs = [img, rots, trans, intrins, post_rots, post_trans, bda, img_hw]
    img_metas = [{} for _ in range(b)]
    return img_inputs, img_metas


def build_lidar_points(batch_size, num_points, device):
    """Build a dummy list of raw LiDAR point clouds [x, y, z, intensity, ring]."""
    return [torch.randn(num_points, 5, device=device) for _ in range(batch_size)]


def build_gt_occ(batch_size, occ_grid=(512, 512, 40), device='cuda'):
    """Build a dummy occupancy GT tensor [B, W, H, D] with values in [0, 16]."""
    B = batch_size
    W, H, D = occ_grid
    gt = torch.randint(0, 17, (B, W, H, D), device=device)
    return gt


# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_stage(name, fn, time_meter=None, mem_meter=None):
    sync()
    mem_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    sync()
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()

    if time_meter is not None:
        time_meter[name] = time_meter.get(name, 0.0) + elapsed
    if mem_meter is not None:
        mem_meter[name] = {
            'alloc_before': mem_before,
            'alloc_after': torch.cuda.memory_allocated(),
            'alloc_delta': torch.cuda.memory_allocated() - mem_before,
            'stage_peak_delta': max(0, peak - mem_before),
        }
    return out


# ---------------------------------------------------------------------------
# Per-model-kind profiling functions
# ---------------------------------------------------------------------------

def profile_occnet_cam(model, img_inputs, img_metas, mode,
                       time_meter=None, mem_meter=None):
    """OccNet camera-only (Cam-OO, FlashOcc)."""
    rots, trans, intrins, post_rots, post_trans, bda = img_inputs[1:7]

    enc = timed_stage('img_encoder',
                      lambda: model.image_encoder(img_inputs[0]),
                      time_meter, mem_meter)
    x = enc['x']

    mlp_input = model.img_view_transformer.get_mlp_input(
        rots, trans, intrins, post_rots, post_trans, bda)
    geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]

    x, depth = timed_stage('view_transformer',
                            lambda: model.img_view_transformer([x] + geo_inputs),
                            time_meter, mem_meter)

    voxel_feats = x
    if model.occ_fuser is not None:
        voxel_feats = timed_stage('occ_fuser',
                                  lambda: model.occ_fuser(x, None),
                                  time_meter, mem_meter)

    enc2 = timed_stage('occ_encoder',
                       lambda: model.occ_encoder(voxel_feats),
                       time_meter, mem_meter)
    if not isinstance(enc2, list):
        enc2 = [enc2]

    if mode == 'extract':
        return enc2

    transform = img_inputs[1:8]
    out = timed_stage('occ_head',
                      lambda: model.pts_bbox_head(
                          voxel_feats=enc2, points=None, img_metas=img_metas,
                          img_feats=enc.get('img_feats'), pts_feats=None,
                          transform=transform),
                      time_meter, mem_meter)
    return out


def profile_occnet_lidar(model, points, img_metas, mode,
                         time_meter=None, mem_meter=None):
    """OccNet LiDAR-only (LiDAR-OO)."""
    pts_voxel, pts_feats_list = timed_stage(
        'pts_encoder',
        lambda: model.extract_pts_feat(points),
        time_meter, mem_meter)

    enc = timed_stage('occ_encoder',
                      lambda: model.occ_encoder(pts_voxel),
                      time_meter, mem_meter)
    if not isinstance(enc, list):
        enc = [enc]

    if mode == 'extract':
        return enc

    out = timed_stage('occ_head',
                      lambda: model.pts_bbox_head(
                          voxel_feats=enc, points=None, img_metas=img_metas,
                          img_feats=None, pts_feats=pts_feats_list,
                          transform=None),
                      time_meter, mem_meter)
    return out


def profile_occnet_mm(model, img_inputs, img_metas, points, mode,
                      time_meter=None, mem_meter=None):
    """OccNet multimodal (MM-OO)."""
    rots, trans, intrins, post_rots, post_trans, bda = img_inputs[1:7]

    # Camera branch — two stages
    enc_img = timed_stage('img_encoder',
                          lambda: model.image_encoder(img_inputs[0]),
                          time_meter, mem_meter)
    x_img = enc_img['x']
    img_feats = enc_img.get('img_feats')

    mlp_input = model.img_view_transformer.get_mlp_input(
        rots, trans, intrins, post_rots, post_trans, bda)
    geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
    img_voxel, depth = timed_stage('view_transformer',
                                   lambda: model.img_view_transformer([x_img] + geo_inputs),
                                   time_meter, mem_meter)

    # LiDAR branch
    pts_voxel, pts_feats_list = timed_stage('pts_encoder',
                                            lambda: model.extract_pts_feat(points),
                                            time_meter, mem_meter)

    # Fusion
    voxel_feats = timed_stage('occ_fuser',
                              lambda: model.occ_fuser(img_voxel, pts_voxel),
                              time_meter, mem_meter)

    enc = timed_stage('occ_encoder',
                      lambda: model.occ_encoder(voxel_feats),
                      time_meter, mem_meter)
    if not isinstance(enc, list):
        enc = [enc]

    if mode == 'extract':
        return enc

    transform = img_inputs[1:8]
    out = timed_stage('occ_head',
                      lambda: model.pts_bbox_head(
                          voxel_feats=enc, points=None, img_metas=img_metas,
                          img_feats=img_feats, pts_feats=pts_feats_list,
                          transform=transform),
                      time_meter, mem_meter)
    return out


def profile_pointocc(model, points, img_metas,
                     time_meter=None, mem_meter=None):
    """PointOccNet — LiDAR-only TPV.

    extract_feat returns logits directly; no separate head call.
    We sub-divide into two stages: TPV feature extraction and aggregator.
    """
    # Pre-compute tpv_norm_shape (same as model.extract_feat does internally)
    agg = model.tpv_aggregator
    tpv_norm = [int(agg.tpv_w * agg.scale_w),
                int(agg.tpv_h * agg.scale_h),
                int(agg.tpv_z * agg.scale_z)]

    # Stage 1: LiDAR preprocessing + CylinderEncoder + Swin + FPN
    def _lidar_tpv():
        grid_ind, voxels_coarse = model._prepare_lidar_inputs(points, tpv_norm)
        tpv_list = model.extract_lidar_tpv(grid_ind)
        return grid_ind, voxels_coarse, tpv_list

    result = timed_stage('lidar_tpv', _lidar_tpv, time_meter, mem_meter)
    grid_ind, voxels_coarse, tpv_list = result

    # Stage 2: TPV aggregator (dense grid_sample + classifier)
    logits = timed_stage('tpv_aggregator',
                         lambda: model.tpv_aggregator(tpv_list, voxels_coarse),
                         time_meter, mem_meter)
    return logits


def profile_flc_pointocc(model, img_inputs, img_metas, points, mode,
                          time_meter=None, mem_meter=None):
    """FLCPointOccNet — dual-modal camera + LiDAR TPV fusion."""
    rots, trans, intrins, post_rots, post_trans, bda = img_inputs[1:7]

    # Camera encoding
    enc_img = timed_stage('img_encoder',
                          lambda: model.image_encoder(img_inputs[0]),
                          time_meter, mem_meter)
    x_img = enc_img['x']
    img_feats = enc_img.get('img_feats')

    mlp_input = model.img_view_transformer.get_mlp_input(
        rots, trans, intrins, post_rots, post_trans, bda)
    geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
    cam_bev, depth = timed_stage('view_transformer',
                                 lambda: model.img_view_transformer([x_img] + geo_inputs),
                                 time_meter, mem_meter)

    # Camera adapter (Version A only; Version B has cam_adapter=None)
    if model.cam_adapter is not None:
        cam_feat = timed_stage('cam_adapter',
                               lambda: model.cam_adapter(cam_bev),
                               time_meter, mem_meter)
    else:
        cam_feat = cam_bev

    # LiDAR TPV branch
    tpv_norm = [int(model.tpv_fuser.tpv_w * model.tpv_fuser.scale_w),
                int(model.tpv_fuser.tpv_h * model.tpv_fuser.scale_h),
                int(model.tpv_fuser.tpv_z * model.tpv_fuser.scale_z)]

    def _lidar_tpv():
        grid_ind, voxels_coarse = model._prepare_lidar_inputs(points, tpv_norm)
        tpv_list = model.extract_lidar_tpv(grid_ind)
        lidar_3d = model.tpv_fuser(tpv_list, voxels_coarse)
        return lidar_3d

    lidar_3d = timed_stage('lidar_tpv', _lidar_tpv, time_meter, mem_meter)

    # Flatten Z and LiDAR adapter
    def _lidar_adapter():
        B, C, X, Y, Z = lidar_3d.shape
        bev = lidar_3d.permute(0, 1, 4, 2, 3).contiguous()
        bev = bev.reshape(B, C * Z, X, Y)
        return model.lidar_adapter(bev)

    lidar_feat = timed_stage('lidar_adapter', _lidar_adapter, time_meter, mem_meter)

    # BEV fusion
    fused = timed_stage('fusion',
                        lambda: model.fuse_conv(
                            torch.cat([cam_feat, lidar_feat], dim=1)),
                        time_meter, mem_meter)

    enc = timed_stage('occ_encoder',
                      lambda: model.occ_encoder(fused),
                      time_meter, mem_meter)
    if not isinstance(enc, list):
        enc = [enc]

    if mode == 'extract':
        return enc

    transform = img_inputs[1:8]
    out = timed_stage('occ_head',
                      lambda: model.pts_bbox_head(
                          voxel_feats=enc, points=None, img_metas=img_metas,
                          img_feats=img_feats, pts_feats=None,
                          transform=transform),
                      time_meter, mem_meter)
    return out


# ---------------------------------------------------------------------------
# Training memory measurement
# ---------------------------------------------------------------------------

def measure_train_memory(model, model_kind, cfg, device, batch_size, num_points):
    """Run one forward+backward pass and return peak GPU memory in bytes."""
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    img_metas = [{} for _ in range(batch_size)]
    occ_gt = build_gt_occ(batch_size, occ_grid=(512, 512, 40), device=device)

    if model_kind in ('occnet_cam', 'occnet_lidar', 'occnet_mm'):
        img_inputs = None
        points = None

        if model_kind in ('occnet_cam', 'occnet_mm'):
            img_inputs, _ = build_cam_inputs(cfg, device, batch_size)
        if model_kind in ('occnet_lidar', 'occnet_mm'):
            points = build_lidar_points(batch_size, num_points, device)

        loss_dict = model.forward_train(
            points=points,
            img_metas=img_metas,
            img_inputs=img_inputs,
            gt_occ=occ_gt,
        )

    elif model_kind == 'pointocc':
        points = build_lidar_points(batch_size, num_points, device)
        loss_dict = model.forward_train(
            points=points,
            img_metas=img_metas,
            gt_occ=occ_gt,
        )

    elif model_kind == 'flc_pointocc':
        img_inputs, _ = build_cam_inputs(cfg, device, batch_size)
        points = build_lidar_points(batch_size, num_points, device)
        loss_dict = model.forward_train(
            points=points,
            img_metas=img_metas,
            img_inputs=img_inputs,
            gt_occ=occ_gt,
        )

    total_loss = sum(v for v in loss_dict.values()
                     if isinstance(v, torch.Tensor) and v.requires_grad)
    if isinstance(total_loss, torch.Tensor):
        total_loss.backward()
    sync()
    return torch.cuda.max_memory_allocated()


# ---------------------------------------------------------------------------
# FLOPs estimation
# ---------------------------------------------------------------------------

def estimate_flops(model, model_kind, cfg, device, batch_size, num_points):
    """Try fvcore then thop. Returns GFLOPs or None on failure."""
    # Build inputs for fvcore (expects tuple/single arg)
    try:
        from fvcore.nn import FlopCountAnalysis

        img_metas = [{} for _ in range(batch_size)]

        if model_kind == 'occnet_cam':
            img_inputs, _ = build_cam_inputs(cfg, device, batch_size)
            flops = FlopCountAnalysis(model.forward_dummy,
                                      (None, img_metas, img_inputs))
        elif model_kind == 'occnet_lidar':
            points = build_lidar_points(batch_size, num_points, device)
            flops = FlopCountAnalysis(model.forward_dummy,
                                      (points, img_metas, None))
        elif model_kind == 'occnet_mm':
            img_inputs, _ = build_cam_inputs(cfg, device, batch_size)
            points = build_lidar_points(batch_size, num_points, device)
            flops = FlopCountAnalysis(model.forward_dummy,
                                      (points, img_metas, img_inputs))
        elif model_kind == 'pointocc':
            points = build_lidar_points(batch_size, num_points, device)
            flops = FlopCountAnalysis(model.extract_feat,
                                      (points, img_metas))
        elif model_kind == 'flc_pointocc':
            img_inputs, _ = build_cam_inputs(cfg, device, batch_size)
            points = build_lidar_points(batch_size, num_points, device)
            flops = FlopCountAnalysis(model.forward_dummy,
                                      (points, img_metas, img_inputs))
        else:
            return None

        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = flops.total()
        return total / 1e9

    except Exception as e:
        print(f'  [fvcore failed: {e}]')

    # Fallback: thop
    try:
        from thop import profile as thop_profile

        if model_kind == 'occnet_cam':
            img_inputs, img_metas2 = build_cam_inputs(cfg, device, batch_size)
            macs, _ = thop_profile(model, inputs=(None, img_metas2, img_inputs),
                                   verbose=False)
        elif model_kind == 'pointocc':
            points = build_lidar_points(batch_size, num_points, device)
            img_metas2 = [{} for _ in range(batch_size)]
            macs, _ = thop_profile(model.extract_feat,
                                   inputs=(points, img_metas2), verbose=False)
        else:
            return None
        return macs / 1e9  # thop returns MACs, report as GFLOPs (≈ 2× MACs, but conventional)

    except Exception as e:
        print(f'  [thop failed: {e}]')

    return None


# ---------------------------------------------------------------------------
# Dispatch: run one profiling iteration
# ---------------------------------------------------------------------------

def run_one(model_kind, model, cfg, device, batch_size, num_cams, num_points,
            mode, time_meter=None, mem_meter=None):
    img_metas = [{} for _ in range(batch_size)]

    if model_kind == 'occnet_cam':
        img_inputs, img_metas = build_cam_inputs(cfg, device, batch_size, num_cams)
        return profile_occnet_cam(model, img_inputs, img_metas, mode, time_meter, mem_meter)

    elif model_kind == 'occnet_lidar':
        points = build_lidar_points(batch_size, num_points, device)
        return profile_occnet_lidar(model, points, img_metas, mode, time_meter, mem_meter)

    elif model_kind == 'occnet_mm':
        img_inputs, img_metas = build_cam_inputs(cfg, device, batch_size, num_cams)
        points = build_lidar_points(batch_size, num_points, device)
        return profile_occnet_mm(model, img_inputs, img_metas, points, mode, time_meter, mem_meter)

    elif model_kind == 'pointocc':
        points = build_lidar_points(batch_size, num_points, device)
        return profile_pointocc(model, points, img_metas, time_meter, mem_meter)

    elif model_kind == 'flc_pointocc':
        img_inputs, img_metas = build_cam_inputs(cfg, device, batch_size, num_cams)
        points = build_lidar_points(batch_size, num_points, device)
        return profile_flc_pointocc(model, img_inputs, img_metas, points, mode, time_meter, mem_meter)

    else:
        raise ValueError(f'Unknown model kind: {model_kind}')


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_bytes(n):
    if n < 0:
        return f'-{fmt_bytes(-n)}'
    if n < 1024 ** 2:
        return f'{n / 1024:.1f} KiB'
    if n < 1024 ** 3:
        return f'{n / 1024 ** 2:.1f} MiB'
    return f'{n / 1024 ** 3:.2f} GiB'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    device = torch.device('cuda')
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    if args.checkpoint:
        ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        ckpt = None
        print('[WARNING] No checkpoint provided — using random initialisation.')

    model.to(device)
    model_kind = detect_model_kind(model)
    print(f'\nDetected model kind : {model_kind}  ({type(model).__name__})')

    # ---- Parameter count (always) ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params        : {total_params / 1e6:.2f} M')
    print(f'Trainable params    : {trainable_params / 1e6:.2f} M')

    if args.params:
        return  # params-only mode

    # ---- FLOPs ----
    if args.flops:
        print('\nEstimating FLOPs (this may take a moment)...')
        model.eval()
        gflops = estimate_flops(model, model_kind, cfg, device,
                                batch_size=1, num_points=args.num_points)
        if gflops is not None:
            print(f'GFLOPs (batch=1)    : {gflops:.1f}')
        else:
            print('GFLOPs              : N/A (unsupported ops in LiDAR encoder)')

    # ---- Training memory ----
    if args.train_mem:
        print(f'\nMeasuring training memory (batch={args.batch_size})...')
        try:
            peak = measure_train_memory(model, model_kind, cfg, device,
                                        args.batch_size, args.num_points)
            print(f'Train peak memory   : {fmt_bytes(peak)}')
        except Exception as e:
            print(f'Train memory measurement failed: {e}')
        model.to(device).eval()

    # ---- Inference profiling ----
    model.eval()
    # Disable internal record_time to avoid interference
    if hasattr(model, 'record_time'):
        model.record_time = False

    print(f'\nProfiling inference ({args.warmup} warmup + {args.measure} measure iters, batch={args.batch_size})...')

    with torch.no_grad():
        for _ in range(args.warmup):
            run_one(model_kind, model, cfg, device,
                    args.batch_size, args.num_cams, args.num_points, args.mode)

        time_meter = OrderedDict()
        total_times = []
        peak_mems = []

        for _ in range(args.measure):
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            run_one(model_kind, model, cfg, device,
                    args.batch_size, args.num_cams, args.num_points, args.mode,
                    time_meter=time_meter)
            sync()
            total_times.append(time.perf_counter() - t0)
            peak_mems.append(torch.cuda.max_memory_allocated())

        # One final run for stage-level memory breakdown
        mem_meter = OrderedDict()
        torch.cuda.reset_peak_memory_stats()
        run_one(model_kind, model, cfg, device,
                args.batch_size, args.num_cams, args.num_points, args.mode,
                mem_meter=mem_meter)

    total_avg = sum(total_times) / len(total_times)
    fps = 1.0 / total_avg
    peak_infer = max(peak_mems)

    # ---- Print results ----
    print(f"\n{'=' * 72}")
    print('Forward Inference Profiling')
    print(f'  config      : {args.config}')
    if args.checkpoint:
        print(f'  checkpoint  : {args.checkpoint}')
    print(f'  model kind  : {model_kind}')
    print(f'  mode        : {args.mode}')
    print(f'  batch size  : {args.batch_size}')
    print(f'  avg forward : {total_avg * 1000:.2f} ms')
    print(f'  FPS         : {fps:.2f}')
    print(f'  Infer peak  : {fmt_bytes(peak_infer)}')
    print(f"{'=' * 72}")

    print(f"\n{'Stage':<22} {'Avg Time (ms)':>14} {'Share':>8}")
    print('-' * 48)
    total_stage = sum(time_meter.values())
    for name, val in time_meter.items():
        avg = val / args.measure
        share = val / total_stage * 100.0 if total_stage > 0 else 0.0
        print(f'{name:<22} {avg * 1000:>13.2f} {share:>7.1f}%')

    print(f"\n{'Stage':<22} {'Alloc Δ':>12} {'Stage Peak Δ':>14} {'After':>12}")
    print('-' * 66)
    for name, s in mem_meter.items():
        print(f"{name:<22} "
              f"{fmt_bytes(s['alloc_delta']):>12} "
              f"{fmt_bytes(s['stage_peak_delta']):>14} "
              f"{fmt_bytes(s['alloc_after']):>12}")

    if ckpt and isinstance(ckpt, dict) and 'meta' in ckpt and ckpt['meta']:
        epoch = ckpt['meta'].get('epoch')
        if epoch is not None:
            print(f'\nLoaded checkpoint epoch: {epoch}')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this profiler.')
    main()
