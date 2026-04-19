import argparse
import importlib
import os
import time
from collections import OrderedDict

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile pure forward inference of OpenOccupancy models')
    parser.add_argument(
        '--config',
        default='projects/configs/baselines/CAM-R50_img256x704_flc_step2_128x128x10_4070ti.py',
        help='config file path')
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/CAM-R50_img256x704_flc_step2_128x128x10_4070ti/latest.pth',
        help='checkpoint file path')
    parser.add_argument(
        '--mode',
        choices=['extract', 'full'],
        default='full',
        help='extract: profile until occ encoder; full: profile until occ head')
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='number of warmup iterations')
    parser.add_argument(
        '--measure',
        type=int,
        default=50,
        help='number of timed iterations')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='batch size for dummy input')
    parser.add_argument(
        '--num-cams',
        type=int,
        default=None,
        help='override number of cameras from config')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed for dummy input')
    return parser.parse_args()


def import_plugins(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir.rstrip('/')   # 'projects/occ_plugin/' → 'projects/occ_plugin'
    else:
        plugin_dir = os.path.dirname(config_path)
    module_path = '.'.join([p for p in plugin_dir.split('/') if p])
    importlib.import_module(module_path)


def build_dummy_input(cfg, device, batch_size, num_cams=None):
    if not hasattr(cfg, 'data_config'):
        raise ValueError('This profiler expects config.data_config to exist.')

    input_size = cfg.data_config['input_size']
    num_cams = num_cams or cfg.data_config.get('Ncams', len(cfg.data_config.get('cams', [])))
    if not num_cams:
        raise ValueError('Failed to infer number of cameras from config.')

    h_img, w_img = input_size
    b = batch_size
    n = num_cams

    img = torch.randn(b, n, 3, h_img, w_img, device=device)
    rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
    trans = torch.zeros(b, n, 3, device=device)
    intrins = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
    post_rots = torch.eye(3, device=device).view(1, 1, 3, 3).repeat(b, n, 1, 1)
    post_trans = torch.zeros(b, n, 3, device=device)
    bda = torch.eye(3, device=device).view(1, 3, 3).repeat(b, 1, 1)

    # OccHead fine image sampling expects transform[6][0] = H and [6][1] = W.
    img_hw = torch.stack([
        torch.full((b,), h_img, device=device, dtype=torch.float32),
        torch.full((b,), w_img, device=device, dtype=torch.float32),
    ], dim=0)

    img_inputs = [img, rots, trans, intrins, post_rots, post_trans, bda, img_hw]
    img_metas = [{} for _ in range(b)]
    return img_inputs, img_metas


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
    mem_after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()

    if time_meter is not None:
        time_meter[name] = time_meter.get(name, 0.0) + elapsed
    if mem_meter is not None:
        mem_meter[name] = {
            'alloc_before': mem_before,
            'alloc_after': mem_after,
            'alloc_delta': mem_after - mem_before,
            'stage_peak_abs': peak,
            'stage_peak_delta': max(0, peak - mem_before),
        }
    return out


def run_profile_once(model, img_inputs, img_metas, mode, time_meter=None, mem_meter=None):
    rots, trans, intrins, post_rots, post_trans, bda = img_inputs[1:7]

    img_enc_feats = timed_stage(
        'img_encoder',
        lambda: model.image_encoder(img_inputs[0]),
        time_meter,
        mem_meter)
    x = img_enc_feats['x']
    img_feats = img_enc_feats['img_feats']

    mlp_input = model.img_view_transformer.get_mlp_input(
        rots, trans, intrins, post_rots, post_trans, bda)
    geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]

    x, depth = timed_stage(
        'view_transformer',
        lambda: model.img_view_transformer([x] + geo_inputs),
        time_meter,
        mem_meter)

    if model.occ_fuser is not None:
        voxel_feats = timed_stage(
            'occ_fuser',
            lambda: model.occ_fuser(x, None),
            time_meter,
            mem_meter)
    else:
        voxel_feats = x

    voxel_feats_enc = timed_stage(
        'occ_encoder',
        lambda: model.occ_encoder(voxel_feats),
        time_meter,
        mem_meter)
    if not isinstance(voxel_feats_enc, list):
        voxel_feats_enc = [voxel_feats_enc]

    if mode == 'extract':
        return {
            'voxel_feats': voxel_feats_enc,
            'img_feats': img_feats,
            'depth': depth,
        }

    transform = img_inputs[1:8]
    head_output = timed_stage(
        'occ_head',
        lambda: model.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            points=None,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=None,
            transform=transform,
        ),
        time_meter,
        mem_meter)
    return head_output


def pretty_size(num_bytes):
    return f'{num_bytes / 1024**3:.2f} GB'


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    model.record_time = False

    img_inputs, img_metas = build_dummy_input(
        cfg, device, args.batch_size, args.num_cams)

    with torch.no_grad():
        for _ in range(args.warmup):
            run_profile_once(model, img_inputs, img_metas, args.mode)

        time_meter = OrderedDict()
        total_times = []
        peak_memories = []

        for _ in range(args.measure):
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            run_profile_once(model, img_inputs, img_metas, args.mode, time_meter=time_meter)
            sync()
            total_times.append(time.perf_counter() - t0)
            peak_memories.append(torch.cuda.max_memory_allocated())

        mem_meter = OrderedDict()
        torch.cuda.reset_peak_memory_stats()
        run_profile_once(model, img_inputs, img_metas, args.mode, mem_meter=mem_meter)

    total_avg = sum(total_times) / len(total_times)
    fps = 1.0 / total_avg
    max_peak_memory = max(peak_memories)

    print(f"\n{'=' * 72}")
    print('Pure Forward Profiling (image -> occupancy logits)')
    print(f'config      : {args.config}')
    print(f'checkpoint  : {args.checkpoint}')
    print(f'mode        : {args.mode}')
    print(f'input shape : img={tuple(img_inputs[0].shape)}')
    print(f'warmup      : {args.warmup}')
    print(f'measure     : {args.measure}')
    print(f'avg forward : {total_avg * 1000:.2f} ms')
    print(f'fps         : {fps:.2f}')
    print(f'peak memory : {pretty_size(max_peak_memory)}')
    print(f"{'=' * 72}")

    print(f"\n{'Stage':<20} {'Avg Time (ms)':>14} {'Share':>9}")
    print('-' * 48)
    total_stage_time = sum(time_meter.values())
    for name, total in time_meter.items():
        avg = total / args.measure
        share = (total / total_stage_time * 100.0) if total_stage_time > 0 else 0.0
        print(f'{name:<20} {avg * 1000:>13.2f} {share:>8.1f}%')

    print(f"\n{'Stage':<20} {'Alloc Delta':>14} {'Stage Peak Delta':>18} {'Alloc After':>14}")
    print('-' * 72)
    for name, stats in mem_meter.items():
        print(
            f"{name:<20} "
            f"{pretty_size(stats['alloc_delta']):>14} "
            f"{pretty_size(stats['stage_peak_delta']):>18} "
            f"{pretty_size(stats['alloc_after']):>14}"
        )

    if checkpoint and 'meta' in checkpoint and checkpoint['meta'] is not None:
        epoch = checkpoint['meta'].get('epoch', None)
        if epoch is not None:
            print(f'\nLoaded checkpoint epoch: {epoch}')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this profiler.')
    main()
