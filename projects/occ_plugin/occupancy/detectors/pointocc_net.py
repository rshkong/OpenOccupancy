"""PointOccNet: LiDAR-only occupancy detector reproducing PointOcc.

Pipeline
--------
    points → _prepare_lidar_inputs → (10ch, grid_ind, voxels_coarse)
        → CylinderEncoder → TPVSwin → TPVFPN → tpv_list
        → TPVAggregator → logits [B, num_cls, W_c, H_c, D_c]
        → CE + lovasz_softmax + sem_scal + geo_scal  (weights [1,1,1,1])

Loss block mirrors PointOcc's `TPVAggregator_Occ.forward(..., return_loss=True)`
but lives in the detector so OpenOccupancy training hooks (`parse_losses`) work.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import DETECTORS
from mmdet.models.detectors import BaseDetector
from mmdet3d.models import builder
from mmcv.runner import auto_fp16

from .lidar_prep_mixin import LidarPrepMixin
from ...utils.lovasz_losses import lovasz_softmax
from ...utils.sem_geo_loss import sem_scal_loss, geo_scal_loss


def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(
        max_label * label.astype(int) + pred,
        minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)


@DETECTORS.register_module()
class PointOccNet(LidarPrepMixin, BaseDetector):
    """LiDAR-only occupancy detector — PointOcc reproduction."""

    def __init__(self,
                 lidar_tokenizer=None,
                 lidar_backbone=None,
                 lidar_neck=None,
                 tpv_aggregator=None,
                 cyl_grid_size=None,
                 cyl_min_bound=None,
                 cyl_max_bound=None,
                 occ_grid_size=None,
                 occ_coarse_ratio=1,
                 pc_range=None,
                 loss_weight=(1.0, 1.0, 1.0, 1.0),
                 ignore_index=255,
                 empty_idx=0,
                 num_classes=17,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        self.lidar_tokenizer = builder.build_backbone(lidar_tokenizer)
        self.lidar_backbone = builder.build_backbone(lidar_backbone)
        self.lidar_neck = builder.build_neck(lidar_neck)
        self.tpv_aggregator = builder.build_fusion_layer(tpv_aggregator)

        self.cyl_grid_size = np.array(cyl_grid_size) if cyl_grid_size is not None \
            else np.array([480, 360, 32])
        self.cyl_min_bound = np.array(cyl_min_bound) if cyl_min_bound is not None \
            else np.array([0.0, -np.pi, -5.0])
        self.cyl_max_bound = np.array(cyl_max_bound) if cyl_max_bound is not None \
            else np.array([50.0, np.pi, 3.0])
        self.occ_grid_size = np.array(occ_grid_size) if occ_grid_size is not None \
            else np.array([128, 128, 10])
        self.occ_coarse_ratio = occ_coarse_ratio
        if pc_range is not None:
            self.pc_range = np.array(pc_range)

        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.empty_idx = empty_idx
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _tpv_norm_shape(self):
        """Target voxels_coarse range derived from aggregator/fuser config."""
        agg = self.tpv_aggregator
        return [
            int(agg.tpv_w * agg.scale_w),
            int(agg.tpv_h * agg.scale_h),
            int(agg.tpv_z * agg.scale_z),
        ]

    def extract_feat(self, points, img_metas=None):
        """Return coarse-resolution logits [B, num_classes, W_c, H_c, D_c]."""
        tpv_norm = self._tpv_norm_shape()
        grid_ind, voxels_coarse = self._prepare_lidar_inputs(
            points, tpv_norm_shape=tpv_norm)
        tpv_list = self.extract_lidar_tpv(grid_ind)
        logits = self.tpv_aggregator(tpv_list, voxels_coarse)
        return logits

    # BaseDetector abstract stubs
    def extract_feats(self, points, img_metas=None):
        return [self.extract_feat(p, m) for p, m in zip(points, img_metas)]

    def aug_test(self, points, img_metas, **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _coarsen_gt(self, gt_occ, target_shape):
        """Downsample gt_occ (B, W, H, D) to the coarse resolution (mode-pool).

        Mirrors PointOcc's voxel_label_coarse construction: any sub-block
        containing only empty cells stays empty; otherwise take the mode of
        the non-empty labels.
        """
        B, W_full, H_full, D_full = gt_occ.shape
        W, H, D = target_shape
        ratio_w = W_full // W
        ratio_h = H_full // H
        ratio_d = D_full // D
        if ratio_w == 1 and ratio_h == 1 and ratio_d == 1:
            return gt_occ.long()
        assert ratio_w == ratio_h == ratio_d, \
            "Non-uniform coarsening ratio is not supported"
        ratio = ratio_w

        coarse = gt_occ.reshape(
            B, W, ratio, H, ratio, D, ratio
        ).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, W, H, D, ratio ** 3)
        empty_mask = coarse.sum(-1) == 0
        coarse = coarse.to(torch.int64)
        occ_space = coarse[~empty_mask]
        occ_space[occ_space == 0] = -torch.arange(
            len(occ_space[occ_space == 0]), device=occ_space.device) - 1
        coarse[~empty_mask] = occ_space
        coarse = torch.mode(coarse, dim=-1)[0]
        coarse[coarse < 0] = self.ignore_index
        return coarse.long()

    def _compute_loss(self, logits, gt_occ):
        B, C, W, H, D = logits.shape
        label_coarse = self._coarsen_gt(gt_occ, (W, H, D))
        w = self.loss_weight
        with torch.cuda.amp.autocast(enabled=False):
            logits = logits.float()
            loss_ce = self.ce_loss(logits, label_coarse)
            loss_lovasz = lovasz_softmax(
                torch.softmax(logits, dim=1), label_coarse,
                ignore=self.ignore_index)
            loss_sem = sem_scal_loss(
                logits, label_coarse, ignore_index=self.ignore_index)
            loss_geo = geo_scal_loss(
                logits, label_coarse, ignore_index=self.ignore_index,
                non_empty_idx=self.empty_idx)
        return {
            'loss_ce': w[0] * loss_ce,
            'loss_lovasz': w[1] * loss_lovasz,
            'loss_sem_scal': w[2] * loss_sem,
            'loss_geo_scal': w[3] * loss_geo,
        }

    # ------------------------------------------------------------------
    # Training / testing entry points
    # ------------------------------------------------------------------
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_occ=None,
                      **kwargs):
        logits = self.extract_feat(points, img_metas)
        return self._compute_loss(logits, gt_occ)

    @auto_fp16(apply_to=('points',))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     gt_occ=None,
                     visible_mask=None,
                     **kwargs):
        return self.simple_test(
            points=points, img_metas=img_metas, gt_occ=gt_occ,
            visible_mask=visible_mask, **kwargs)

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    gt_occ=None,
                    visible_mask=None,
                    **kwargs):
        logits = self.extract_feat(points, img_metas)  # (B, C, W, H, D)
        pred_c = logits

        SC_metric, _ = self._evaluate(
            pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, _ = self._evaluate(
            pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        return {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': None,
        }

    def _evaluate(self, pred, gt, eval_type, visible_mask=None):
        """Same logic as OccNet.evaluation_semantic."""
        _, H, W, D = gt.shape
        pred = F.interpolate(
            pred, size=[H, W, D], mode='trilinear',
            align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy().astype(np.int)
        noise_mask = gt != self.ignore_index

        if eval_type == 'SC':
            gt = gt.copy()
            pred = pred.copy()
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None

        hist_occ = None
        if visible_mask is not None:
            visible_mask = visible_mask[0].cpu().numpy()
            mask = noise_mask & (visible_mask != 0)
            hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)
        hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
        return hist, hist_occ
