import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models import HEADS
from .occ_head import OccHead


@HEADS.register_module()
class FLCOccHead(OccHead):
    """FLC-Occ prediction head with Conv2d + C2H MLP (matches FlashOcc BEVOCCHead2D).

    Pipeline (per FlashOcc BEVOCCHead2D):
        1. Conv2d (k=3, p=1) — spatial context aggregation across neighbouring pillars
        2. Linear → Softplus → Linear (per-pillar MLP) — channel-to-height projection

    Input to forward_coarse_voxel:
        bev_feat: [B, C_bev, X, Y]   e.g. [B, 256, 128, 128]

    Output coarse_occ:
        [B, num_cls, X, Y, Dz]       e.g. [B, 17, 128, 128, 10]
        — fully compatible with parent's loss_voxel / loss_point / fine branch.

    Args:
        in_channels (int): BEV feature channels from FPN_LSS output (e.g. 256).
        out_channel  (int): Number of semantic classes (e.g. 17).
        Dz           (int): Number of height bins in the coarse grid (e.g. 10).
        conv_out_dim (int): Output channels of the Conv2d layer (e.g. 256).
        hidden_dim   (int): Hidden size of the C2H MLP (e.g. 512).
        norm_cfg_2d  (dict): norm_cfg for the Conv2d ConvModule.
        All other kwargs are forwarded to OccHead.__init__ unchanged.
    """

    def __init__(
        self,
        in_channels,
        out_channel,
        Dz=10,
        conv_out_dim=256,
        hidden_dim=512,
        norm_cfg_2d=dict(type='BN'),
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channel=out_channel,
            **kwargs,
        )

        # OccHead.__init__ builds Conv3d layers (occ_convs, occ_pred_conv) that
        # FLCOccHead never uses. Replace them with empty/identity placeholders so
        # DDP does not complain about parameters that never receive gradients.
        self.occ_convs = nn.ModuleList()      # was: list of Conv3d blocks
        self.occ_pred_conv = nn.Sequential()  # was: two Conv3d layers

        self.Dz = Dz
        self.bev_in_channels = in_channels if not isinstance(in_channels, list) \
            else in_channels[0]

        # Step 1: Conv2d with spatial receptive field (k=3, same padding)
        # Mirrors FlashOcc BEVOCCHead2D.final_conv
        self.final_conv = ConvModule(
            self.bev_in_channels,
            conv_out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg_2d,
        )

        # Step 2: per-pillar MLP: conv_out_dim → hidden → Dz * num_cls
        # Mirrors FlashOcc BEVOCCHead2D.predicter
        self.c2h_predicter = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, Dz * out_channel),
        )

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def forward_coarse_voxel(self, voxel_feats):
        """Conv2d + C2H MLP coarse prediction (matches FlashOcc BEVOCCHead2D).

        Args:
            voxel_feats (list[Tensor] | Tensor): [B, C_bev, X, Y]

        Returns:
            dict:
                'occ'            : list[ [B, num_cls, X, Y, Dz] ]
                'out_voxel_feats': list[ None ]
        """
        if isinstance(voxel_feats, (list, tuple)):
            bev_feat = voxel_feats[0]
        else:
            bev_feat = voxel_feats

        # Step 1: Conv2d — aggregate spatial context  [B, C, X, Y] → [B, conv_out_dim, X, Y]
        x = self.final_conv(bev_feat)

        B, C, X, Y = x.shape

        # Step 2: per-pillar MLP (same as FlashOcc predicter)
        # [B, C, X, Y] → [B, X, Y, C] → [B*X*Y, C]
        x = x.permute(0, 2, 3, 1).contiguous().view(B * X * Y, C)
        # [B*X*Y, C] → [B*X*Y, Dz * num_cls]
        x = self.c2h_predicter(x)
        # → [B, X, Y, Dz, num_cls] → [B, num_cls, X, Y, Dz]
        x = x.view(B, X, Y, self.Dz, self.out_channel)
        coarse_occ = x.permute(0, 4, 1, 2, 3).contiguous()

        return {
            'occ': [coarse_occ],
            'out_voxel_feats': [None],
        }

    # ------------------------------------------------------------------
    # Override forward() to fix the assert and None-indexing in parent
    # ------------------------------------------------------------------

    def forward(self, voxel_feats, img_feats=None, pts_feats=None,
                transform=None, **kwargs):
        """Forward pass.

        Accepts voxel_feats as either:
          - list[ [B, C, X, Y] ]   (standard call from OccNet)
          - [B, C, X, Y]           (direct call)
        """
        # Normalise to list so forward_coarse_voxel handles both cases
        if not isinstance(voxel_feats, (list, tuple)):
            voxel_feats = [voxel_feats]

        # --- Coarse prediction via C2H ---
        output = self.forward_coarse_voxel(voxel_feats)

        out_voxel_feats = output['out_voxel_feats'][0]   # None in Step 2
        coarse_occ      = output['occ'][0]               # [B, 17, X, Y, Dz]

        # --- Fine branch (Step 4, disabled when upsample_ratio==1 or
        #     sample_from_voxel/img are both False) ---
        if self.cascade_ratio != 1:
            if self.sample_from_img or self.sample_from_voxel:
                # sample_from_voxel requires out_voxel_feats to be a real
                # tensor; guard against accidental misconfiguration.
                if self.sample_from_voxel and out_voxel_feats is None:
                    raise ValueError(
                        'sample_from_voxel=True but FLCOccHead returned '
                        'out_voxel_feats=None. Either set '
                        'sample_from_voxel=False or upgrade to Step 3+ '
                        'which provides dense voxel features.'
                    )

                import torch.nn.functional as F
                from projects.occ_plugin.utils import (
                    coarse_to_fine_coordinates, project_points_on_img)

                coarse_occ_mask = coarse_occ.argmax(1) != self.empty_idx
                assert coarse_occ_mask.sum() > 0, 'no foreground in coarse voxel'
                B, W, H, D = coarse_occ_mask.shape
                coarse_coord_x, coarse_coord_y, coarse_coord_z = \
                    torch.meshgrid(
                        torch.arange(W).to(coarse_occ.device),
                        torch.arange(H).to(coarse_occ.device),
                        torch.arange(D).to(coarse_occ.device),
                        indexing='ij')

                output['fine_output'] = []
                output['fine_coord']  = []

                if self.sample_from_img and img_feats is not None:
                    img_feats_ = img_feats[0]
                    B_i, N_i, C_i, W_i, H_i = img_feats_.shape
                    img_feats_ = img_feats_.reshape(-1, C_i, W_i, H_i)
                    img_feats = [
                        self.img_mlp_0(img_feats_).reshape(B_i, N_i, -1, W_i, H_i)
                    ]

                for b in range(B):
                    append_feats = []
                    this_coarse_coord = torch.stack([
                        coarse_coord_x[coarse_occ_mask[b]],
                        coarse_coord_y[coarse_occ_mask[b]],
                        coarse_coord_z[coarse_occ_mask[b]],
                    ], dim=0)   # [3, N]

                    if self.training:
                        this_fine_coord = coarse_to_fine_coordinates(
                            this_coarse_coord, self.cascade_ratio,
                            topk=self.fine_topk)
                    else:
                        this_fine_coord = coarse_to_fine_coordinates(
                            this_coarse_coord, self.cascade_ratio)

                    output['fine_coord'].append(this_fine_coord)
                    new_coord = this_fine_coord[None].permute(0, 2, 1).float().contiguous()

                    if self.sample_from_voxel:
                        fc = this_fine_coord.float()
                        fc[0, :] = (fc[0, :] / (self.final_occ_size[0] - 1) - 0.5) * 2
                        fc[1, :] = (fc[1, :] / (self.final_occ_size[1] - 1) - 0.5) * 2
                        fc[2, :] = (fc[2, :] / (self.final_occ_size[2] - 1) - 0.5) * 2
                        fc = fc[None, None, None].permute(0, 4, 1, 2, 3).float()
                        new_feat = F.grid_sample(
                            out_voxel_feats[b:b+1].permute(0, 1, 4, 3, 2),
                            fc, mode='bilinear', padding_mode='zeros',
                            align_corners=False)
                        append_feats.append(new_feat[0, :, :, 0, 0].permute(1, 0))

                    if img_feats is not None and self.sample_from_img:
                        W_new = W * self.cascade_ratio
                        H_new = H * self.cascade_ratio
                        D_new = D * self.cascade_ratio
                        img_uv, img_mask = project_points_on_img(
                            new_coord,
                            rots=transform[0][b:b+1],
                            trans=transform[1][b:b+1],
                            intrins=transform[2][b:b+1],
                            post_rots=transform[3][b:b+1],
                            post_trans=transform[4][b:b+1],
                            bda_mat=transform[5][b:b+1],
                            W_img=transform[6][1][b:b+1],
                            H_img=transform[6][0][b:b+1],
                            pts_range=self.point_cloud_range,
                            W_occ=W_new, H_occ=H_new, D_occ=D_new)
                        for img_feat in img_feats:
                            sampled = F.grid_sample(
                                img_feat[b].contiguous(), img_uv.contiguous(),
                                align_corners=True, mode='bilinear',
                                padding_mode='zeros')
                            sampled = sampled * img_mask.permute(2, 1, 0)[:, None]
                            sampled = self.img_mlp(
                                sampled.sum(0)[:, :, 0].permute(1, 0))
                            append_feats.append(sampled)

                    output['fine_output'].append(
                        self.fine_mlp(torch.concat(append_feats, dim=1)))

        return {
            'output_voxels':       output['occ'],
            'output_voxels_fine':  output.get('fine_output', None),
            'output_coords_fine':  output.get('fine_coord',  None),
        }
