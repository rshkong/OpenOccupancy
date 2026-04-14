# FlashOCC-style 2D view transformer for OpenOccupancy
# Collapses the Z dimension into channels during LSS voxel pooling,
# yielding [B, numC_Trans * Dz, X, Y] instead of [B, C, X, Y, Z].
# This lets the BEV encoder use pure 2D convolutions (CustomResNet + FPN_LSS).

from mmdet3d.models.builder import NECKS
from .ViewTransformerLSSVoxel import ViewTransformerLiftSplatShootVoxel
from .ViewTransformerLSSBEVDepth import ViewTransformerLSSBEVDepth


@NECKS.register_module()
class ViewTransformerLSSFlash(ViewTransformerLiftSplatShootVoxel):
    """View transformer with Z-collapse (FlashOCC style).

    Identical to ViewTransformerLiftSplatShootVoxel but overrides
    ``voxel_pooling`` to collapse the Z axis into the channel dimension,
    producing a 2-D BEV feature map of shape [B, numC_Trans*Dz, X, Y].

    All depth-supervision utilities (get_depth_loss, etc.) are inherited
    unchanged.
    """

    def voxel_pooling(self, geom_feats, x):
        """Pool lifted features into BEV and collapse Z into channels.

        Args:
            geom_feats: [B, N, D, H, W, 3]  ego-frame coordinates
            x:          [B, N, D, H, W, C]  lifted image features

        Returns:
            final: [B, C * Dz, X, Y]  2-D BEV feature map
        """
        # Delegate to the grandparent (ViewTransformerLSSBEVDepth) pooling
        # which already contains the collapse_z logic:
        #   final = torch.cat(final.unbind(dim=2), 1)  → [B, C*Z, X, Y]
        return ViewTransformerLSSBEVDepth.voxel_pooling(self, geom_feats, x)

    def forward(self, input):
        """Forward pass — identical to parent but returns [B, C*Dz, X, Y].

        Returns:
            bev_feat: [B, numC_Trans * Dz, X, Y]
            depth_prob: [B*N, D, H_feat, W_feat]
        """
        return super().forward(input)
