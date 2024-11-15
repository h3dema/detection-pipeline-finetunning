"""
A lot of scripts borrowed/adapted from Detectron2.
https://github.com/facebookresearch/detectron2/blob/38af375052d3ae7331141bc1a22cfa2713b02987/detectron2/modeling/backbone/backbone.py#L11
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torchvision

from functools import partial
from torchvision.models.detection import FasterRCNN
from models.backbones.layers import (
    Backbone,
    PatchEmbed,
    Block,
    get_abs_pos,
    get_norm,
    Conv2d,
    LastLevelMaxPool
)
from models.utils import _assert_strides_are_log2_contiguous
from models.backbones.vit import ViT
from models.backbones.simplefeaturepyramid import SimpleFeaturePyramid


def create_model(num_classes=81, pretrained=True, coco_model=False):
    """
    Creates a Faster RCNN model with a Vision Transformer (ViT) backbone.

    Args:
        num_classes (int): Number of output classes for the model. Default is 81.
        pretrained (bool): If True, loads pretrained weights for the ViT backbone. Default is True.
        coco_model (bool): If True, returns a model pretrained on COCO dataset classes. Default is False.

    Returns:
        model (FasterRCNN): A Faster RCNN model with ViT backbone configured for object detection.
    """

    # Base
    embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
    # Load the pretrained SqueezeNet1_1 backbone.
    net = ViT(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    )

    if pretrained:
        print('Loading MAE Pretrained ViT Base weights...')
        # ckpt = torch.utis('weights/mae_pretrain_vit_base.pth')
        ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth')
        net.load_state_dict(ckpt['model'], strict=False)

    backbone = SimpleFeaturePyramid(
        net,
        in_feature="last_feat",
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=LastLevelMaxPool(),
        norm="LN",
        square_pad=1024,
    )

    backbone.out_channels = 256

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=backbone._out_features,
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        box_roi_pool=roi_pooler
    )
    return model


#
# Example
# -------
#
# cd detection_pipeline_finetunning
# python3 -m models.fasterrcnn_vitdet
#
if __name__ == '__main__':
    from models.model_summary import summary
    model = create_model(81, pretrained=True)
    summary(model)

