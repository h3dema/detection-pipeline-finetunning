"""
The FCOS model is based on the FCOS: Fully Convolutional One-Stage Object Detection paper.


Torchvision link: https://pytorch.org/vision/stable/models/fcos.html
ResNet paper: https://arxiv.org/abs/1904.01355
"""

import torchvision
import torch.nn as nn

from torchvision.models.detection import FCOS
from torchvision.models.detection.rpn import AnchorGenerator


def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the pretrained MobileNet_V2 backbone.
    backbone = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT).features

    # We need the output channels of the last convolutional layers from
    # output channels in a backbone. For mobilenet_v2, it's 1280.
    backbone.out_channels = 1280

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((1.0,),)
    )

    # Final Faster RCNN model.
    model = FCOS(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
    )
    return model


if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)

