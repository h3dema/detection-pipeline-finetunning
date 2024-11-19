"""
SSD model with the VGG16 backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16_bn.html#torchvision.models.VGG16_BN_Weights
SSD paper: https://arxiv.org/abs/1512.02325
Resnet: https://arxiv.org/abs/1512.03385
"""

import torchvision
import torch.nn as nn

from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.rpn import AnchorGenerator

from models.backbones.resnet50 import ResNet50


def create_model(num_classes, pretrained=True, coco_model=False, WINDOW_SIZE = 500, v2: bool = True):
    """
    Create an SSD model with ResNet50 backbone.

    Args:
        num_classes (int): The number of classes in the dataset.
        pretrained (bool, optional): Not used.
        coco_model (bool, optional): Not used.
        WINDOW_SIZE (int, optional): The size for the square window used
            in the SSD detection model. Defaults to 500.
        v2 (bool, optional): Select wether to use v2 or v1 of ResNet's weights. Defaults to v2.

    Returns:
        SSD: The SSD model with ResNet50 backbone.
    """

    # Load the pretrained ResNet50 backbone.
    backbone = ResNet50(v2)

    # We need the output channels of the last convolutional layers RetinaNet model.
    # It is 2048 for ResNet50.
    # Notic ethat SSD need out_channels to be a list !!!
    backbone.out_channels = [backbone.get_out_channels()]

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Final SSD model.
    model = SSD(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
        size=(WINDOW_SIZE, WINDOW_SIZE),
    )
    return model


if __name__ == '__main__':
    from models.model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)
