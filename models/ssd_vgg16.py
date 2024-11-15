"""
SSD model with the VGG16 backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16_bn.html#torchvision.models.VGG16_BN_Weights
SSD paper: https://arxiv.org/abs/1512.02325
"""

import torchvision
import torch.nn as nn

from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.rpn import AnchorGenerator


def create_model(num_classes, pretrained=True, coco_model=False, WINDOW_SIZE = 500):
    """
    Create an SSD model with VGG16 backbone.

    Args:
        num_classes (int): The number of classes in the dataset.
        pretrained (bool, optional): If True, use the weights from the
            default pretrained model to initialize the VGG16 backbone.
            Defaults to True.
        coco_model (bool, optional): If True, use the class labels from the
            COCO dataset. Defaults to False.
        WINDOW_SIZE (int, optional): The size for the square window used
            in the SSD detection model. Defaults to 500.

    Returns:
        SSD: The SSD model with VGG16 backbone.
    """

    # Load the pretrained VGG16 backbone.
    model_backbone = torchvision.models.vgg16_bn(weights='DEFAULT')

    backbone = model_backbone.features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 512 for VGG16.
    backbone.out_channels = [model_backbone.features[40].out_channels]

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
