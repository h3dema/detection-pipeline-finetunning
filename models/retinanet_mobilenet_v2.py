"""
RetinaNet model with the ResNet backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/main/models/retinanet.html
ResNet paper: https://arxiv.org/abs/1708.02002
"""

import torchvision
from torch import nn

from torchvision.models.detection import RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator


def create_model(num_classes, pretrained=True, coco_model=False, v2: bool = True):

    """
    Creates a RetinaNet model with a MobileNetV2 backbone.

    Args:
        num_classes (int): The number of classes for the model's output layer.
        pretrained (bool): If True, loads pretrained weights for the backbone.
        coco_model (bool): Not used in this implementation.
        v2 (bool): If True, uses the V2 version of the MobileNet model.

    Returns:
        RetinaNet: A RetinaNet model with a MobileNetV2 backbone, configured
        with specified RPN anchor generator.
    """

    # Load the pretrained ResNet50 backbone.
    model_backbone = torchvision.models.mobilenet_v2(
        torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2 if v2 else torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    # We need the output channels of the last convolutional layers RetinaNet model.
    # It is 1280 for Mobinet_v2.
    backbone = model_backbone.features
    backbone.out_channels = model_backbone.features[-1][0].out_channels

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Final RetinaNet model.
    model = RetinaNet(
        backbone=backbone,
        num_classes=num_classes,
        anchor_generator=anchor_generator,
    )
    return model


if __name__ == '__main__':
    from models.model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)
