"""
RetinaNet model with the ResNet backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/main/models/retinanet.html
ResNet paper: https://arxiv.org/abs/1708.02002
"""

import torchvision
from torch import nn

from torchvision.models.detection import RetinaNet
from torchvision.models.detection.rpn import AnchorGenerator

from models.backbones.resnet50 import ResNet50


def create_model(num_classes, pretrained=True, coco_model=False, v2: bool = True):
    """
    Creates a RetinaNet model with a ResNet50 backbone.

    Args:
        num_classes (int): The number of classes for the model's output layer.
        pretrained (bool): If True, loads pretrained weights for the backbone.
        coco_model (bool): Not used in this implementation.
        v2 (bool): If True, uses the V2 version of the ResNet50 model.

    Returns:
        RetinaNet: A RetinaNet model with a ResNet50 backbone, configured
        with specified RPN anchor generator.
    """

    # # Load the pretrained ResNet50 backbone.
    backbone = ResNet50(v2)

    # # We need the output channels of the last convolutional layers RetinaNet model.
    # # It is 2048 for ResNet50.
    backbone.out_channels = backbone.get_out_channels()

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
