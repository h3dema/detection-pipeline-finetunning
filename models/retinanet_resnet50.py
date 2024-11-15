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

    # Load the pretrained ResNet50 backbone.
    model_backbone = torchvision.models.resnet50(
        torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if v2 else torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )

    conv1 = model_backbone.conv1
    bn1 = model_backbone.bn1
    relu = model_backbone.relu
    max_pool = model_backbone.maxpool
    layer1 = model_backbone.layer1
    layer2 = model_backbone.layer2
    layer3 = model_backbone.layer3
    layer4 = model_backbone.layer4
    backbone = nn.Sequential(
        conv1,
        bn1,
        relu,
        max_pool,
        layer1,
        layer2,
        layer3,
        layer4
    )

    # We need the output channels of the last convolutional layers RetinaNet model.
    # It is 2048 for ResNet50.
    backbone.out_channels = backbone[-1][2].conv3.out_channels

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
