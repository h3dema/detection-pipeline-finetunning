"""
Custom Faster RCNN model with a smaller DarkNet backbone and a very small detection
head as well.
Detection head representation size is 128.
"""
import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN  # type: ignore
from torchvision.models.detection.rpn import AnchorGenerator  # type: ignore

from models.backbones.darknet_nano import DarkNet


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def create_model(num_classes, pretrained=True, coco_model=False):
    """
    Create a custom Faster RCNN model with a smaller DarkNet backbone and a very small detection head.

    Args:
        num_classes (int): number of output classes (including background)
        pretrained (bool): not used.
        coco_model (bool): not used.

    Returns:
        torch.nn.Module: the custom Faster RCNN model.
    """

    # Load the Mini DarkNet model features.
    backbone = DarkNet(num_classes=10).features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 128 for this custom Mini DarkNet model.
    backbone.out_channels = 128

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    representation_size = 128

    # Box head.
    box_head = TwoMLPHead(
        in_channels=backbone.out_channels * roi_pooler.output_size[0] ** 2,
        representation_size=representation_size
    )

    # Box predictor.
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=None,  # Num classes shoule be None when `box_predictor` is provided.
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=box_predictor
    )
    return model


if __name__ == '__main__':
    from models.model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)

