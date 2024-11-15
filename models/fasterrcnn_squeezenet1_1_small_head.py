"""
Backbone: SqueezeNet1_1
Torchvision link: https://pytorch.org/vision/stable/models.html#id15
SqueezeNet repo: https://github.com/forresti/SqueezeNet/tree/master/SqueezeNet_v1.1

Detection Head: Custom Mini Faster RCNN Head.
"""

import torchvision
import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        """
        Initializes the TwoMLPHead.

        Args:
            in_channels (int): The number of input channels.
            representation_size (int): The size of the intermediate representation.
        """
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        """
        Forward pass through the two fully connected layers.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the two fully connected layers.
        """
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
        """
        Initializes the FastRCNNPredictor with the number of input channels and the number of output classes.

        Args:
            in_channels (int): The number of input channels.
            num_classes (int): The number of output classes (including background).
        """
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        """
        Computes the class and bounding box predictions for the input x.

        Args:
            x (Tensor): The input tensor, should be of shape (N, in_channels, 1, 1) where N is the batch size.

        Returns:
            Tuple[Tensor, Tensor]:
                - scores (Tensor): The class prediction tensor, should be of shape (N, num_classes)
                - bbox_deltas (Tensor): The bounding box prediction tensor, should be of shape (N, num_classes * 4)
        """
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def create_model(num_classes=81, pretrained=True, coco_model=False):
    """
    Create a Faster RCNN model with a SqueezeNet1_1 backbone.

    Args:
        num_classes (int): Number of classes to predict.
        pretrained (bool): Whether to use a pre-trained model.
        coco_model (bool): Whether to use a pre-trained model on COCO dataset.

    Returns:
        model (FasterRCNN): The created model.
    """

    # Load the pretrained SqueezeNet1_1 backbone.
    backbone = torchvision.models.squeezenet1_1(pretrained=pretrained).features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 512 for SqueezeNet1_1.
    backbone.out_channels = 512

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

    representation_size = 512

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

