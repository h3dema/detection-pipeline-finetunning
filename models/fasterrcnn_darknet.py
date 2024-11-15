import torchvision
import torch


from torchvision.models.detection import FasterRCNN  # type: ignore
from torchvision.models.detection.rpn import AnchorGenerator  # type: ignore

from models.backbones.darknet import DarkNet


def create_model(num_classes, pretrained=True, coco_model=False) -> FasterRCNN:
    """
    Create a Faster RCNN model with a DarkNet backbone.

    Args:
        num_classes: The number of classes to detect.
        pretrained: Not used.
        coco_model: If True, the model is loaded with COCO weights.

    Returns:
        A FasterRCNN model.
    """

    # Load the pretrained Darknet backbone.
    backbone = DarkNet(num_classes=10).features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 1024 for Darknet.
    backbone.out_channels = 1024  # TODO: extract this value from the darknet layer itself

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

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model


#
# Example
# -------
#
# cd detection_pipeline_finetunning
# python3 -m models.fasterrcnn_darknet
#
if __name__ == '__main__':
    from models.model_summary import summary
    model: FasterRCNN = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)

