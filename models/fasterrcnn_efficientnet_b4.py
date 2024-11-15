"""
Faster RCNN model with the EfficientNetB4 backbone.
"""
import torchvision

from torchvision.models.detection import FasterRCNN  # type: ignore
from torchvision.models.detection.rpn import AnchorGenerator  # type: ignore


def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the pretrained EfficientNetB0 large features.
    """
    Create a Faster RCNN model with the EfficientNetB4 backbone.

    Args:
        num_classes: The number of classes to detect.
        pretrained: not used.
        coco_model: not used.

    Returns:
        A Faster RCNN model with the EfficientNetB4 backbone.
    """
    backbone = torchvision.models.efficientnet_b4(weights='DEFAULT').features

    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    backbone.out_channels = 1792

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
# python3 -m models.fasterrcnn_efficientnet_b4
#
if __name__ == '__main__':
    from models.model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)

