import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes, pretrained=True, coco_model=False):
    """
    Load a pre-trained Faster RCNN model with a ResNet50 backbone and FPN.

    Parameters:
    - num_classes (int): The number of classes that the model should output.
    - pretrained (bool): Whether to use ImageNet weights or not. Default is True.
    - coco_model (bool): Whether to return the model with the COCO classes. Default is False.

    Returns:
    - model: The pre-trained model with the specified number of classes.
    - coco_model (bool): The model with the COCO classes if coco_model is True.
    """

    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT'
    )
    if coco_model:  # Return the COCO pretrained model for COCO classes.
        return model, coco_model

    # Get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == '__main__':
    from models.model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)

