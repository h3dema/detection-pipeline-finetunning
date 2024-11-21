""" This is a generic Efficientnet-based backbone """
from torch import nn
import torchvision


class EfficientnetBackbone(nn.Module):
    
    def __init__(self, b: str = "b0"):
        """
            b (str):
                Define what type of EfficientNet is created from b0 to b7.
                Defaults to b0.
        """
        super().__init__()
        self.models = {
            "b0": torchvision.models.efficientnet_b0, 
            "b1": torchvision.models.efficientnet_b1, 
            "b2": torchvision.models.efficientnet_b2, 
            "b3": torchvision.models.efficientnet_b3, 
            "b4": torchvision.models.efficientnet_b4, 
            "b5": torchvision.models.efficientnet_b5, 
            "b6": torchvision.models.efficientnet_b6, 
            "b7": torchvision.models.efficientnet_b7,
        }
        b = b.lower()
        assert b in self.models.keys(), "Efficientnet_{b} is not valid"
        self.backbone = self.models[b](weights='DEFAULT').features
        self._out_channels = self.backbone[-1][0].out_channels

    @property
    def out_channels(self):
        # We need the output channels of the last convolutional layers from
        # the features for the model we are inserting this backbone.
        return self._out_channels

    def forward(self, x):
        return self.backbone(x)
