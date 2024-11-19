from torch import nn
import torchvision


class ResNet50(nn.Module):
    def __init__(self, v2: bool = True):
        super().__init__()
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
        self.backbone = nn.Sequential(
            conv1,
            bn1,
            relu,
            max_pool,
            layer1,
            layer2,
            layer3,
            layer4
        )

    def get_out_channels(self):
        return self.backbone[-1][2].conv3.out_channels

    def forward(self, x):
        return self.backbone(x)
