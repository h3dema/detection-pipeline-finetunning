from torch import nn
from torch.nn import functional as F


class DarkNet(nn.Module):
    # A DarkNet model with reduced output channels for each layer.

    def __init__(self, initialize_weights=True, num_classes=1000):
        """
        Args:
            initialize_weights (bool): Whether to initialize the weights of the model.
            num_classes (int): The number of classes in the classification task.
        """
        super(DarkNet, self).__init__()

        self.num_classes = num_classes
        self.features = self._create_conv_layers()
        self.pool = self._pool()
        self.fcs = self._create_fc_layers()

        if initialize_weights:
            # Random initialization of the weights
            # just like the original paper.
            self._initialize_weights()

    def _create_conv_layers(self):
        """
        Creates the convolutional layers of the DarkNet backbone.

        The convolutional layers are based on the YOLOv3 paper.

        Returns:
            nn.Sequential: The convolutional layers of the DarkNet backbone.
        """
        conv_layers = nn.Sequential(
            nn.Conv2d(3, 4, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(4, 8, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return conv_layers

    def _pool(self):
        """
        Creates a pooling layer for the DarkNet backbone.

        This method utilizes average pooling to downsample the feature map.

        Returns:
            nn.Sequential: The pooling layer of the DarkNet backbone.
        """
        pool = nn.Sequential(
            nn.AvgPool2d(7),
        )
        return pool

    def _create_fc_layers(self):
        """
        Creates the fully connected layers of the DarkNet backbone.

        The fully connected layers consist of a single linear layer
        with the number of inputs equal to the number of outputs from
        the convolutional layers and the number of outputs equal to
        the number of classes.

        Returns:
            nn.Sequential: The fully connected layers of the DarkNet backbone.
        """
        fc_layers = nn.Sequential(
            nn.Linear(128, self.num_classes)
        )
        return fc_layers

    def _initialize_weights(self):
        """
        Initialize the weights of the model.

        This method is called by the model constructor and can be overridden
        by subclasses to customize the initialization of the weights.

        The default implementation initializes the weights of the model
        using the Kaiming initialization method for convolutional layers
        and the Xavier initialization method for linear layers.

        See the documentation for PyTorch's nn.init module for more details
        about the initialization methods used by this method.

        :return: None
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_in',
                                       nonlinearity='leaky_relu'
                                       )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the DarkNet backbone.

        Args:
            x (torch.Tensor): The input tensor to the DarkNet backbone.

        Returns:
            torch.Tensor: The output tensor of the DarkNet backbone.
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.fcs(x)
        return x
