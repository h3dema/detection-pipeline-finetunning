from torch import nn
from torch.nn import functional as F


class DarkNet(nn.Module):

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
            # random initialization of the weights...
            # ... just like the original paper
            self._initialize_weights()

    def _create_conv_layers(self):
        """
        Creates the convolutional layers of the DarkNet backbone.

        The convolutional layers are based on the YOLOv3 paper.

        Args:
            None

        Returns:
            nn.Sequential: The convolutional layers of the DarkNet backbone.
        """
        conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        return conv_layers

    # def _create_fc_layers(self):
    #     fc_layers = nn.Sequential(
    #         nn.AvgPool2d(7),
    #         nn.Linear(1024, self.num_classes)
    #     )
    #     return fc_layers

    def _pool(self):
        """
        Create a pooling layer for the DarkNet backbone.

        The pooling layer consists of a single average pooling operation
        with a kernel size of 7, which reduces the spatial dimensions of
        the feature map.

        Returns:
            nn.Sequential: The pooling layer.
        """
        pool = nn.Sequential(
            nn.AvgPool2d(7),
        )
        return pool

    def _create_fc_layers(self):
        """
        Create the fully connected layers of the model.

        The fully connected layers consists of a single linear layer
        with the number of inputs equal to the number of outputs from
        the convolutional layers and the number of outputs equal to
        the number of classes.

        Returns:
            nn.Sequential: The fully connected layers.
        """
        fc_layers = nn.Sequential(
            nn.Linear(1024, self.num_classes)
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
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                        nonlinearity='leaky_relu'
                                        )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Perform a forward pass through the DarkNet model.

        :param x: Input tensor with shape (batch_size, channels, height, width).
        :return: Output tensor after passing through convolutional, pooling, and fully connected layers.
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.fcs(x)
        return x

