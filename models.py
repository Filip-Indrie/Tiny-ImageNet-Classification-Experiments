from torch import nn
from torchinfo import summary
from abc import ABC

class CustomModel(nn.Module, ABC):
    """
        Abstract class designed to ease the implementation
        of other models. It is designed to be used as a
        template only for models taking as input a 128x128 RGB image.
    """

    def __init__(self):
        super(CustomModel, self).__init__()
        self._net = None

    def forward(self, x):
        return self._net(x)

    def __str__(self):
        return str(summary(self._net, input_size=(1, 3, 128, 128), verbose=0))



class AlexNet(CustomModel):
    def __init__(self, in_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(1024, 4096), nn.ReLU(), # made for 128x128 input images
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))



class VGG11(CustomModel):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG11, self).__init__()
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        conv_blks = []
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.__vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self._net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.Linear(conv_arch[-1][1] * 4 * 4, 4096), nn.ReLU(), nn.Dropout(0.5), # made for 128x128 input images
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

    def __vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)



class ResNet18(CustomModel):
    def __init__(self, in_channels=3, num_classes=1000):
        super(ResNet18, self).__init__()

        b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*self.__resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.__resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.__resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.__resnet_block(256, 512, 2))

        self._net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(512, num_classes))

    def __resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):

        class Residual(nn.Module):
            """The Residual block of ResNet."""

            def __init__(self, input_channels, num_channels,
                         use_1x1conv=False, strides=1):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, num_channels,
                                       kernel_size=3, padding=1, stride=strides)
                self.bn1 = nn.BatchNorm2d(num_channels)
                self.conv2 = nn.Conv2d(num_channels, num_channels,
                                       kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(num_channels)
                if use_1x1conv:
                    self.conv3 = nn.Conv2d(input_channels, num_channels,
                                           kernel_size=1, stride=strides)
                else:
                    self.conv3 = None

            def forward(self, X):
                Y = nn.ReLU()(self.bn1(self.conv1(X)))
                Y = self.bn2(self.conv2(Y))
                if self.conv3:
                    X = self.conv3(X)
                Y += X
                return nn.ReLU()(Y)

        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))

        return blk
