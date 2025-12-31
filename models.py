from torch import nn

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), # in_features depends on the size of the input image (make it 64x64)
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        return self.net(x)

class VGG11(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG11, self).__init__()
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        conv_blks = []
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.Linear(conv_arch[-1][1] * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5), # change in_channels for 64x64 images
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        return self.net(x)

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

class ResNet18(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(ResNet18, self).__init__()

        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))

        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(512, 10))

    def forward(self, x):
        return self.net(x)

    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):

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