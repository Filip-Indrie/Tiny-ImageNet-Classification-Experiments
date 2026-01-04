import torch
from torch import nn
from torchinfo import summary
from abc import ABC

__all__ = ["AlexNet", "VGG11", "ResNet18", "ResNet34", "ResNet50", "Scope2", "Scope3", "Scope2Atrous", "Scope3Atrous", "ShallowBottleNet", "BottleNet", "DeepBottleNet", "DeeperBottleNet"]

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
    def __init__(self):
        super(AlexNet, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
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
            nn.Linear(4096, 200))



class VGG11(CustomModel):
    def __init__(self):
        super(VGG11, self).__init__()
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        conv_blks = []
        in_channels = 3
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.__vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self._net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.Linear(conv_arch[-1][1] * 4 * 4, 4096), nn.ReLU(), nn.Dropout(0.5), # made for 128x128 input images
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 200))

    @staticmethod
    def __vgg_block(num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)



class Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, in_channels, channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        if strides != 1 or in_channels != channels:
            self.conv3 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=strides)
            self.bn3 = nn.BatchNorm2d(channels)
        else:
            self.conv3 = None
            self.bn3 = None

    def forward(self, x):
        y = nn.ReLU()(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.bn3(self.conv3(x))
        y += x
        return nn.ReLU()(y)



class ResNet18(CustomModel):
    def __init__(self):
        super(ResNet18, self).__init__()

        b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*self.__resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.__resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.__resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.__resnet_block(256, 512, 2))

        self._net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(512, 200))

    @staticmethod
    def __resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))

        return blk



class ResNet34(CustomModel):
    def __init__(self):
        super(ResNet34, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = self.__basic_block(64, 64, 3, first_block=True)
        b2 = self.__basic_block(64, 128, 4)
        b3 = self.__basic_block(128, 256, 6)
        b4 = self.__basic_block(256, 512, 3)

        self._net = nn.Sequential(
            stem, b1, b2, b3, b4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 200)
        )

    @staticmethod
    def __basic_block(input_channels, output_channels, num_residuals, first_block=False):
        blocks = []
        for block in range(num_residuals):
            if block == 0 and not first_block:
                blocks.append(Residual(input_channels, output_channels, strides=2))
            else:
                blocks.append(Residual(output_channels, output_channels))

        return nn.Sequential(*blocks)



class ResidualBottleneck(nn.Module):
    bottleneck = 4

    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels // self.bottleneck, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(output_channels // self.bottleneck)

        self.conv2 = nn.Conv2d(output_channels // self.bottleneck, output_channels // self.bottleneck, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels // self.bottleneck)

        self.conv3 = nn.Conv2d(output_channels // self.bottleneck, output_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(output_channels)

        self.act = nn.ReLU()

        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.act(out + residual)



class ResNet50(CustomModel):
    def __init__(self):
        super(ResNet50, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = self.__basic_block(64, 256, 3, first_block=True)
        b2 = self.__basic_block(256, 512, 4)
        b3 = self.__basic_block(512, 1024, 6)
        b4 = self.__basic_block(1024, 2048, 3)

        self._net = nn.Sequential(
            stem, b1, b2, b3, b4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 200)
        )

    @staticmethod
    def __basic_block(input_channels, output_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0:
                if first_block:
                    blk.append(ResidualBottleneck(input_channels, output_channels, stride=1))
                else:
                    blk.append(ResidualBottleneck(input_channels, output_channels, stride=2))
            else:
                blk.append(ResidualBottleneck(output_channels, output_channels))

        return nn.Sequential(*blk)



class Scope2Block(nn.Module):
    def __init__(self, in_chan, out_chan, num_convs=1, kernel_size1=3, kernel_size2=5, stride=1, dilation=1):
        super().__init__()

        column1 = [nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size1, padding=1, dilation=1), nn.BatchNorm2d(out_chan), nn.ReLU()]
        for _ in range(num_convs - 1):
            column1 += [nn.Conv2d(out_chan, out_chan, kernel_size=kernel_size1, padding=1, dilation=1), nn.BatchNorm2d(out_chan), nn.ReLU()]
        self.conv1 = nn.Sequential(*column1)

        column2 = [nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size2, padding=kernel_size2//2*dilation, dilation=dilation), nn.BatchNorm2d(out_chan), nn.ReLU()]
        for _ in range(num_convs - 1):
            column2 += [nn.Conv2d(out_chan, out_chan, kernel_size=kernel_size2, padding=kernel_size2//2*dilation, dilation=dilation), nn.BatchNorm2d(out_chan), nn.ReLU()]
        self.conv2 = nn.Sequential(*column2)

        self.project = nn.Conv2d(out_chan * 2, out_chan, kernel_size=1, padding=0, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_chan)

        self.residual = nn.Conv2d(in_chan, out_chan, kernel_size=1, padding=0, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_chan)

        self.act = nn.ReLU()

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        concat = torch.cat([a, b], dim=1)
        projection = self.bn1(self.project(concat))

        residual = self.bn2(self.residual(x))

        # MINOR TWEAK: You can scale back the residual connection (0.1) to prevent it dominating the output
        return self.act(projection + residual)



class Scope3Block(nn.Module):
    def __init__(self, in_chan, out_chan, num_convs=1, kernel_size1=3, kernel_size2=5, kernel_size3=7, stride=1, dilation1=1, dilation2=1):
        super().__init__()

        column1 = [nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size1, padding=1, dilation=1), nn.BatchNorm2d(out_chan), nn.ReLU()]
        for _ in range(num_convs - 1):
            column1 += [nn.Conv2d(out_chan, out_chan, kernel_size=kernel_size1, padding=1, dilation=1), nn.BatchNorm2d(out_chan), nn.ReLU()]
        self.conv1 = nn.Sequential(*column1)

        column2 = [nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size2, padding=kernel_size2//2*dilation1, dilation=dilation1), nn.BatchNorm2d(out_chan), nn.ReLU()]
        for _ in range(num_convs - 1):
            column2 += [nn.Conv2d(out_chan, out_chan, kernel_size=kernel_size2, padding=kernel_size2//2*dilation1, dilation=dilation1), nn.BatchNorm2d(out_chan), nn.ReLU()]
        self.conv2 = nn.Sequential(*column2)

        column3 = [nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size3, padding=kernel_size3//2*dilation2, dilation=dilation2), nn.BatchNorm2d(out_chan), nn.ReLU()]
        for _ in range(num_convs - 1):
            column3 += [nn.Conv2d(out_chan, out_chan, kernel_size=kernel_size3, padding=kernel_size3//2*dilation2, dilation=dilation2), nn.BatchNorm2d(out_chan), nn.ReLU()]
        self.conv3 = nn.Sequential(*column3)

        self.project = nn.Conv2d(out_chan * 3, out_chan, kernel_size=1, padding=0, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_chan)

        self.residual = nn.Conv2d(in_chan, out_chan, kernel_size=1, padding=0, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_chan)

        self.act = nn.ReLU()

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        c = self.conv3(x)
        concat = torch.cat([a, b, c], dim=1)
        projection = self.bn1(self.project(concat))

        residual = self.bn2(self.residual(x))

        # MINOR TWEAK: You can scale back the residual connection (0.1) to prevent it dominating the output
        return self.act(projection + residual)



class Scope2(CustomModel):
    def __init__(self):
        super(Scope2, self).__init__()
        b1 = Scope2Block(3, 64, num_convs=1, kernel_size1=3, kernel_size2=5, stride=2, dilation=1)
        b2 = Scope2Block(64, 128, num_convs=1, kernel_size1=3, kernel_size2=5, stride=2, dilation=1)
        b3 = Scope2Block(128, 256, num_convs=1, kernel_size1=3, kernel_size2=5, stride=2, dilation=1)
        b4 = Scope2Block(256, 512, num_convs=1, kernel_size1=3, kernel_size2=5, stride=2, dilation=1)
        b5 = Scope2Block(512, 512, num_convs=1, kernel_size1=3, kernel_size2=5, stride=2, dilation=1)

        self._net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 200)
        )



class Scope3(CustomModel):
    def __init__(self):
        super(Scope3, self).__init__()
        b1 = Scope3Block(3, 64, num_convs=1, kernel_size1=3, kernel_size2=5, kernel_size3=7, stride=2, dilation1=1, dilation2=1)
        b2 = Scope3Block(64, 128, num_convs=1, kernel_size1=3, kernel_size2=5, kernel_size3=7, stride=2, dilation1=1, dilation2=1)
        b3 = Scope3Block(128, 256, num_convs=1, kernel_size1=3, kernel_size2=5, kernel_size3=7, stride=2, dilation1=1, dilation2=1)
        b4 = Scope3Block(256, 512, num_convs=1, kernel_size1=3, kernel_size2=5, kernel_size3=7, stride=2, dilation1=1, dilation2=1)
        b5 = Scope3Block(512, 512, num_convs=1, kernel_size1=3, kernel_size2=5, kernel_size3=7, stride=2, dilation1=1, dilation2=1)

        self._net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 200)
        )



class Scope2Atrous(CustomModel):
    def __init__(self):
        super(Scope2Atrous, self).__init__()

        b1 = Scope2Block(3, 64, num_convs=1, kernel_size1=3, kernel_size2=3, stride=2, dilation=3)
        b2 = Scope2Block(64, 128, num_convs=1, kernel_size1=3, kernel_size2=3, stride=2, dilation=3)
        b3 = Scope2Block(128, 256, num_convs=1, kernel_size1=3, kernel_size2=3, stride=2, dilation=3)
        b4 = Scope2Block(256, 512, num_convs=1, kernel_size1=3, kernel_size2=3, stride=2, dilation=3)
        b5 = Scope2Block(512, 512, num_convs=1, kernel_size1=3, kernel_size2=3, stride=2, dilation=3)

        self._net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 200)
        )



class Scope3Atrous(CustomModel):
    def __init__(self):
        super(Scope3Atrous, self).__init__()

        b1 = Scope3Block(3, 64, num_convs=1, kernel_size1=3, kernel_size2=3, kernel_size3=3, stride=2, dilation1=2, dilation2=3)
        b2 = Scope3Block(64, 128, num_convs=1, kernel_size1=3, kernel_size2=3, kernel_size3=3, stride=2, dilation1=2, dilation2=3)
        b3 = Scope3Block(128, 256, num_convs=1, kernel_size1=3, kernel_size2=3, kernel_size3=3, stride=2, dilation1=2, dilation2=3)
        b4 = Scope3Block(256, 512, num_convs=1, kernel_size1=3, kernel_size2=3, kernel_size3=3, stride=2, dilation1=2, dilation2=3)
        b5 = Scope3Block(512, 512, num_convs=1, kernel_size1=3, kernel_size2=3, kernel_size3=3, stride=2, dilation1=2, dilation2=3)

        self._net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 200)
        )



class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck, num_convs, stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels // bottleneck, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels // bottleneck)

        self.conv2 = nn.Conv2d(out_channels // bottleneck, out_channels // bottleneck, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // bottleneck)

        convs = []
        for _ in range(num_convs - 1):
            convs += [nn.Conv2d(out_channels // bottleneck, out_channels // bottleneck, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channels // bottleneck), nn.ReLU()]
        self.convs = nn.Sequential(*convs)

        self.conv3 = nn.Conv2d(out_channels // bottleneck, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.convs(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.act(out + residual)


def bottleneck_stage(in_channels, out_channels, bottleneck, num_blocks, num_convs, first_block=False):
    blk = []
    for i in range(num_blocks):
        if i == 0:
            if first_block:
                blk.append(BottleneckBlock(in_channels, out_channels, bottleneck, num_convs, stride=1))
            else:
                blk.append(BottleneckBlock(in_channels, out_channels, bottleneck, num_convs, stride=2))
        else:
            blk.append(BottleneckBlock(out_channels, out_channels, bottleneck, num_convs, stride=1))

    return nn.Sequential(*blk)



class ShallowBottleNet(CustomModel):
    def __init__(self):
        super(ShallowBottleNet, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(64, 128, 4, 2, 1, first_block=True)
        b2 = bottleneck_stage(128, 256, 4, 3, 1)
        b3 = bottleneck_stage(256, 512, 4, 4, 1)
        b4 = bottleneck_stage(512, 1024, 2, 2, 1)

        self._net = nn.Sequential(
            stem, b1, b2, b3, b4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 200),
        )



class BottleNet(CustomModel): # haha, u get it? Bottleneck <--> BottleNet :)))))))
    def __init__(self):
        super(BottleNet, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(64, 256, 4, 3, 1, first_block=True)
        b2 = bottleneck_stage(256, 512, 4, 4, 1)
        b3 = bottleneck_stage(512, 1024, 2, 6, 1)
        b4 = bottleneck_stage(1024, 2048, 2, 3, 1)

        self._net = nn.Sequential(
            stem, b1, b2, b3, b4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 200),
        )



class DeepBottleNet(CustomModel):
    def __init__(self):
        super(DeepBottleNet, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(64, 256, 4, 3, 1, first_block=True)
        b2 = bottleneck_stage(256, 512, 4, 4, 1)
        b3 = bottleneck_stage(512, 1024, 2, 6, 2)
        b4 = bottleneck_stage(1024, 2048, 2, 3, 2)

        self._net = nn.Sequential(
            stem, b1, b2, b3, b4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 200),
        )

class DeeperBottleNet(CustomModel):
    def __init__(self):
        super(DeeperBottleNet, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(64, 256, 4, 3, 2, first_block=True)
        b2 = bottleneck_stage(256, 512, 4, 4, 2)
        b3 = bottleneck_stage(512, 1024, 2, 6, 3)
        b4 = bottleneck_stage(1024, 2048, 2, 3, 3)

        self._net = nn.Sequential(
            stem, b1, b2, b3, b4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 200),
        )
