import torch
import math
from torch import nn
from torchinfo import summary
from abc import ABC

__all__ = ["AlexNet", "VGG11", "ResNet18", "ResNet34", "ResNet50", "Scope2", "Scope3",
           "Scope2Atrous", "Scope3Atrous", "ShallowBottleNet", "BottleNet", "DeepBottleNet", "DeeperBottleNet",
           "DilatedHeadNet", "MultiHeadNet", "ShallowMultiHeadNet",
           "StandardTransformer", "DeepTransformer", "WideTransformer", "DeepWideTransformer",
           "WideTransformerV2", "DeepWideTransformerV2", "WideTransformerV3", "DeepWideTransformerV3",
           "LowResTransformer", "DeepLowResTransformer", "WideLowResTransformer", "DeepWideLowResTransformer","DeeperWideLowResTransformer",
           "WideLowResTransformerV2", "DeepWideLowResTransformerV2", "DeeperWideLowResTransformerV2",
           "CNNViT", "CNNViTNoBottleneck", "LowResCNNViT", "LowResCNNViTNoBottleneck",]

class CustomModel(nn.Module, ABC):
    """
        Abstract class designed to ease the implementation
        of other models. It is designed to be used as a
        template only for models taking as input a 128x128 RGB image.
    """

    def __init__(self):
        super(CustomModel, self).__init__()
        self._net = None

    def forward(self, x, **kwargs):
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
    def __init__(self, in_channels, out_channels, bottleneck, num_convs, stride=1, dilation=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels // bottleneck, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bn1 = nn.BatchNorm2d(out_channels // bottleneck)

        self.conv2 = nn.Conv2d(out_channels // bottleneck, out_channels // bottleneck, kernel_size=3, stride=stride, padding=3//2*dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels // bottleneck)

        convs = []
        for _ in range(num_convs - 1):
            convs += [nn.Conv2d(out_channels // bottleneck, out_channels // bottleneck, kernel_size=3, stride=1, padding=3//2*dilation, dilation=dilation), nn.BatchNorm2d(out_channels // bottleneck), nn.ReLU()]
        self.convs = nn.Sequential(*convs)

        self.conv3 = nn.Conv2d(out_channels // bottleneck, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, dilation=1),
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


def bottleneck_stage(in_channels, out_channels, bottleneck, num_blocks, num_convs, dilation=1, maintain_resolution=False):
    blk = []
    for i in range(num_blocks):
        if i == 0:
            if maintain_resolution:
                blk.append(BottleneckBlock(in_channels, out_channels, bottleneck, num_convs, stride=1, dilation=dilation))
            else:
                blk.append(BottleneckBlock(in_channels, out_channels, bottleneck, num_convs, stride=2, dilation=dilation))
        else:
            blk.append(BottleneckBlock(out_channels, out_channels, bottleneck, num_convs, stride=1, dilation=dilation))

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

        b1 = bottleneck_stage(64, 128, 4, 2, 1, maintain_resolution=True)
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

        b1 = bottleneck_stage(64, 256, 4, 3, 1, maintain_resolution=True)
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

        b1 = bottleneck_stage(64, 256, 4, 3, 1, maintain_resolution=True)
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

        b1 = bottleneck_stage(64, 256, 4, 3, 2, maintain_resolution=True)
        b2 = bottleneck_stage(256, 512, 4, 4, 2)
        b3 = bottleneck_stage(512, 1024, 2, 6, 3)
        b4 = bottleneck_stage(1024, 2048, 2, 3, 3)

        self._net = nn.Sequential(
            stem, b1, b2, b3, b4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 200),
        )



class DilatedHeadNet(CustomModel):
    def __init__(self):
        super(DilatedHeadNet, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(64, 128, 4, 3, 1, maintain_resolution=True)
        b2 = bottleneck_stage(128, 256, 4, 4, 1)
        b3 = bottleneck_stage(256, 512, 4, 2, 1, dilation=2, maintain_resolution=True)
        b4 = bottleneck_stage(512, 1024, 4, 2, 1, dilation=4, maintain_resolution=True)
        b5 = bottleneck_stage(1024, 1024, 2, 2, 1, dilation=6, maintain_resolution=True)

        self._net = nn.Sequential(
            stem, b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 200)
        )

class MultiheadBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiheadBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3//2*2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3//2*4, dilation=4)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3//2*6, dilation=6)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):
        a = self.bn1(self.conv1(x))
        b = self.bn2(self.conv2(x))
        c = self.bn3(self.conv3(x))
        d = self.bn4(self.conv4(x))

        concat = torch.cat([a, b, c, d], dim=1)

        return self.act(concat)


class MultiHeadNet(CustomModel):
    def __init__(self):
        super(MultiHeadNet, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(64, 128, 4, 3, 1, maintain_resolution=True)
        b2 = bottleneck_stage(128, 256, 4, 4, 1)
        b3 = bottleneck_stage(256, 512, 2, 6, 3, maintain_resolution=True)
        multihead = MultiheadBlock(512, 512)

        self._net = nn.Sequential(
            stem, b1, b2, b3, multihead,
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 200)
        )

class ShallowMultiHeadNet(CustomModel):
    def __init__(self):
        super(ShallowMultiHeadNet, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(64, 128, 4, 3, 1, maintain_resolution=True)
        b2 = bottleneck_stage(128, 256, 4, 4, 1)
        multihead = MultiheadBlock(256, 512)

        self._net = nn.Sequential(
            stem, b1, b2, multihead,
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 200)
        )

class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_dim):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_dim, stride=patch_dim)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2) # B x C x H x W --> B x ((H / embed) * (W / embed)) x C
        return x

class Embeddings(nn.Module):
    def __init__(self, in_channels=3, image_shape=128, embed_size=48, patch_dim=4, dropout=0.1):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(in_channels=in_channels, embed_dim=embed_size, patch_dim=patch_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size)) # adds the class token to the embeddings
        num_patches = (image_shape // patch_dim) ** 2
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_size)) # learned positional encodings
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # 1 class token for each image in a batch
        x = torch.cat((cls_tokens, x), dim=1) # concatenates the class token to the other tokens
        x = x + self.position_embeddings # adds positional encoding to the image embedding
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, attention_head_size):
        super().__init__()
        self.hidden_size = embed_dim
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(embed_dim, attention_head_size)
        self.key = nn.Linear(embed_dim, attention_head_size)
        self.value = nn.Linear(embed_dim, attention_head_size)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)
        return attention_output, attention_probs

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, attention_heads):
        super().__init__()
        self.hidden_size = embed_size
        self.num_attention_heads = attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(int(self.hidden_size), int(self.attention_head_size))
            self.heads.append(head)
        self.output_projection = nn.Linear(int(self.all_head_size), int(self.hidden_size))

    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        if not output_attentions:
            return attention_output, None
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return attention_output, attention_probs


def GELU(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()
        self.dense_1 = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = GELU
        self.dense_2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(self.dropout(x))
        x = self.dropout(self.dense_2(x))
        return x

class Block(nn.Module):
    def __init__(self, embed_size, num_heads, mlp_hidden_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.layernorm_1 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, mlp_hidden_size, dropout)
        self.layernorm_2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x + self.dropout(attention_output) # residual
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output # residual; output layer is performed by the MLP block
        if not output_attentions:
            return x, None
        else:
            return x, attention_probs

class Encoder(nn.Module):
    def __init__(self, num_blocks, embed_size, num_heads, mlp_hidden_size, dropout):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_size)
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            block = Block(embed_size, num_heads, mlp_hidden_size, dropout=dropout)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)

        x = self.layernorm(x)
        if not output_attentions:
            return x, None
        else:
            return x, all_attentions


def _init_transformer_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, Embeddings):
        module.position_embeddings.data = nn.init.trunc_normal_(
            module.position_embeddings.data.to(torch.float32),
            mean=0.0,std=0.02,).to(module.position_embeddings.dtype)
        module.cls_token.data = nn.init.trunc_normal_(
            module.cls_token.data.to(torch.float32),
            mean=0.0,std=0.02,).to(module.cls_token.dtype)


class ClassificationTransformer(nn.Module):
    def __init__(self, embed_size, patch_dim, num_blocks, num_heads, mlp_hidden_size, dropout=0.1, in_channels=3, image_shape=128):
        super().__init__()
        self.num_classes = 200
        self.embedding = Embeddings(in_channels=in_channels, image_shape=image_shape, embed_size=embed_size, patch_dim=patch_dim, dropout=dropout)
        self.encoder = Encoder(num_blocks, embed_size, num_heads, mlp_hidden_size, dropout=dropout)
        self.classifier = nn.Linear(embed_size, self.num_classes)
        self.apply(_init_transformer_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        logits = self.classifier(encoder_output[:, 0, :]) # only taking into account the class token
        if not output_attentions:
            return logits, None
        else:
            return logits, all_attentions


class StandardTransformer(CustomModel):
    def __init__(self):
        super(StandardTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=48, patch_dim=8, num_blocks=4, num_heads=4, mlp_hidden_size=256)

class DeepTransformer(CustomModel):
    def __init__(self):
        super(DeepTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=48, patch_dim=8, num_blocks=6, num_heads=4, mlp_hidden_size=256)

class WideTransformer(CustomModel):
    def __init__(self):
        super(WideTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=96, patch_dim=8, num_blocks=4, num_heads=4, mlp_hidden_size=512)

class DeepWideTransformer(CustomModel):
    def __init__(self):
        super(DeepWideTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=96, patch_dim=8, num_blocks=6, num_heads=4, mlp_hidden_size=512)

class WideTransformerV2(CustomModel):
    def __init__(self):
        super(WideTransformerV2, self).__init__()

        self._net = ClassificationTransformer(embed_size=128, patch_dim=8, num_blocks=4, num_heads=4, mlp_hidden_size=512)

class DeepWideTransformerV2(CustomModel):
    def __init__(self):
        super(DeepWideTransformerV2, self).__init__()

        self._net = ClassificationTransformer(embed_size=128, patch_dim=8, num_blocks=6, num_heads=4, mlp_hidden_size=512)

class WideTransformerV3(CustomModel):
    def __init__(self):
        super(WideTransformerV3, self).__init__()

        self._net = ClassificationTransformer(embed_size=256, patch_dim=8, num_blocks=4, num_heads=4, mlp_hidden_size=1024)

class DeepWideTransformerV3(CustomModel):
    def __init__(self):
        super(DeepWideTransformerV3, self).__init__()

        self._net = ClassificationTransformer(embed_size=256, patch_dim=8, num_blocks=6, num_heads=4, mlp_hidden_size=1024)

class LowResTransformer(CustomModel):
    def __init__(self):
        super(LowResTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=64, patch_dim=16, num_blocks=4, num_heads=4, mlp_hidden_size=256)

class DeepLowResTransformer(CustomModel):
    def __init__(self):
        super(DeepLowResTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=64, patch_dim=16, num_blocks=6, num_heads=4, mlp_hidden_size=256)

class WideLowResTransformer(CustomModel):
    def __init__(self):
        super(WideLowResTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=128, patch_dim=16, num_blocks=4, num_heads=4, mlp_hidden_size=512)

class DeepWideLowResTransformer(CustomModel):
    def __init__(self):
        super(DeepWideLowResTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=128, patch_dim=16, num_blocks=6, num_heads=4, mlp_hidden_size=512)

class DeeperWideLowResTransformer(CustomModel):
    def __init__(self):
        super(DeeperWideLowResTransformer, self).__init__()

        self._net = ClassificationTransformer(embed_size=128, patch_dim=16, num_blocks=8, num_heads=4, mlp_hidden_size=512)

class WideLowResTransformerV2(CustomModel):
    def __init__(self):
        super(WideLowResTransformerV2, self).__init__()

        self._net = ClassificationTransformer(embed_size=256, patch_dim=16, num_blocks=4, num_heads=4, mlp_hidden_size=1024)

class DeepWideLowResTransformerV2(CustomModel):
    def __init__(self):
        super(DeepWideLowResTransformerV2, self).__init__()

        self._net = ClassificationTransformer(embed_size=256, patch_dim=16, num_blocks=6, num_heads=4, mlp_hidden_size=1024)

class DeeperWideLowResTransformerV2(CustomModel):
    def __init__(self):
        super(DeeperWideLowResTransformerV2, self).__init__()

        self._net = ClassificationTransformer(embed_size=256, patch_dim=16, num_blocks=8, num_heads=4, mlp_hidden_size=1024)

class CNNViT(CustomModel):
    def __init__(self):
        super(CNNViT, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(32, 64, 2, 3, 1, maintain_resolution=True)

        transformer = ClassificationTransformer(embed_size=256, patch_dim=2, num_blocks=4, num_heads=4, mlp_hidden_size=1024, in_channels=64, image_shape=32)

        self._net = nn.Sequential(
            stem,
            b1,
            transformer
        )

class CNNViTNoBottleneck(CustomModel):
    def __init__(self):
        super(CNNViTNoBottleneck, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(32, 64, 1, 3, 1, maintain_resolution=True)

        transformer = ClassificationTransformer(embed_size=256, patch_dim=2, num_blocks=4, num_heads=4, mlp_hidden_size=1024, in_channels=64, image_shape=32)

        self._net = nn.Sequential(
            stem,
            b1,
            transformer
        )

class LowResCNNViT(CustomModel):
    def __init__(self):
        super(LowResCNNViT, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(32, 64, 2, 3, 1, maintain_resolution=True)

        transformer = ClassificationTransformer(embed_size=512, patch_dim=4, num_blocks=4, num_heads=4, mlp_hidden_size=2048, in_channels=64, image_shape=32)

        self._net = nn.Sequential(
            stem,
            b1,
            transformer
        )

class LowResCNNViTNoBottleneck(CustomModel):
    def __init__(self):
        super(LowResCNNViTNoBottleneck, self).__init__()

        stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        b1 = bottleneck_stage(32, 64, 1, 3, 1, maintain_resolution=True)

        transformer = ClassificationTransformer(embed_size=512, patch_dim=4, num_blocks=4, num_heads=4, mlp_hidden_size=2048, in_channels=64, image_shape=32)

        self._net = nn.Sequential(
            stem,
            b1,
            transformer
        )

if __name__ == "__main__":
    net = LowResCNNViTNoBottleneck()
    print(net)