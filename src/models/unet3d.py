import torch
import torch.nn as nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.bn2(x)

        x = self.relu(x)

        return x


class EncoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock3d(in_channels, out_channels)
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class UNetEncoder3d(nn.Module):
    def __init__(self, list_channels):
        super().__init__()

        self.nb_block = len(list_channels) - 1
        for i in range(self.nb_block):
            self.__setattr__(
                f"block_{i}", EncoderBlock3d(list_channels[i], list_channels[i + 1])
            )

    def forward(self, inputs):
        list_skips = []
        x = inputs
        for i in range(self.nb_block):
            outputs, x = self.__getattr__(f"block_{i}")(x)
            list_skips.append(outputs)

        return x, list_skips


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv = ConvBlock3d(out_channels + out_channels, out_channels)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class UNetDecoder3d(nn.Module):
    def __init__(self, list_channels):
        super().__init__()

        self.nb_block = len(list_channels) - 1
        for i in range(self.nb_block, -1, -1):
            self.__setattr__(
                f"block_{i - 1}", DecoderBlock3d(list_channels[i], list_channels[i - 1])
            )

    def forward(self, inputs, list_skips):
        x = inputs

        for i in range(self.nb_block - 1, -1, -1):
            x = self.__getattr__(f"block_{i}")(x, list_skips[i])

        return x


class ClassifierHead(nn.Module):
    def __init__(self, in_channels, out_channels, inputs_size) -> None:
        super().__init__()

        self.outputs3d = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

        self.outputs2d = nn.AdaptiveMaxPool3d((out_channels, inputs_size, inputs_size))

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # 3D outputs
        x = self.outputs3d(inputs)

        # 2D outputs
        x = self.outputs2d(x)

        outputs = self.sigmoid(x)

        return outputs


class UNet3d(nn.Module):
    def __init__(self, list_channels, inputs_size):
        super().__init__()

        # Architecture
        self.encoder = UNetEncoder3d(list_channels[:-1])

        self.bottleneck = ConvBlock3d(*list_channels[-2:])

        self.decoder = UNetDecoder3d(list_channels[1:])

        self.classifier = ClassifierHead(list_channels[1], list_channels[0], inputs_size)

    def forward(self, inputs):
        x = torch.unsqueeze(inputs, 1)

        # Encoder
        x, list_skips = self.encoder(x)

        # Botteneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder(x, list_skips)

        # Classifier
        x = self.classifier(x)
        
        outputs = torch.squeeze(x, 2)
        outputs = torch.squeeze(outputs, 1)

        return outputs


if __name__ == "__main__":
    import os, sys

    parent = os.path.abspath(os.path.curdir)
    sys.path.insert(1, parent)

    from constant import TILE_SIZE

    model = UNet3d(list_channels=[1, 32, 64], inputs_size=TILE_SIZE)
    inputs = torch.randn((8, 8, 256, 256))
    print(model(inputs).shape)
