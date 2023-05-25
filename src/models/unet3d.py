import torch
import torch.nn as nn
from typing import List


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)

        # * Uncomment if we get more gpu memory in the future
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
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

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class UNetEncoder3d(nn.Module):
    def __init__(self, list_channels: List):
        super().__init__()

        self.encoderblocks = nn.ModuleList([])
        for in_channels, out_channels in zip(list_channels[:-1], list_channels[1:]):
            self.encoderblocks.append(EncoderBlock3d(in_channels, out_channels))
            
    def forward(self, x):
        list_skips = []
        
        for encoderblock in self.encoderblocks:
            skip, x = encoderblock(x)
            list_skips.append(skip)

        return x, list_skips


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv = ConvBlock3d(out_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class UNetDecoder3d(nn.Module):
    def __init__(self, list_channels: List):
        super().__init__()
        
        list_channels.reverse()
        
        self.decoderblocks = nn.ModuleList([])
        for in_channels, out_channels in zip(list_channels[:-1], list_channels[1:]):
            self.decoderblocks.append(DecoderBlock3d(in_channels, out_channels))

    def forward(self, x, list_skips: List):
        list_skips.reverse()
        
        for decoderblock, skip in zip(self.decoderblocks, list_skips):
            x = decoderblock(x, skip)

        return x


class SegmenterHead(nn.Module):
    def __init__(self, in_channels, out_channels, inputs_size) -> None:
        super().__init__()

        self.outputs3d = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

        self.outputs2d = nn.AdaptiveMaxPool3d((out_channels, inputs_size, inputs_size))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 3D outputs
        x = self.outputs3d(x)

        # 2D outputs
        x = self.outputs2d(x)

        outputs = self.sigmoid(x)

        return outputs


class UNet3D(nn.Module):
    def __init__(self, list_channels, inputs_size):
        super().__init__()

        # Architecture
        self.encoder = UNetEncoder3d(list_channels[:-1])
        self.bottleneck = ConvBlock3d(*list_channels[-2:])
        self.decoder = UNetDecoder3d(list_channels[1:])
        self.segmenter = SegmenterHead(
            list_channels[1], list_channels[0], inputs_size
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        # Encoder
        x, list_skips = self.encoder(x)

        # Botteneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder(x, list_skips)

        # Segmenter Head
        x = self.segmenter(x)

        # * For Torch 2.0: torch.squeeze(x, (1, 2))
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 1)

        return x


if __name__ == "__main__":
    import os
    import sys

    parent = os.path.abspath(os.path.curdir)
    sys.path.insert(1, parent)

    from constant import TILE_SIZE

    model = UNet3D(list_channels=[1, 32, 64], inputs_size=TILE_SIZE)
    inputs = torch.randn((8, 8, 256, 256))
    print(model(inputs).shape)
