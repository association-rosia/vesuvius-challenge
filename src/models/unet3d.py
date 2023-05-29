import torch
import torch.nn as nn
from typing import List


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels1)

        # * Uncomment if we get more gpu memory in the future
        self.conv2 = nn.Conv3d(out_channels1, out_channels2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels2)

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
    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()

        self.conv = ConvBlock3d(in_channels, out_channels1, out_channels2)
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class UNetEncoder3d(nn.Module):
    def __init__(self, nb_blocks: int):
        super().__init__()
        
        self.encoderblocks = nn.ModuleList([])
        self.encoderblocks.append(EncoderBlock3d(1, 32, 64))
        for num_block in range(1, nb_blocks):
            in_channels, out_channels = 2**(5 + num_block), 2**(6 + num_block) # 2**(5 + 1) = 64
            self.encoderblocks.append(EncoderBlock3d(in_channels, out_channels, out_channels))
            
    def forward(self, x):
        list_skips = []
        
        for encoderblock in self.encoderblocks:
            skip, x = encoderblock(x)
            list_skips.append(skip)

        return x, list_skips


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv = ConvBlock3d(in_channels + out_channels1, out_channels1, out_channels2)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class UNetDecoder3d(nn.Module):
    def __init__(self, nb_blocks: int):
        super().__init__()
        
        self.decoderblocks = nn.ModuleList([])
        for num_block in range(nb_blocks, 0, -1):
            in_channels, out_channels = 2**(6 + num_block), 2**(5 + num_block) # 2**(5 + 1) = 64
            self.decoderblocks.append(DecoderBlock3d(in_channels, out_channels, out_channels))

    def forward(self, x, list_skips: List):
        list_skips.reverse()
        
        for decoderblock, skip in zip(self.decoderblocks, list_skips):
            x = decoderblock(x, skip)

        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, inputs_size) -> None:
        super().__init__()

        self.outputs3d = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0)

        self.outputs2d = nn.AdaptiveMaxPool3d((out_channels, inputs_size, inputs_size))

    def forward(self, x):
        # 3D outputs
        x = self.outputs3d(x)

        # 2D outputs
        x = self.outputs2d(x)

        return x


class Unet3d(nn.Module):
    def __init__(self, nb_blocks, inputs_size):
        super().__init__()

        # Architecture
        self.encoder = UNetEncoder3d(nb_blocks=nb_blocks)
        self.bottleneck = ConvBlock3d(2**(5 + nb_blocks), 2**(5 + nb_blocks), 2**(6 + nb_blocks))
        self.decoder = UNetDecoder3d(nb_blocks=nb_blocks)
        self.segmenter = SegmentationHead(
            64, 1, inputs_size
        )

    def forward(self, x):
        # Encoder
        x, list_skips = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder(x, list_skips)

        # Segmentation Head
        x = self.segmenter(x)

        x = torch.squeeze(x, (1, 2))

        return x


if __name__ == "__main__":
    import os
    import sys

    parent = os.path.abspath(os.path.curdir)
    sys.path.insert(1, parent)

    from constant import TILE_SIZE

    model = Unet3d(nb_blocks=3, inputs_size=TILE_SIZE).to(device='cuda').half()
    print(model)
    inputs = torch.randn((8, 1, 8, 256, 256)).to(device='cuda').half()
    print(model(inputs))
