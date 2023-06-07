import torch
from torch import nn
from torchvision import models, ops
from functools import partial


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_size):
        super(DecoderBlock, self).__init__()

        self.up_sample = nn.Upsample(up_sample_size, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels_1, out_channels_2):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = torch.squeeze(x, dim=1)

        return x


class EfficientUNetV2(nn.Module):
    def __init__(self, in_channels):
        super(EfficientUNetV2, self).__init__()

        self.in_channels = in_channels
        self.efficientnet_v2 = models.efficientnet_v2_s().features
        self.efficientnet_v2[0] = ops.Conv2dNormActivation(in_channels=in_channels,
                                                           out_channels=24,
                                                           norm_layer=partial(nn.BatchNorm2d, eps=0.001),
                                                           activation_layer=nn.SiLU)

        self.encoder_block_1 = self.efficientnet_v2[:2]
        self.encoder_block_2 = self.efficientnet_v2[2:3]
        self.encoder_block_3 = self.efficientnet_v2[3:4]
        self.encoder_block_4 = self.efficientnet_v2[4:5]
        self.encoder_block_5 = self.efficientnet_v2[5:7]

        self.bottleneck_block = BottleneckBlock(256)

        self.decoder_block_4 = DecoderBlock(256, 128, 32)
        self.decoder_block_3 = DecoderBlock(128, 64, 64)
        self.decoder_block_2 = DecoderBlock(64, 48, 128)
        self.decoder_block_1 = DecoderBlock(48, 24, 256)

        self.segmentation_head = SegmentationHead(24, 12, 1)

    def forward(self, x):
        skip_1 = self.encoder_block_1(x)
        skip_2 = self.encoder_block_2(skip_1)
        skip_3 = self.encoder_block_3(skip_2)
        skip_4 = self.encoder_block_4(skip_3)
        x = self.encoder_block_5(skip_4)
        x = self.bottleneck_block(x)
        x = self.decoder_block_4(x, skip_4)
        x = self.decoder_block_3(x, skip_3)
        x = self.decoder_block_2(x, skip_2)
        x = self.decoder_block_1(x, skip_1)
        x = self.segmentation_head(x)

        return x


if __name__ == '__main__':
    in_channels = 3
    model = EfficientUNetV2(in_channels=in_channels).to('mps')
    input = torch.randn(16, in_channels, 256, 256).to('mps')
    output = model(input)
    print(output.shape)

