import torch
from torch import nn
import torch.nn.functional as F
from networks.res import BottleneckBlock, BasicBlock

class ResUNet4L(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channels=(16, 32, 64, 128)):
        super().__init__()
        self.conv_down_1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            BottleneckBlock(hidden_channels[0], hidden_channels[0]),
        )
        self.conv_down_2 = nn.Sequential(
            nn.MaxPool2d(2),
            BottleneckBlock(hidden_channels[0], hidden_channels[1]),
        )
        self.conv_down_4 = nn.Sequential(
            nn.MaxPool2d(2),
            BottleneckBlock(hidden_channels[1], hidden_channels[2]),
        )
        self.conv_down_8 = nn.Sequential(
            nn.MaxPool2d(2),
            BottleneckBlock(hidden_channels[2], hidden_channels[3]),
            nn.Conv2d(hidden_channels[3], hidden_channels[2], kernel_size=1, bias=False)
        )
        self.conv_up_4 = nn.Sequential(
            BottleneckBlock(hidden_channels[2]*2, hidden_channels[2]),
            nn.Conv2d(hidden_channels[2], hidden_channels[1], kernel_size=1, bias=False)
        )
        self.conv_up_2 = nn.Sequential(
            BottleneckBlock(hidden_channels[1]*2, hidden_channels[1]),
            nn.Conv2d(hidden_channels[1], hidden_channels[0], kernel_size=1, bias=False)
        )
        self.conv_up_1 = nn.Sequential(
            BottleneckBlock(hidden_channels[0]*2, hidden_channels[0]),
            nn.Conv2d(hidden_channels[0], out_channel, kernel_size=1)
        )

    def forward(self, x):
        feature_map_1 = self.conv_down_1(x)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
                                                     F.interpolate(feature_map_8, size=feature_map_4.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
                                                     F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
                                                     F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        return feature_map_up_1

class ResUNet5L(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.conv_down_1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            BottleneckBlock(hidden_channels[0], hidden_channels[0]),
            BottleneckBlock(hidden_channels[0], hidden_channels[0]),
        )
        self.conv_down_2 = nn.Sequential(
            nn.MaxPool2d(2),
            BottleneckBlock(hidden_channels[0], hidden_channels[1]),
            BottleneckBlock(hidden_channels[1], hidden_channels[1]),
        )
        self.conv_down_4 = nn.Sequential(
            nn.MaxPool2d(2),
            BottleneckBlock(hidden_channels[1], hidden_channels[2]),
            BottleneckBlock(hidden_channels[2], hidden_channels[2]),
        )
        self.conv_down_8 = nn.Sequential(
            nn.MaxPool2d(2),
            BottleneckBlock(hidden_channels[2], hidden_channels[3]),
            BottleneckBlock(hidden_channels[3], hidden_channels[3]),
        )
        self.conv_down_16 = nn.Sequential(
            nn.MaxPool2d(2),
            BottleneckBlock(hidden_channels[3], hidden_channels[4]),
            BottleneckBlock(hidden_channels[4], hidden_channels[4]),
            nn.Conv2d(hidden_channels[4], hidden_channels[3], kernel_size=1, bias=False)
        )
        self.conv_up_8 = nn.Sequential(
            BottleneckBlock(hidden_channels[3]*2, hidden_channels[3]),
            BottleneckBlock(hidden_channels[3], hidden_channels[3]),
            nn.Conv2d(hidden_channels[3], hidden_channels[2], kernel_size=1, bias=False)
        )
        self.conv_up_4 = nn.Sequential(
            BottleneckBlock(hidden_channels[2]*2, hidden_channels[2]),
            BottleneckBlock(hidden_channels[2], hidden_channels[2]),
            nn.Conv2d(hidden_channels[2], hidden_channels[1], kernel_size=1, bias=False)
        )
        self.conv_up_2 = nn.Sequential(
            BottleneckBlock(hidden_channels[1]*2, hidden_channels[1]),
            BottleneckBlock(hidden_channels[1], hidden_channels[1]),
            nn.Conv2d(hidden_channels[1], hidden_channels[0], kernel_size=1, bias=False)
        )
        self.conv_up_1 = nn.Sequential(
            BottleneckBlock(hidden_channels[0]*2, hidden_channels[0]),
            BottleneckBlock(hidden_channels[0], hidden_channels[0]),
            nn.Conv2d(hidden_channels[0], out_channel, kernel_size=1)
        )

    def forward(self, x):
        feature_map_1 = self.conv_down_1(x)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8,
                                                     F.interpolate(feature_map_16, size=feature_map_8.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
                                                     F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
                                                     F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
                                                     F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        return feature_map_up_1

if __name__ == '__main__':
    model = ResUNet5L(5, 2)
    inp = torch.zeros((2, 5, 64, 64))
    print(model(inp).shape)
    print(1)