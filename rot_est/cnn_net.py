import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), out_dim=1024):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 6x6
            nn.MaxPool2d(2),
            # 3x3

            nn.Conv2d(256, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 1x1
            nn.Conv2d(128, out_dim, kernel_size=1, padding=0),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x).reshape(x.shape[0], -1)