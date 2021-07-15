import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            torch.nn.Linear(512*8*8, 1024),
            nn.ReLU(inplace=True),
        )

        self.dp_fc = torch.nn.Linear(1024, 2)
        self.dxy_fc = torch.nn.Linear(1024, 9)
        self.dz_fc = torch.nn.Linear(1024, 3)
        self.dtheta_fc = torch.nn.Linear(1024, 1)

    def forward(self, x):
        h = self.conv(x)
        dp = self.dp_fc(h)
        dxy = self.dxy_fc(h)
        dz = self.dz_fc(h)
        dtheta = self.dtheta_fc(h)
        return dp, dxy, dz, dtheta