import torch
from e2cnn import nn

class Equivariant(torch.nn.Module):
    def __init__(self, group, obs_channel=2, n_hidden=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = group
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                      nn.FieldType(self.group, n_hidden // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden // 8 * [self.group.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.group, n_hidden // 8 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden // 4 * [self.group.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.group, n_hidden // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden // 2 * [self.group.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.group, n_hidden // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden * 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * 2 * [self.group.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.group, n_hidden * 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), inplace=True),
            # 1x1
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      nn.FieldType(self.group, 8 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            # nn.ReLU(nn.FieldType(self.group, n_reg * [self.group.regular_repr]), inplace=True),
        )

    def forward(self, x):
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x).tensor.reshape(x.shape[0], -1)