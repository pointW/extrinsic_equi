import torch
import torch.nn.functional as F
from torch.distributions import Normal

from e2cnn import gspaces
from e2cnn import nn

from networks.sac_networks import SACGaussianPolicyBase
from networks.ssm import SpatialSoftArgmax
from networks.res import BasicBlock

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class EquiResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, N, initialize=True):
        super(EquiResBlock, self).__init__()
        r2_act = gspaces.Rot2dOnR2(N=N)
        rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            nn.ReLU(feat_type_hid, inplace=True)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),

        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out

class EquivariantEncoder128Dihedral(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, obs_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, x):
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.d4_act, self.obs_channel*[self.d4_act.trivial_repr]))
        return self.conv(x)

class EquivariantEncoder128(torch.nn.Module):
    def __init__(self, group, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = group
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                      nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * 2 * [self.group.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.group, n_out * 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, x):
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)

class EquivariantEncoder64(torch.nn.Module):
    def __init__(self, group, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.group = group
        self.conv = torch.nn.Sequential(
            # 64x64
            nn.R2Conv(nn.FieldType(self.group, obs_channel * [self.group.trivial_repr]),
                      nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.group, n_out // 4 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.group, n_out // 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * 2 * [self.group.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * 2 * [self.group.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.group, n_out * 2 * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.group, n_out * [self.group.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      nn.FieldType(self.group, n_out * [self.group.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_out * [self.group.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, x):
        if type(x) is torch.Tensor:
            x = nn.GeometricTensor(x, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        return self.conv(x)

class NonEquivariantEncBase(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=64, N=4, backbone='cnn'):
        super().__init__()
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.n_hidden = n_hidden
        if backbone == 'cnn':
            if obs_shape[1] == 128:
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    # 64x64
                    torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    # 32x32
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    # 16x16
                    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    # 8x8
                    torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
            else:
                self.conv = torch.nn.Sequential(
                    # 64x64
                    torch.nn.Conv2d(obs_shape[0], 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    # 32x32
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    # 16x16
                    torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                    # 8x8
                    torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                )
        elif backbone == 'res':
            if obs_shape[1] == 128:
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    BasicBlock(32, 32),
                    torch.nn.MaxPool2d(2),
                    # 64x64
                    BasicBlock(32, 64),
                    torch.nn.MaxPool2d(2),
                    # 32x32
                    BasicBlock(64, 128),
                    torch.nn.MaxPool2d(2),
                    # 16x16
                    BasicBlock(128, 256),
                    torch.nn.MaxPool2d(2),
                    # 8x8
                    BasicBlock(256, 512),
                )
            else:
                self.conv = torch.nn.Sequential(
                    # 64x64
                    torch.nn.Conv2d(obs_shape[0], 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(inplace=True),
                    BasicBlock(64, 64),
                    torch.nn.MaxPool2d(2),
                    # 32x32
                    BasicBlock(64, 128),
                    torch.nn.MaxPool2d(2),
                    # 16x16
                    BasicBlock(128, 256),
                    torch.nn.MaxPool2d(2),
                    # 8x8
                    BasicBlock(256, 512),
                )
        else:
            raise NotImplementedError

        self.reducer = None

    def forward(self, x):
        enc_out = self.reducer(self.conv(x))
        enc_out = enc_out.reshape(x.shape[0], -1, 1, 1)
        enc_out = nn.GeometricTensor(enc_out, nn.FieldType(self.d4_act, self.n_hidden * [self.d4_act.regular_repr]))
        return enc_out

    def forwardNormalTensor(self, x):
        return self.forward(x)

class NonEquivariantEncConv(NonEquivariantEncBase):
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=64, N=4, backbone='cnn'):
        super().__init__(obs_shape, n_hidden, N, backbone)
        self.reducer = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            # 6x6
            torch.nn.MaxPool2d(2),
            # 3x3
            torch.nn.Conv2d(512, n_hidden * N * 2, kernel_size=3),
            torch.nn.ReLU(inplace=True),
            # 1x1
        )

class NonEquivariantEncFC(NonEquivariantEncBase):
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=64, N=4, backbone='cnn'):
        super().__init__(obs_shape, n_hidden, N, backbone)
        self.reducer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 8 * 8, n_hidden * N * 2),
            torch.nn.ReLU(inplace=True),
        )

# ssm + fc
class NonEquivariantEncSSM(NonEquivariantEncBase):
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=64, N=4, backbone='cnn'):
        super().__init__(obs_shape, n_hidden, N, backbone)
        self.reducer = torch.nn.Sequential(
            SpatialSoftArgmax(),
            torch.nn.Linear(512 * 2, n_hidden * N * 2),
            torch.nn.ReLU(inplace=True),
        )

# ssm + std->reg equi conv
class NonEquivariantEncSSMStd(NonEquivariantEncBase):
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=64, N=4, backbone='cnn'):
        super().__init__(obs_shape, n_hidden, N, backbone)
        self.reducer = torch.nn.Sequential(
            SpatialSoftArgmax(),
        )
        self.equi_conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, 512 * [self.d4_act.irrep(1, 1)]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=1, padding=0),
        )

    def forward(self, x):
        enc_out = self.reducer(self.conv(x))
        enc_out = enc_out.reshape(x.shape[0], 512 * 2, 1, 1)
        enc_out = nn.GeometricTensor(enc_out, nn.FieldType(self.d4_act, 512 * [self.d4_act.irrep(1, 1)]))
        enc_out = self.equi_conv(enc_out)
        return enc_out

# cnn conv + equi conv
class NonEquivariantEncEqui(NonEquivariantEncBase):
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=64, N=4, backbone='cnn'):
        super().__init__(obs_shape, n_hidden, N, backbone)
        self.reducer = torch.nn.Sequential(
            # 8x8
            nn.R2Conv(nn.FieldType(self.d4_act, 512 * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden * 2 * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * 2 * [self.d4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=3, padding=0),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, x):
        cnn_out = self.conv(x)
        cnn_out = nn.GeometricTensor(cnn_out, nn.FieldType(self.d4_act, 512 * [self.d4_act.trivial_repr]))
        enc_out = self.reducer(cnn_out)
        return enc_out

    def forwardNormalTensor(self, x):
        return self.forward(x)

class NonEquivariantEncSSMParallelEqui(NonEquivariantEncSSM):
    def __init__(self, obs_shape=(2, 128, 128), n_hidden=64, N=4, backbone='cnn'):
        super().__init__(obs_shape, n_hidden, N, backbone)
        if obs_shape[-1] == 128:
            self.equi_conv = EquivariantEncoder128Dihedral(obs_shape[0], n_hidden, N=N)
        else:
            self.equi_conv = EquivariantEncoder64(obs_shape[0], n_hidden, N=N)
        self.out_layer = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, 2 * n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=1, padding=0),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr])),
        )

    def forward(self, x):
        cnn_enc_out = self.reducer(self.conv(x))
        cnn_enc_out = cnn_enc_out.reshape(x.shape[0], -1, 1, 1)

        equi_enc_out = self.equi_conv(x)
        cat = torch.cat([cnn_enc_out, equi_enc_out.tensor], dim=1)
        cat = nn.GeometricTensor(cat, nn.FieldType(self.d4_act, 2 * self.n_hidden * [self.d4_act.regular_repr]))
        return self.out_layer(cat)

def getNonEquivariantEnc(obs_shape=(2, 128, 128), n_hidden=64, N=4, enc_type='fc', backbone='cnn'):
    if enc_type == 'fc':
        return NonEquivariantEncFC(obs_shape, n_hidden, N, backbone)
    elif enc_type == 'equi':
        return NonEquivariantEncEqui(obs_shape, n_hidden, N, backbone)
    elif enc_type == 'ssm':
        return NonEquivariantEncSSM(obs_shape, n_hidden, N, backbone)
    elif enc_type == 'ssmstd':
        return NonEquivariantEncSSMStd(obs_shape, n_hidden, N, backbone)
    elif enc_type == 'ssm+equi':
        return NonEquivariantEncSSMParallelEqui(obs_shape, n_hidden, N, backbone)
    else:
        raise NotImplementedError

class EquivariantEncoder128DihedralK5(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.d4_act, obs_channel * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 8 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 4 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.d4_act, n_out // 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 14x14
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]), inplace=True),
            # 12x12
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * 2 * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 10x10
            nn.PointwiseMaxPool(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), 2),
            # 5x5
            nn.R2Conv(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]),
                      kernel_size=5, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_out * [self.d4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128NoPool(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 4),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(geo, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128Small(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 4),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 4),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            # 6x6
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            # 1x1
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(geo, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128Res(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out // 8 * [self.c4_act.regular_repr]), inplace=True),
            # 128x128
            EquiResBlock(n_out//8, n_out//8, 3, N, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            EquiResBlock(n_out//8, n_out//4, 3, N, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            EquiResBlock(n_out//4, n_out//2, 3, N, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            EquiResBlock(n_out//2, n_out, 3, N, initialize),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            EquiResBlock(n_out, n_out*2, 3, N, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder64_1(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder64_2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

def getEnc(obs_size, enc_id):
    assert obs_size in [128, 64]
    if obs_size == 128:
        return EquivariantEncoder128
    else:
        return EquivariantEncoder64

class EquivariantSACCritic(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_hidden = n_hidden
        self.N = N
        self.group = self.getGroup()
        if obs_shape[-1] == 128:
            self.img_conv = EquivariantEncoder128(self.group, self.obs_channel, n_hidden, initialize)
        elif obs_shape[-1] == 64:
            self.img_conv = EquivariantEncoder64(self.group, self.obs_channel, n_hidden, initialize)
        else:
            raise NotImplementedError

        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(self.getMixFieldType(),
                      nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.group, n_hidden * [self.group.regular_repr])),
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.trivial_repr]),
                      nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(self.getMixFieldType(),
                      nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.group, n_hidden * [self.group.regular_repr])),
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.trivial_repr]),
                      nn.FieldType(self.group, 1 * [self.group.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def getGroup(self):
        return gspaces.Rot2dOnR2(self.N)

    def getMixFieldType(self):
        n_rho1 = 2 if self.N==2 else 1
        return nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr] +
                            (self.action_dim-2) * [self.group.trivial_repr] +
                            n_rho1*[self.group.irrep(1)])

    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        cat = torch.cat((inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        return cat

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.group, self.obs_channel * [self.group.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        action_tensor = self.getActionGeometricTensor(act)
        cat = torch.cat((conv_out.tensor, action_tensor), dim=1)
        cat_geo = nn.GeometricTensor(cat, self.getMixFieldType())
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticDihedral(EquivariantSACCritic):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, N)

    def getGroup(self):
        return gspaces.FlipRot2dOnR2(self.N)

    def getMixFieldType(self):
        n_rho1 = 2 if self.N == 2 else 1
        return nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr] +
                            (self.action_dim - 3) * [self.group.trivial_repr] +
                            n_rho1 * [self.group.irrep(1, 1)] +
                            1 * [self.group.quotient_repr((None, self.N))])

    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:4]), dim=1)
        dtheta = act[:, 4:5]
        n_inv = inv_act.shape[1]
        cat = torch.cat((inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1),
                         dtheta.reshape(batch_size, 1, 1, 1), (-dtheta).reshape(batch_size, 1, 1, 1)), dim=1)
        return cat

class EquivariantSACCriticDihedralAllInv(EquivariantSACCritic):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, N)

    def getMixFieldType(self):
        return nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr] +
                            self.action_dim * [self.group.trivial_repr])

    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        return act.reshape(batch_size, -1, 1, 1)

class EquivariantSACCriticFlip(EquivariantSACCritic):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, 1)

    def getGroup(self):
        return gspaces.Flip2dOnR2()

    def getMixFieldType(self):
        return nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr] +
                            (self.action_dim - 2) * [self.group.trivial_repr] +
                            2 * [self.group.irrep(1)])

    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        dy = act[:, 2:3]
        inv_act = torch.cat((act[:, 0:2], act[:, 3:4]), dim=1)
        dtheta = act[:, 4:5]
        n_inv = inv_act.shape[1]
        cat = torch.cat((inv_act.reshape(batch_size, n_inv, 1, 1),
                         dy.reshape(batch_size, 1, 1, 1),
                         dtheta.reshape(batch_size, 1, 1, 1)), dim=1)
        return cat

class EquivariantSACCriticTrivial(EquivariantSACCritic):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, 1)

    def getGroup(self):
        return gspaces.TrivialOnR2()

    def getMixFieldType(self):
        return nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr] + self.action_dim * [self.group.trivial_repr])

    def getActionGeometricTensor(self, act):
        batch_size = act.shape[0]
        return act.reshape(batch_size, -1, 1, 1)

class EquivariantSACCriticDihedralWithNonEquiEnc(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, initialize=True, N=4, enc_type='fc', backbone='cnn', n_channels=[64, 64]):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        n_layer = len(n_channels)
        assert n_layer >= 2
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_channels[0]
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.img_conv = getNonEquivariantEnc(obs_shape, n_channels[0], N, enc_type, backbone)
        self.n_rho1 = 2 if N==2 else 1
        def getLayers():
            layers = [
                nn.R2Conv(nn.FieldType(self.d4_act, n_channels[0] * [self.d4_act.regular_repr] +
                                       (action_dim - 3) * [self.d4_act.trivial_repr] +
                                       self.n_rho1 * [self.d4_act.irrep(1, 1)] +
                                       1 * [self.d4_act.quotient_repr((None, 4))]),
                          nn.FieldType(self.d4_act, n_channels[1] * [self.d4_act.regular_repr]),
                          kernel_size=1, padding=0, initialize=initialize),
                nn.ReLU(nn.FieldType(self.d4_act, n_channels[1] * [self.d4_act.regular_repr]), inplace=True)
            ]
            for i in range(1, n_layer - 1):
                layers.append(nn.R2Conv(nn.FieldType(self.d4_act, n_channels[i] * [self.d4_act.regular_repr]),
                                        nn.FieldType(self.d4_act, n_channels[i+1] * [self.d4_act.regular_repr]),
                                        kernel_size=1, padding=0, initialize=initialize))
                layers.append(nn.ReLU(nn.FieldType(self.d4_act, n_channels[i+1] * [self.d4_act.regular_repr]), inplace=True))
            layers.append(nn.GroupPooling(nn.FieldType(self.d4_act, n_channels[-1] * [self.d4_act.regular_repr])))
            layers.append(nn.R2Conv(nn.FieldType(self.d4_act, n_channels[-1] * [self.d4_act.trivial_repr]),
                                    nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                                    kernel_size=1, padding=0, initialize=initialize))
            return layers

        self.critic_1 = torch.nn.Sequential(*getLayers())

        self.critic_2 = torch.nn.Sequential(*getLayers())

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        enc_out = self.img_conv(obs)
        if len(act.shape) == 2:
            dxy = act[:, 1:3]
            inv_act = torch.cat((act[:, 0:1], act[:, 3:4]), dim=1)
            dtheta = act[:, 4:5]
            n_inv = inv_act.shape[1]
            cat = torch.cat((enc_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1), dtheta.reshape(batch_size, 1, 1, 1), (-dtheta).reshape(batch_size, 1, 1, 1)), dim=1)
            cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.d4_act, self.n_hidden * [self.d4_act.regular_repr] + n_inv * [self.d4_act.trivial_repr] + self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))]))
            out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
            out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
            return out1, out2
        else:
            dxy = act[:, :, 1:3]
            inv_act = torch.cat((act[:, :, 0:1], act[:, :, 3:4]), dim=2)
            dtheta = act[:, :, 4:5]
            n_inv = inv_act.shape[2]
            enc_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1)

            fused = torch.cat([enc_out.tensor.unsqueeze(1).expand(-1, act.size(1), -1, -1, -1),
                               inv_act.reshape(batch_size, act.size(1), n_inv, 1, 1),
                               dxy.reshape(batch_size, act.size(1), 2, 1, 1),
                               dtheta.reshape(batch_size, act.size(1), 1, 1, 1),
                               (-dtheta).reshape(batch_size, act.size(1), 1, 1, 1)], dim=2)
            B, N, D, _, _ = fused.size()
            fused = fused.reshape(B * N, D, 1, 1)
            fused_geo = nn.GeometricTensor(fused, nn.FieldType(self.d4_act,
                                                               self.n_hidden * [self.d4_act.regular_repr] +
                                                               n_inv * [self.d4_act.trivial_repr] +
                                                               self.n_rho1 * [self.d4_act.irrep(1, 1)] +
                                                               1 * [self.d4_act.quotient_repr((None, 4))]))
            out1 = self.critic_1(fused_geo).tensor.reshape(B, N)
            out2 = self.critic_2(fused_geo).tensor.reshape(B, N)
        return out1, out2

class EquivariantSACCriticDihedralShareEnc(torch.nn.Module):
    def __init__(self, enc, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.img_conv = enc
        self.n_rho1 = 2 if N==2 else 1
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr] + (action_dim - 3) * [self.d4_act.trivial_repr] + self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr] + (action_dim - 3) * [self.d4_act.trivial_repr] + self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.d4_act, self.obs_channel * [self.d4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:4]), dim=1)
        dtheta = act[:, 4:5]
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat((conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1), dtheta.reshape(batch_size, 1, 1, 1), (-dtheta).reshape(batch_size, 1, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.d4_act, self.n_hidden * [self.d4_act.regular_repr] + n_inv * [self.d4_act.trivial_repr] + self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACActor(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_hidden = n_hidden
        self.N = N
        self.n_rho1 = 2 if self.N==2 else 1
        self.group = self.getGroup()
        if obs_shape[-1] == 128:
            self.img_conv = EquivariantEncoder128(self.group, self.obs_channel, n_hidden, initialize)
        elif obs_shape[-1] == 64:
            self.img_conv = EquivariantEncoder64(self.group, self.obs_channel, n_hidden, initialize)
        else:
            raise NotImplementedError
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.group, n_hidden * [self.group.regular_repr]),
                      self.getOutFieldType(),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def getGroup(self):
        return gspaces.Rot2dOnR2(self.N)

    def getOutFieldType(self):
        return nn.FieldType(self.group, self.n_rho1 * [self.group.irrep(1)] + (self.action_dim*2-2) * [self.group.trivial_repr])

    def getOutput(self, conv_out):
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.group, self.obs_channel*[self.group.trivial_repr]))
        conv_out = self.conv(self.img_conv(obs_geo)).tensor.reshape(batch_size, -1)
        return self.getOutput(conv_out)

class EquivariantSACActorDihedral(EquivariantSACActor):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, N)

    def getGroup(self):
        return gspaces.FlipRot2dOnR2(self.N)

    def getOutFieldType(self):
        return nn.FieldType(self.group, self.n_rho1 * [self.group.irrep(1, 1)] +
                            1 * [self.group.quotient_repr((None, self.N))] +
                            (self.action_dim * 2 - 3) * [self.group.trivial_repr])

    def getOutput(self, conv_out):
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim+1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        log_std = conv_out[:, self.action_dim+1:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorDihedralAllInv(EquivariantSACActorDihedral):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, N)

    def getOutFieldType(self):
        return nn.FieldType(self.group, (self.action_dim * 2) * [self.group.trivial_repr])

    def getOutput(self, conv_out):
        mean = conv_out[:, :self.action_dim]
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorFlip(EquivariantSACActor):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, 1)

    def getGroup(self):
        return gspaces.Flip2dOnR2()

    def getOutFieldType(self):
        return nn.FieldType(self.group, self.n_rho1 * [self.group.irrep(1)] + (self.action_dim * 2 - 2) * [self.group.trivial_repr])

    def getOutput(self, conv_out):
        dy = conv_out[:, 0:1]
        dtheta = conv_out[:, 1:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:2], dy, inv_act[:, 2:3], dtheta), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorTrivial(EquivariantSACActor):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, 1)

    def getGroup(self):
        return gspaces.TrivialOnR2()

    def getOutFieldType(self):
        return nn.FieldType(self.group, (self.action_dim * 2) * [self.group.trivial_repr])

    def getOutput(self, conv_out):
        mean = conv_out[:, :self.action_dim]
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantPolicyDihedral(EquivariantSACActorDihedral):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, N)

    def forward(self, obs):
        return super().forward(obs)[0]

class EquivariantSACActorDihedralWithNonEquiEnc(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, initialize=True, N=4, enc_type='fc', backbone='cnn', n_channels=[64]):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        n_layer = len(n_channels)
        assert n_layer >= 1
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.img_conv = getNonEquivariantEnc(obs_shape, n_channels[0], N, enc_type, backbone)
        layers = []
        for i in range(n_layer - 1):
            layers.append(nn.R2Conv(nn.FieldType(self.d4_act, n_channels[i] * [self.d4_act.regular_repr]),
                                    nn.FieldType(self.d4_act, n_channels[i+1] * [self.d4_act.regular_repr]),
                                    kernel_size=1, padding=0, initialize=initialize))
            layers.append(nn.ReLU(nn.FieldType(self.d4_act, n_channels[i+1] * [self.d4_act.regular_repr]), inplace=True))
        layers.append(nn.R2Conv(nn.FieldType(self.d4_act, n_channels[-1] * [self.d4_act.regular_repr]),
                                nn.FieldType(self.d4_act, self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))] + (action_dim * 2 - 3) * [self.d4_act.trivial_repr]),
                                kernel_size=1, padding=0, initialize=initialize))
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, obs):
        batch_size = obs.shape[0]
        enc_out = self.img_conv(obs)
        conv_out = self.conv(enc_out).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim+1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        log_std = conv_out[:, self.action_dim+1:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantPolicyDihedralWithNonEquiEnc(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, enc_type='fc', backbone='cnn'):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_hidden = n_hidden
        self.img_conv = getNonEquivariantEnc(obs_shape, n_hidden, N, enc_type, backbone)
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))] + (action_dim - 3) * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        enc_out = self.img_conv(obs)
        conv_out = self.conv(enc_out).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim+1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        return mean

class EquivariantSACActorDihedralShareEnc(SACGaussianPolicyBase):
    def __init__(self, enc, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, kernel_size=3):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        assert kernel_size in [3, 5]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.img_conv = enc
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))] + (action_dim * 2 - 3) * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.d4_act, self.obs_channel * [self.d4_act.trivial_repr]))
        with torch.no_grad():
            enc_out = self.img_conv(obs_geo)
        conv_out = self.conv(enc_out).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim + 1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        log_std = conv_out[:, self.action_dim + 1:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantPolicy(EquivariantSACActor):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, N)

    def forward(self, obs):
        return super().forward(obs)[0]

class EquivariantSACVecCriticBase(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_rho1 = (obs_dim - 1) // 4
        self.num_obs_inv = obs_dim - 2*self.num_obs_rho1
        self.act = None
        self.q1 = None
        self.q2 = None

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_p = obs[:, 0:1]
        obs_rho1s = []
        obs_invs = []
        for i in range(self.num_obs_rho1):
            obs_rho1s.append(obs[:, 1+i*4:1+i*4+2])
            obs_invs.append(obs[:, 1+i*4+2:1+i*4+4])
        obs_rho1s = torch.cat(obs_rho1s, 1)
        obs_invs = torch.cat(obs_invs, 1)

        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)

        inp = torch.cat((dxy, obs_rho1s, inv_act, obs_p, obs_invs), dim=1).reshape(batch_size, -1, 1, 1)
        inp_geo = nn.GeometricTensor(inp, nn.FieldType(self.act, (self.num_obs_rho1 + 1) * [self.act.irrep(1)] + (self.num_obs_inv + self.action_dim - 2) * [self.act.trivial_repr]))
        out1 = self.q1(inp_geo).tensor.reshape(batch_size, 1)
        out2 = self.q2(inp_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACVecCritic(EquivariantSACVecCriticBase):
    def __init__(self, obs_dim=7, action_dim=5, n_hidden=1024, N=4, initialize=True):
        super().__init__(obs_dim, action_dim)
        self.act = gspaces.Rot2dOnR2(N)
        self.q1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.act, (self.num_obs_rho1 + 1) * [self.act.irrep(1)] + (self.num_obs_inv + self.action_dim - 2) * [self.act.trivial_repr]),
                      nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.act, n_hidden * [self.act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.act, n_hidden * [self.act.trivial_repr]),
                      nn.FieldType(self.act, 1 * [self.act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )
        self.q2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.act, (self.num_obs_rho1 + 1) * [self.act.irrep(1)] + (self.num_obs_inv + self.action_dim - 2) * [self.act.trivial_repr]),
                      nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      nn.FieldType(self.act, n_hidden * [self.act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.act, n_hidden * [self.act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.act, n_hidden * [self.act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.act, n_hidden * [self.act.trivial_repr]),
                      nn.FieldType(self.act, 1 * [self.act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

class EquivariantSACVecGaussianPolicy(SACGaussianPolicyBase):
    def __init__(self, obs_dim=7, action_dim=5, n_hidden=1024, N=4, initialize=True):
        super().__init__()
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_obs_rho1 = (obs_dim - 1) // 4
        self.num_obs_inv = obs_dim - 2*self.num_obs_rho1
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, self.num_obs_rho1 * [self.c4_act.irrep(1)] + self.num_obs_inv * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      kernel_size=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.irrep(1)] + (action_dim*2-2) * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_p = obs[:, 0:1]
        obs_rho1s = []
        obs_invs = []
        for i in range(self.num_obs_rho1):
            obs_rho1s.append(obs[:, 1+i*4:1+i*4+2])
            obs_invs.append(obs[:, 1+i*4+2:1+i*4+4])
        obs_rho1s = torch.cat(obs_rho1s, 1)
        obs_invs = torch.cat(obs_invs, 1)
        inp = torch.cat((obs_rho1s, obs_p, obs_invs), dim=1).reshape(batch_size, -1, 1, 1)
        inp_geo = nn.GeometricTensor(inp, nn.FieldType(self.c4_act, self.num_obs_rho1 * [self.c4_act.irrep(1)] + self.num_obs_inv * [self.c4_act.trivial_repr]))
        conv_out = self.conv(inp_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    o = torch.zeros(1, 2, 64, 64)
    o[0, 0, 10:30, 10:20] = 1
    a = torch.zeros(1, 5)
    a[0, 1:3] = torch.tensor([-1., -1.])

    # o2 = torch.rot90(o, 1, [2, 3])
    # a2 = torch.zeros(1, 5)
    # a2[0, 1:3] = torch.tensor([1., -1.])

    o2 = torch.zeros(1, 2, 64, 64)
    o2[0, 0, 10:30, -20:-10] = 1
    a2 = torch.zeros(1, 5)
    a2[0, 1:3] = torch.tensor([-1., 1.])

    # out = critic(o, a)
    actor = EquivariantSACActorTrivial(obs_shape=(2, 64, 64), action_dim=5, n_hidden=32, initialize=True)
    critic = EquivariantSACCriticTrivial(obs_shape=(2, 64, 64), action_dim=5, n_hidden=32, initialize=True)
    print(critic(o, a))
    # actor = EquivariantSACActor2(obs_shape=(2, 128, 128), action_dim=5, n_hidden=64, initialize=False)
    # out3 = actor(o)
    #
    # critic = EquivariantSACCritic(obs_shape=(2, 64, 64), action_dim=4, n_hidden=64, initialize=False)
    # o = torch.zeros(1, 2, 64, 64)
    # o[0, 0, 10:20, 10:20] = 1
    # a = torch.zeros(1, 4)
    # a[0, 1:3] = torch.tensor([-1., -1.])
    #
    # o2 = torch.rot90(o, 1, [2, 3])
    # a2 = torch.zeros(1, 4)
    # a2[0, 1:3] = torch.tensor([1., -1.])
    #
    # out = critic(o, a)
    #
    # actor = EquivariantSACActor(obs_shape=(2, 64, 64), action_dim=5, n_hidden=64, initialize=False)
    # out2 = actor(o)
    # actor = EquivariantSACActor2(obs_shape=(2, 64, 64), action_dim=5, n_hidden=64, initialize=False)
    # out3 = actor(o)

    critic = EquivariantSACVecCritic(obs_dim=5, action_dim=5, n_hidden=64, initialize=True)
    obs = torch.zeros(1, 5)
    obs[0, 1] = 1
    obs[0, 2] = 0
    act = torch.zeros(1, 5)
    act[0, 1] = 1
    act[0, 2] = 0
    out1 = critic(obs, act)

    obs = torch.zeros(1, 5)
    obs[0, 1] = 0
    obs[0, 2] = 1
    act = torch.zeros(1, 5)
    act[0, 1] = 0
    act[0, 2] = 1
    out2 = critic(obs, act)

    obs = torch.zeros(1, 5)
    obs[0, 1] = 1
    obs[0, 2] = 0
    act = torch.zeros(1, 5)
    act[0, 1] = 0
    act[0, 2] = 1
    out3 = critic(obs, act)

    actor = EquivariantSACVecGaussianPolicy(obs_dim=5, action_dim=5, n_hidden=64, initialize=False)
    out5 = actor(obs)