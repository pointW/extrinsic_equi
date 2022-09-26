import torch

from e2cnn import gspaces
from e2cnn import nn

from networks.sac_networks import SACGaussianPolicyBase

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

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
    def __init__(self, group, obs_channel=2, n_out=128, initialize=True, backbone=None):
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
            nn.R2Conv(nn.FieldType(self.group, n_out // 8 * [self.group.regular_repr]),
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

def getEnc(obs_size, enc_id):
    assert obs_size in [128]
    return EquivariantEncoder128

class EquivariantSACCritic(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, backbone=None):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_hidden = n_hidden
        self.N = N
        self.group = self.getGroup()
        if obs_shape[-1] == 128:
            self.img_conv = EquivariantEncoder128(self.group, self.obs_channel, n_hidden, initialize, backbone)
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
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, backbone=None):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, N, backbone)

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

class EquivariantSACCriticFlip(EquivariantSACCritic):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, backbone=None):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, 1, backbone)

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

class EquivariantSACCriticDihedralWithNonEquiEnc(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=64, initialize=True, N=4, enc='fc'):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        if enc == 'conv':
            self.img_conv = torch.nn.Sequential(
                # 128x128
                torch.nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 64x64
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 32x32
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 16x16
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 8x8
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),

                torch.nn.Conv2d(128, 256, kernel_size=3, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),

                torch.nn.Conv2d(256, 256, kernel_size=3, padding=0),
                torch.nn.ReLU(inplace=True),

                torch.nn.Conv2d(256, n_hidden*8, kernel_size=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
        elif enc == 'fc':
            self.img_conv = torch.nn.Sequential(
                # 128x128
                torch.nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 64x64
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 32x32
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 16x16
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 8x8
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),

                torch.nn.Conv2d(128, 160, kernel_size=3, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),

                torch.nn.Flatten(),
                torch.nn.Linear(160 * 3 * 3, n_hidden*8),
                torch.nn.ReLU(inplace=True),
            )
        elif enc == 'fc_2':
            self.img_conv = torch.nn.Sequential(
                # 128x128
                torch.nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 64x64
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 32x32
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 16x16
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 8x8
                torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),

                torch.nn.Flatten(),
                torch.nn.Linear(64 * 8 * 8, n_hidden*8),
                torch.nn.ReLU(inplace=True),
            )
        self.n_rho1 = 2 if N==2 else 1
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr] +
                                       (action_dim - 3) * [self.d4_act.trivial_repr] +
                                       self.n_rho1 * [self.d4_act.irrep(1, 1)] +
                                       1 * [self.d4_act.quotient_repr((None, 4))]),
                      nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]), inplace=True),
            nn.GroupPooling(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr])),
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.trivial_repr]),
                      nn.FieldType(self.d4_act, 1 * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr] +
                                       (action_dim - 3) * [self.d4_act.trivial_repr] +
                                       self.n_rho1 * [self.d4_act.irrep(1, 1)] +
                                       1 * [self.d4_act.quotient_repr((None, 4))]),
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
        enc_out = self.img_conv(obs)
        enc_out = enc_out.reshape(obs.shape[0], -1, 1, 1)
        enc_out = nn.GeometricTensor(enc_out, nn.FieldType(self.d4_act, self.n_hidden * [self.d4_act.regular_repr]))
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

class EquivariantSACActor(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, backbone=None):
        super().__init__()
        self.obs_channel = obs_shape[0]
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_hidden = n_hidden
        self.N = N
        self.n_rho1 = 2 if self.N==2 else 1
        self.group = self.getGroup()
        if obs_shape[-1] == 128:
            self.img_conv = EquivariantEncoder128(self.group, self.obs_channel, n_hidden, initialize, backbone)
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
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, N=4, backbone=None):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, N, backbone)

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

class EquivariantSACActorFlip(EquivariantSACActor):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, backbone=None):
        super().__init__(obs_shape, action_dim, n_hidden, initialize, 1, backbone)

    def getGroup(self):
        return gspaces.Flip2dOnR2()

    def getOutFieldType(self):
        return nn.FieldType(self.group, 2 * [self.group.irrep(1)] + (self.action_dim * 2 - 2) * [self.group.trivial_repr])

    def getOutput(self, conv_out):
        dy = conv_out[:, 0:1]
        dtheta = conv_out[:, 1:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:2], dy, inv_act[:, 2:3], dtheta), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorDihedralWithNonEquiEnc(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=64, initialize=True, N=4, enc='fc'):
        super().__init__()
        assert obs_shape[1] in [128, 64]
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.d4_act = gspaces.FlipRot2dOnR2(N)
        self.n_rho1 = 2 if N==2 else 1
        self.n_hidden = n_hidden
        if enc == 'conv':
            self.img_conv = torch.nn.Sequential(
                # 128x128
                torch.nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 64x64
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 32x32
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 16x16
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 8x8
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),

                torch.nn.Conv2d(128, 256, kernel_size=3, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),

                torch.nn.Conv2d(256, 256, kernel_size=3, padding=0),
                torch.nn.ReLU(inplace=True),

                torch.nn.Conv2d(256, n_hidden*8, kernel_size=1, padding=0),
                torch.nn.ReLU(inplace=True),
            )
        elif enc == 'fc':
            self.img_conv = torch.nn.Sequential(
                # 128x128
                torch.nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 64x64
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 32x32
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 16x16
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 8x8
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),

                torch.nn.Conv2d(128, 160, kernel_size=3, padding=0),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),

                torch.nn.Flatten(),
                torch.nn.Linear(160 * 3 * 3, n_hidden * 8),
                torch.nn.ReLU(inplace=True),
            )
        elif enc == 'fc_2':
            self.img_conv = torch.nn.Sequential(
                # 128x128
                torch.nn.Conv2d(obs_shape[0], 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 64x64
                torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 32x32
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 16x16
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                # 8x8
                torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),

                torch.nn.Flatten(),
                torch.nn.Linear(64 * 8 * 8, n_hidden * 8),
                torch.nn.ReLU(inplace=True),
            )
        self.conv = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.d4_act, n_hidden * [self.d4_act.regular_repr]),
                      nn.FieldType(self.d4_act, self.n_rho1 * [self.d4_act.irrep(1, 1)] + 1 * [self.d4_act.quotient_repr((None, 4))] + (action_dim * 2 - 3) * [self.d4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        enc_out = self.img_conv(obs)
        enc_out = enc_out.reshape(obs.shape[0], -1, 1, 1)
        enc_out = nn.GeometricTensor(enc_out, nn.FieldType(self.d4_act, self.n_hidden * [self.d4_act.regular_repr]))
        conv_out = self.conv(enc_out).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        dtheta = conv_out[:, 2:3] - conv_out[:, 3:4]
        inv_act = conv_out[:, 4:self.action_dim+1]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:2], dtheta), dim=1)
        log_std = conv_out[:, self.action_dim+1:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    o = torch.zeros(1, 4, 128, 128)
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
    # actor = EquivariantSACActor(obs_shape=(4, 128, 128), action_dim=5, n_hidden=64, initialize=True, N=8)
    # critic = EquivariantSACCritic(obs_shape=(4, 128, 128), action_dim=5, n_hidden=64, initialize=True, N=8)
    actor = EquivariantSACActorDihedralWithNonEquiEnc((4, 128, 128), 5,
                                                      initialize=True, N=4, enc='fc_2', n_hidden=32)
    critic = EquivariantSACCriticDihedralWithNonEquiEnc((4, 128, 128), 5,
                                                        initialize=True, N=4, enc='fc_2', n_hidden=32)
    print(actor(o))
    print(critic(o, a))
    print(1)