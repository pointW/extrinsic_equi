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

class EquivariantEncoder128SO2_1(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = self.c4_act.irrep(0) + self.c4_act.irrep(1) + self.c4_act.irrep(2) + self.c4_act.irrep(3)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//8 * [self.repr]),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//8 * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out//8 * [self.repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.repr]),
                      nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//4 * [self.repr]),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//4 * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out // 4 * [self.repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.repr]),
                      nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//2 * [self.repr]),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out//2 * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out // 2 * [self.repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr]),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out * [self.repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.repr]),
                      nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out*2 * [self.repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out*2 * [self.repr])),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr]),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr])),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out * [self.repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.repr]),
                      nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.c4_act, n_out * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act, n_out * [self.repr])),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantSO2Layer(torch.nn.Module):
    def __init__(self, gs, r1, out_channel, kernel_size, padding, stride, initialize):
        super().__init__()
        self.c4_act = gs
        irreps = []
        for n, irr in self.c4_act.fibergroup.irreps.items():
            if n != self.c4_act.trivial_repr.name:
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
        I = len(irreps)
        # S = nn.FieldType(self.c4_act, irreps).size + 1
        # M = S + I
        trivials = nn.FieldType(self.c4_act, [self.c4_act.trivial_repr] * out_channel)
        gates = nn.FieldType(self.c4_act, [self.c4_act.trivial_repr] * out_channel * I)
        gated = nn.FieldType(self.c4_act, irreps * out_channel).sorted()
        gate = gates + gated
        r2 = trivials + gate
        self.conv = nn.R2Conv(r1, r2, kernel_size=kernel_size, padding=padding, stride=stride, initialize=initialize)
        labels = ["trivial"] * len(trivials) + ["gate"] * len(gate)
        modules = [
            (nn.ELU(trivials), "trivial"),
            (nn.GatedNonLinearity1(gate), "gate")
        ]
        self.nnl = nn.MultipleModule(self.conv.out_type, labels, modules)

    def forward(self, x):
        return self.nnl(self.conv(x))

class EquivariantEncoder128SO2_2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.c4_act.irrep(0), self.c4_act.irrep(1), self.c4_act.irrep(2), self.c4_act.irrep(3)]

        self.conv = torch.nn.Sequential(
            # 128x128
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                                n_out//8, kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out//8 * self.repr).sorted(), 2),
            # 64x64
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out//8 * self.repr).sorted(),
                                n_out//4, kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out//4 * self.repr).sorted(), 2),
            # 32x32
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out // 4 * self.repr).sorted(),
                                n_out // 2, kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out // 2 * self.repr).sorted(), 2),
            # 16x16
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out // 2 * self.repr).sorted(),
                                n_out, kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out * self.repr).sorted(), 2),
            # 8x8
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out * self.repr).sorted(),
                                n_out*2, kernel_size=3, padding=1, stride=1, initialize=initialize),
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out*2 * self.repr).sorted(),
                                n_out, kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.NormMaxPool(nn.FieldType(self.c4_act, n_out * self.repr).sorted(), 2),
            # 3x3
            EquivariantSO2Layer(self.c4_act, nn.FieldType(self.c4_act, n_out * self.repr).sorted(),
                                n_out, kernel_size=3, padding=0, stride=1, initialize=initialize),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128SO2_3(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.so2.irrep(0), self.so2.irrep(1), self.so2.irrep(2), self.so2.irrep(3)]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.so2, obs_channel * [self.so2.trivial_repr]),
                      nn.FieldType(self.so2, n_out // 8 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 8 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.so2, n_out // 8 * self.repr),
                      nn.FieldType(self.so2, n_out // 4 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 4 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.so2, n_out // 4 * self.repr),
                      nn.FieldType(self.so2, n_out // 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out // 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.so2, n_out // 2 * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.so2, n_out * self.repr),
                      nn.FieldType(self.so2, n_out * 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * 2 * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.so2, n_out * 2 * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.so2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.so2, n_out * self.repr),
                      nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.so2, n_out * len(self.repr) * [self.so2.trivial_repr]) + nn.FieldType(self.so2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128O2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.o2, obs_channel * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.o2, n_out // 8 * self.repr),
                      nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.o2, n_out // 4 * self.repr),
                      nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.o2, n_out // 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.o2, n_out * 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128O2_2(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.o2, obs_channel * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.o2, n_out // 8 * self.repr),
                      nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.o2, n_out // 4 * self.repr),
                      nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.o2, n_out // 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.o2, n_out * 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantEncoder128O2_3(torch.nn.Module):
    def __init__(self, obs_channel=2, n_out=128, initialize=True):
        super().__init__()
        self.obs_channel = obs_channel
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3), self.o2.induced_repr((None, -1), self.so2.irrep(0)), self.o2.induced_repr((None, -1), self.so2.irrep(1)), self.o2.induced_repr((None, -1), self.so2.irrep(2)), self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.o2, obs_channel * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 8 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 8 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 8 * self.repr), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.o2, n_out // 8 * self.repr),
                      nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 4 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 4 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 4 * self.repr), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.o2, n_out // 4 * self.repr),
                      nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out // 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out // 2 * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out // 2 * self.repr), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.o2, n_out // 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=1, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * 2 * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * 2 * self.repr)),

            nn.R2Conv(nn.FieldType(self.o2, n_out * 2 * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, stride=1, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            nn.NormMaxPool(nn.FieldType(self.o2, n_out * self.repr), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.o2, n_out * self.repr),
                      nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(nn.FieldType(self.o2, n_out * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2, n_out * self.repr)),
            # 1x1
        )

    def forward(self, geo):
        # geo = nn.GeometricTensor(x, nn.FieldType(self.c4_act, self.obs_channel*[self.c4_act.trivial_repr]))
        return self.conv(geo)

class EquivariantSACCriticSO2_1(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128SO2_1
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        self.repr = self.c4_act.irrep(0) + self.c4_act.irrep(1) + self.c4_act.irrep(2) + self.c4_act.irrep(3)
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.repr] + (action_dim - 2) * [
                self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act,
                                                                                                      n_hidden * [
                                                                                                          self.repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act,
                                                                                                n_hidden * [
                                                                                                    self.repr])),
            nn.NormPool(nn.FieldType(self.c4_act, n_hidden * [self.repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.repr] + (action_dim - 2) * [
                self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1)]),
                      nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act,
                                                                                                      n_hidden * [
                                                                                                          self.repr]),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]) + nn.FieldType(self.c4_act,
                                                                                                n_hidden * [
                                                                                                    self.repr])),
            nn.NormPool(nn.FieldType(self.c4_act, n_hidden * [self.repr])),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat(
            (conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.c4_act, self.n_hidden * [self.repr] + n_inv * [
            self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticSO2_2(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128SO2_2
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        self.repr = [self.c4_act.irrep(0), self.c4_act.irrep(1), self.c4_act.irrep(2), self.c4_act.irrep(3)]
        self.critic_1 = torch.nn.Sequential(
            EquivariantSO2Layer(self.c4_act,
                                nn.FieldType(self.c4_act, n_hidden * self.repr).sorted() + nn.FieldType(self.c4_act,
                                                                                                        (
                                                                                                                    action_dim - 2) * [
                                                                                                            self.c4_act.trivial_repr] + self.n_rho1 * [
                                                                                                            self.c4_act.irrep(
                                                                                                                1)]),
                                n_hidden, kernel_size=1, padding=0, stride=1, initialize=initialize),
            nn.NormPool(nn.FieldType(self.c4_act, n_hidden * self.repr).sorted()),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * 4 * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            EquivariantSO2Layer(self.c4_act,
                                nn.FieldType(self.c4_act, n_hidden * self.repr).sorted() + nn.FieldType(self.c4_act,
                                                                                                        (
                                                                                                                    action_dim - 2) * [
                                                                                                            self.c4_act.trivial_repr] + self.n_rho1 * [
                                                                                                            self.c4_act.irrep(
                                                                                                                1)]),
                                n_hidden, kernel_size=1, padding=0, stride=1, initialize=initialize),
            nn.NormPool(nn.FieldType(self.c4_act, n_hidden * self.repr).sorted()),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * 4 * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat(
            (conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat,
                                     nn.FieldType(self.c4_act, self.n_hidden * self.repr).sorted() + nn.FieldType(
                                         self.c4_act,
                                         n_inv * [self.c4_act.trivial_repr] + self.n_rho1 * [self.c4_act.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticSO2_3(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128SO2_3
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        self.repr = [self.so2.irrep(0), self.so2.irrep(1), self.so2.irrep(2), self.so2.irrep(3)]
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.so2, n_hidden * self.repr + (action_dim - 2) * [
                self.so2.trivial_repr] + self.n_rho1 * [self.so2.irrep(1)]),
                      nn.FieldType(self.so2, n_hidden * 4 * [self.so2.trivial_repr]) + nn.FieldType(self.so2,
                                                                                                    n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.so2, n_hidden * 4 * [self.so2.trivial_repr]) + nn.FieldType(self.so2,
                                                                                              n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.so2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.so2, n_hidden * 4 * [self.so2.trivial_repr]),
                      nn.FieldType(self.so2, 1 * [self.so2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.so2, n_hidden * self.repr + (action_dim - 2) * [
                self.so2.trivial_repr] + self.n_rho1 * [self.so2.irrep(1)]),
                      nn.FieldType(self.so2, n_hidden * 4 * [self.so2.trivial_repr]) + nn.FieldType(self.so2,
                                                                                                    n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.so2, n_hidden * 4 * [self.so2.trivial_repr]) + nn.FieldType(self.so2,
                                                                                              n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.so2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.so2, n_hidden * 4 * [self.so2.trivial_repr]),
                      nn.FieldType(self.so2, 1 * [self.so2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.so2, self.obs_channel * [self.so2.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat(
            (conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.so2, self.n_hidden * self.repr + n_inv * [
            self.so2.trivial_repr] + self.n_rho1 * [self.so2.irrep(1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticO2(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128O2
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.induced_repr((None, -1), self.so2.irrep(0)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(1)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(2)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [
                self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(
                          self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2,
                                                                                                         n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [
                self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(
                          self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2,
                                                                                                         n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat(
            (conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.o2, self.n_hidden * self.repr + n_inv * [
            self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticO2_2(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128O2_2
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2),
                     self.o2.irrep(1, 3)]
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [
                self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(
                          self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2,
                                                                                                         n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [
                self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(
                          self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2,
                                                                                                         n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat(
            (conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.o2, self.n_hidden * self.repr + n_inv * [
            self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACCriticO2_3(torch.nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        enc = EquivariantEncoder128O2_3
        self.img_conv = enc(self.obs_channel, n_hidden, initialize)
        self.n_rho1 = 1
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2),
                     self.o2.irrep(1, 3), self.o2.induced_repr((None, -1), self.so2.irrep(0)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(1)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(2)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.critic_1 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [
                self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(
                          self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2,
                                                                                                         n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

        self.critic_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr + (action_dim - 2) * [
                self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]),
                      nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(
                          self.o2, n_hidden * self.repr),
                      kernel_size=1, padding=0, initialize=initialize),
            nn.GatedNonLinearity1(
                nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]) + nn.FieldType(self.o2,
                                                                                                         n_hidden * self.repr)),
            nn.NormPool(nn.FieldType(self.o2, n_hidden * self.repr)),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * len(self.repr) * [self.o2.trivial_repr]),
                      nn.FieldType(self.o2, 1 * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize),
        )

    def forward(self, obs, act):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.img_conv(obs_geo)
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]
        # dxy_geo = nn.GeometricTensor(dxy.reshape(batch_size, 2, 1, 1), nn.FieldType(self.c4_act, 1*[self.c4_act.irrep(1)]))
        # inv_act_geo = nn.GeometricTensor(inv_act.reshape(batch_size, n_inv, 1, 1), nn.FieldType(self.c4_act, n_inv*[self.c4_act.trivial_repr]))
        cat = torch.cat(
            (conv_out.tensor, inv_act.reshape(batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        cat_geo = nn.GeometricTensor(cat, nn.FieldType(self.o2, self.n_hidden * self.repr + n_inv * [
            self.o2.trivial_repr] + self.n_rho1 * [self.o2.irrep(1, 1)]))
        out1 = self.critic_1(cat_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic_2(cat_geo).tensor.reshape(batch_size, 1)
        return out1, out2

class EquivariantSACActorSO2_1(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128SO2_1
        self.repr = self.c4_act.irrep(0) + self.c4_act.irrep(1) + self.c4_act.irrep(2) + self.c4_act.irrep(3)
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim * 2 - 2) * [
                          self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorSO2_2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128SO2_2
        self.repr = [self.c4_act.irrep(0), self.c4_act.irrep(1), self.c4_act.irrep(2), self.c4_act.irrep(3)]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * self.repr).sorted(),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim * 2 - 2) * [
                          self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorSO2_3(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128SO2_3
        self.repr = [self.c4_act.irrep(0), self.c4_act.irrep(1), self.c4_act.irrep(2), self.c4_act.irrep(3)]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * self.repr),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim * 2 - 2) * [
                          self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorO2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128O2
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.induced_repr((None, -1), self.so2.irrep(0)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(1)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(2)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr),
                      nn.FieldType(self.o2, self.n_rho1 * [self.o2.irrep(1, 1)] + (action_dim * 2 - 2) * [
                          self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorO2_2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128O2_2
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2),
                     self.o2.irrep(1, 3)]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr),
                      nn.FieldType(self.o2, self.n_rho1 * [self.o2.irrep(1, 1)] + (action_dim * 2 - 2) * [
                          self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantSACActorO2_3(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128O2_3
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.irrep(0, 0), self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2),
                     self.o2.irrep(1, 3), self.o2.induced_repr((None, -1), self.so2.irrep(0)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(1)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(2)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr),
                      nn.FieldType(self.o2, self.n_rho1 * [self.o2.irrep(1, 1)] + (action_dim * 2 - 2) * [
                          self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        log_std = conv_out[:, self.action_dim:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class EquivariantPolicySO2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.c4_act = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128SO2_1
        self.repr = self.c4_act.irrep(0) + self.c4_act.irrep(1) + self.c4_act.irrep(2) + self.c4_act.irrep(3)
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.c4_act, n_hidden * [self.repr]),
                      nn.FieldType(self.c4_act, self.n_rho1 * [self.c4_act.irrep(1)] + (action_dim - 2) * [
                          self.c4_act.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.c4_act, self.obs_channel * [self.c4_act.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        return mean

class EquivariantPolicyO2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, n_hidden=128, initialize=True, kernel_size=3):
        super().__init__()
        assert obs_shape[1] == 128
        assert kernel_size == 3
        self.obs_channel = obs_shape[0]
        self.action_dim = action_dim
        self.o2 = gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=3)
        self.so2 = gspaces.Rot2dOnR2(N=-1, maximum_frequency=3)
        self.n_rho1 = 1
        enc = EquivariantEncoder128O2
        # self.repr = [self.o2.irrep(1, 0), self.o2.irrep(1, 1), self.o2.irrep(1, 2), self.o2.irrep(1, 3)]
        self.repr = [self.o2.induced_repr((None, -1), self.so2.irrep(0)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(1)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(2)),
                     self.o2.induced_repr((None, -1), self.so2.irrep(3))]
        self.conv = torch.nn.Sequential(
            enc(self.obs_channel, n_hidden, initialize),
            nn.R2Conv(nn.FieldType(self.o2, n_hidden * self.repr),
                      nn.FieldType(self.o2,
                                   self.n_rho1 * [self.o2.irrep(1, 1)] + (action_dim - 2) * [self.o2.trivial_repr]),
                      kernel_size=1, padding=0, initialize=initialize)
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.o2, self.obs_channel * [self.o2.trivial_repr]))
        conv_out = self.conv(obs_geo).tensor.reshape(batch_size, -1)
        dxy = conv_out[:, 0:2]
        inv_act = conv_out[:, 2:self.action_dim]
        mean = torch.cat((inv_act[:, 0:1], dxy, inv_act[:, 1:]), dim=1)
        return mean