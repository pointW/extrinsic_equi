import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import torch_utils
from networks.ssm import SpatialSoftArgmax

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class SACEncoder(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), out_dim=1024, ssm=False):
        super().__init__()
        if obs_shape[1] == 128:
            self.conv = torch.nn.Sequential(
                # 128x128
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1),
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
            )
        else:
            self.conv = torch.nn.Sequential(
                # 64x64
                nn.Conv2d(obs_shape[0], 64, kernel_size=3, padding=1),
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
            )

        if ssm:
            self.fc = torch.nn.Sequential(
                SpatialSoftArgmax(),
                torch.nn.Linear(512 * 2, out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.fc = torch.nn.Sequential(
                nn.Flatten(),
                torch.nn.Linear(512 * 8 * 8, out_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.fc(self.conv(x))

class SACEncoderSimFC(nn.Module):
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
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.fc = torch.nn.Sequential(
            nn.Flatten(),
            torch.nn.Linear(256 * 3 * 3, out_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.fc(self.conv(x))

class SACEncoderSimFC2(nn.Module):
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
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc = torch.nn.Sequential(
            nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, out_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.fc(self.conv(x))

class SACEncoderFullyConv(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), out_dim=1024):
        super().__init__()
        if obs_shape[1] == 128:
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
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=3, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(256, out_dim, kernel_size=3, padding=0),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = torch.nn.Sequential(
                # 64x64
                nn.Conv2d(obs_shape[0], 64, kernel_size=3, padding=1),
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

                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(1024, 512, kernel_size=3, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(512, 512, kernel_size=3, padding=0),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.conv(x)

class SACCriticSimFC(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, enc_id=1):
        super().__init__()
        if enc_id == 1:
            self.conv = SACEncoderSimFC(obs_shape, 256)
        elif enc_id == 2:
            self.conv = SACEncoderSimFC2(obs_shape, 256)
        else:
            raise NotImplementedError

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(256+action_dim, 256),
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(256+action_dim, 256),
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        conv_out = self.conv(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act), dim=1))
        return out_1, out_2

class SACCriticFullyConv(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.state_conv_1 = SACEncoderFullyConv(obs_shape, 256)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            nn.Conv2d(256 + action_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            nn.Conv2d(256 + action_dim, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act.reshape(act.shape[0], act.shape[1], 1, 1)), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act.reshape(act.shape[0], act.shape[1], 1, 1)), dim=1))
        out_1 = out_1.reshape(obs.shape[0], -1)
        out_2 = out_2.reshape(obs.shape[0], -1)
        return out_1, out_2

class SACGaussianPolicyBase(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

class SACGaussianPolicySimFC(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, enc_id=1):
        super().__init__()
        if enc_id == 1:
            self.conv = SACEncoderSimFC(obs_shape, 256)
        elif enc_id == 2:
            self.conv = SACEncoderSimFC2(obs_shape, 256)
        else:
            raise NotImplementedError
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class SACGaussianPolicyFullyConv(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = torch.nn.Sequential(
            SACEncoderFullyConv(obs_shape, 256),
            nn.Conv2d(256, action_dim*2, kernel_size=1),
        )

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        mean = x[:, :x.shape[1]//2]
        log_std = x[:, x.shape[1]//2:]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

if __name__ == '__main__':
    actor = SACGaussianPolicySimFC(obs_shape=(4, 128, 128), enc_id=2)
    critic = SACCriticSimFC(obs_shape=(4, 128, 128), enc_id=2)
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
    print(sum(p.numel() for p in critic.parameters() if p.requires_grad))
    print(1)