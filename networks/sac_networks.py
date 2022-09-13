import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import torch_utils
from networks.ssm import SpatialSoftArgmax

from vit_pytorch import SimpleViT, ViT
from networks.vit import VisionTransformer

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
    def __init__(self, obs_shape=(2, 128, 128), out_dim=1024, ssm=False):
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

        if ssm:
            self.fc = torch.nn.Sequential(
                SpatialSoftArgmax(),
                torch.nn.Linear(512 * 2, out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.fc = torch.nn.Sequential(
                nn.Flatten(),
                torch.nn.Linear(256 * 3 * 3, out_dim),
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

class SACCritic(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, ssm=False):
        super().__init__()
        self.state_conv_1 = SACEncoder(obs_shape, 1024, ssm)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(1024+action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(1024+action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act), dim=1))
        return out_1, out_2

class SACCriticSimFC(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, ssm=False):
        super().__init__()
        self.state_conv_1 = SACEncoderSimFC(obs_shape, 256, ssm)

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
        conv_out = self.state_conv_1(obs)
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

class SACCriticViT(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.state_conv_1 = torch.nn.Sequential(
            # ViT(channels=obs_shape[0], image_size=obs_shape[-1], patch_size=8, num_classes=1024,
            #     dim=128, depth=4, heads=8, mlp_dim=128),
            VisionTransformer(img_size=64, patch_size=4, in_chans=5, embed_dim=128, depth=4, num_heads=8,
                              mlp_ratio=1., qkv_bias=False),
            torch.nn.ReLU(inplace=True),
        )

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128 + action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(128 + action_dim, 512),
            nn.ReLU(inplace=True),
            torch.nn.Linear(512, 1)
        )

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act), dim=1))
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

class SACGaussianPolicy(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, ssm=False):
        super().__init__()
        self.conv = SACEncoder(obs_shape, 1024, ssm)
        self.mean_linear = nn.Linear(1024, action_dim)
        self.log_std_linear = nn.Linear(1024, action_dim)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class SACGaussianPolicySimFC(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5, ssm=False):
        super().__init__()
        self.conv = SACEncoderSimFC(obs_shape, 256, ssm)
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

class SACGaussianPolicyViT(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = torch.nn.Sequential(
            VisionTransformer(img_size=64, patch_size=4, in_chans=5, embed_dim=128, depth=4, num_heads=8,
                              mlp_ratio=1., qkv_bias=False),
            torch.nn.ReLU(inplace=True),
        )

        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class SACDeterministicPolicy(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super(SACDeterministicPolicy, self).__init__()
        self.conv = SACEncoder(obs_shape, 1024)
        self.mean = nn.Linear(1024, action_dim)
        self.noise = torch.Tensor(action_dim)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = torch.tanh(self.mean(x))
        return mean

    def sample(self, x):
        mean = self.forward(x)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean


# similar amount of parameters
class SACEncoder2(nn.Module):
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
            nn.Conv2d(256, out_dim, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.conv(x)

# similar amount of parameters
class SACCritic2(nn.Module):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.state_conv_1 = SACEncoder2(obs_shape, 128)

        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(128+action_dim, 128),
            nn.ReLU(inplace=True),
            torch.nn.Linear(128, 1)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        conv_out = self.state_conv_1(obs)
        out_1 = self.critic_fc_1(torch.cat((conv_out, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((conv_out, act), dim=1))
        return out_1, out_2

# similar amount of parameters
class SACGaussianPolicy2(SACGaussianPolicyBase):
    def __init__(self, obs_shape=(2, 128, 128), action_dim=5):
        super().__init__()
        self.conv = SACEncoder2(obs_shape, 128)
        self.mean_linear = nn.Linear(128, action_dim)
        self.log_std_linear = nn.Linear(128, action_dim)

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        x = self.conv(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

class SACVecCritic(nn.Module):
    def __init__(self, obs_dim=7, action_dim=5):
        super().__init__()
        # Q1
        self.critic_fc_1 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim+action_dim, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1)
        )

        # Q2
        self.critic_fc_2 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim+action_dim, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1)
        )
        self.apply(torch_utils.weights_init)

    def forward(self, obs, act):
        out_1 = self.critic_fc_1(torch.cat((obs, act), dim=1))
        out_2 = self.critic_fc_2(torch.cat((obs, act), dim=1))
        return out_1, out_2

class SACVecGaussianPolicy(SACGaussianPolicyBase):
    def __init__(self, obs_dim=7, action_dim=5):
        super().__init__()
        self.action_dim = action_dim
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 2 * action_dim)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, x):
        out = self.fc(x)
        mean = out[:, :self.action_dim]
        log_std = out[:, self.action_dim:]
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

if __name__ == '__main__':
    actor = SACGaussianPolicySimFC(obs_shape=(4, 128, 128))
    critic = SACCriticSimFC(obs_shape=(4, 128, 128))
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))
    print(sum(p.numel() for p in critic.parameters() if p.requires_grad))
    print(1)