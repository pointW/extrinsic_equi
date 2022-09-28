# This file has been created to use equivariant versions
# of the original networks


from functools import partial

import numpy as np

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import nn as esnn
from torch.cuda.amp import GradScaler, autocast

import utils
from drqv2 import RandomShiftsAug


class EquiEncoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, out_dim, gspace, device):
        super().__init__()

        assert len(obs_shape) == 3

        self.gspace = gspace
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.repr_shape = (out_dim, 7, 7)
        self.repr_dim = np.prod(self.repr_shape)
        self.device = device

        self.in_type = esnn.FieldType(
            self.gspace, obs_shape[0] * [self.gspace.trivial_repr]
        )
        self.hid_type = esnn.FieldType(
            self.gspace, hidden_dim * [self.gspace.regular_repr]
        )
        self.out_type = esnn.FieldType(
            self.gspace, out_dim * [self.gspace.regular_repr]
        )
        self.flat_type = esnn.FieldType(
            self.gspace,
            self.gspace.fibergroup.order() * self.repr_dim * [self.gspace.regular_repr],
        )
        self.flat_out_type = esnn.FieldType(
            self.gspace, self.repr_dim * [self.gspace.regular_repr]
        )

        self.convnet = esnn.SequentialModule(
            # 85x85
            esnn.R2Conv(self.in_type, self.hid_type, 3, stride=2),
            esnn.ReLU(self.hid_type),
            # 42x42
            esnn.R2Conv(self.hid_type, self.hid_type, 3, stride=1),
            esnn.ReLU(self.hid_type),
            # 40x40
            esnn.PointwiseMaxPool(self.hid_type, 2),
            # 20x20
            esnn.R2Conv(self.hid_type, self.hid_type, 3, stride=1),
            esnn.ReLU(self.hid_type),
            # 18x18
            esnn.PointwiseMaxPool(self.hid_type, 2),
            # 9x9
            esnn.R2Conv(self.hid_type, self.out_type, 3, stride=1),
            esnn.ReLU(self.out_type),
            # 7x7
        )

        self.last = esnn.SequentialModule(
            esnn.R2Conv(self.flat_type, self.flat_out_type, 1, bias=False),
            esnn.ReLU(self.flat_out_type),
        )

        self.basespace_transforms = self.precompute()

    def precompute(self):
        # Precalculate inverse of basespace transforms
        if self.gspace.fibergroup.name == "C2":
            basespace_transforms = [
                torch.nn.Identity(),  # inverse of (0)
                partial(torch.flip, dims=(-1,)),  # inverse of (1)
            ]
        elif self.gspace.fibergroup.name == "D2":
            basespace_transforms = [
                torch.nn.Identity(),  # inverse of (0, 0)
                partial(torch.rot90, k=2, dims=(-2, -1)),  # inverse of (0, 1)
                partial(torch.flip, dims=(-1,)),  # inverse of (1, 0)
                partial(torch.flip, dims=(-2,)),  # inverse of (1, 1)
            ]
        else:
            raise NotImplementedError("only implemented for groups C2,D2")

        return basespace_transforms

    def forward(self, obs):
        # Need odd-sized inputs for stride=2 to preserve equivariance
        if isinstance(obs, esnn.GeometricTensor):
            inp = obs.tensor
        else:
            inp = obs
        inp = inp / 255.0 - 0.5
        inp = esnn.GeometricTensor(inp, self.in_type)
        h = self.convnet(inp)

        h = self.restrict_functor(h)

        h = self.last(h)

        return h.tensor

    def diff_transform(self, input, element, basespace_transform):
        input_tensor = input.tensor

        # Fibergroup
        representation = input.type.fiber_representation(element).to(
            dtype=input_tensor.dtype, device=input_tensor.device
        )
        output = torch.einsum(
            "oi,bi...->bo...", representation, input_tensor
        ).contiguous()

        # Basespace
        output = basespace_transform(output)

        return output

    def restrict_functor(self, h):
        ginv_x = []
        for g, bs_trans in zip(self.gspace.testing_elements, self.basespace_transforms):

            # Need to use g_inverse
            val = self.diff_transform(h, ~g, bs_trans)
            ginv_x.append(val)

        ginv_x = torch.stack(ginv_x, dim=-1)

        ginv_x = ginv_x.flatten(start_dim=1).view(h.shape[0], -1, 1, 1)
        ginv_x = esnn.GeometricTensor(ginv_x, self.flat_type)

        return ginv_x


class EquiActor(nn.Module):
    def __init__(
        self,
        repr_dim,
        action_shape,
        feature_dim,
        hidden_dim,
        gspace,
        device,
        cup_catch=False,
    ):
        super().__init__()

        self.gspace = gspace
        self.device = device
        self.cup_catch = cup_catch

        self.in_type = esnn.FieldType(
            self.gspace, repr_dim * [self.gspace.regular_repr]
        )
        self.feat_type = esnn.FieldType(
            self.gspace, feature_dim * [self.gspace.regular_repr]
        )
        self.hid_type = esnn.FieldType(
            self.gspace, hidden_dim * [self.gspace.regular_repr]
        )
        if self.gspace.fibergroup.name == "C2":
            if self.cup_catch:
                # Cup catch
                self.out_type = esnn.FieldType(
                    self.gspace, [self.gspace.regular_repr] + [self.gspace.trivial_repr]
                )
                # To get [a,b,c] -> [a - b, c]
                self.register_buffer(
                    "sign_mat",
                    torch.tensor([[1, 0], [-1, 0], [0, 1]], dtype=torch.float),
                )
            else:
                # CartPole/Pendulum/Acrobot
                self.out_type = esnn.FieldType(
                    self.gspace,
                    action_shape[0] * [self.gspace.regular_repr],
                )
                # To get [a,b] -> [a - b]
                self.register_buffer(
                    "sign_mat", torch.tensor([[1], [-1]], dtype=torch.float)
                )
        elif self.gspace.fibergroup.name == "D2":
            # Reacher
            self.out_type = esnn.FieldType(
                self.gspace,
                action_shape[0]
                * [self.gspace.quotient_repr((None, self.gspace.rotations_order))],
            )
            # To get [a,b] -> [a - b]
            self.register_buffer(
                "sign_mat", torch.tensor([[1], [-1]], dtype=torch.float)
            )
        else:
            raise NotImplementedError("only implemented for groups C2,D2")

        # Unflatten obs
        self.first = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Unflatten(dim=-1, unflattened_size=(-1, 1, 1)),
        )

        # 1x1 convs
        self.trunk = esnn.SequentialModule(
            esnn.R2Conv(self.in_type, self.feat_type, 1),
            esnn.InnerBatchNorm(self.feat_type),
            esnn.PointwiseNonLinearity(self.feat_type, "p_tanh"),
        )

        self.policy = esnn.SequentialModule(
            esnn.R2Conv(self.feat_type, self.hid_type, 1),
            esnn.ReLU(self.hid_type, inplace=True),
            esnn.R2Conv(self.hid_type, self.hid_type, 1),
            esnn.ReLU(self.hid_type, inplace=True),
            esnn.R2Conv(self.hid_type, self.out_type, 1),
            # esnn.PointwiseNonLinearity(self.out_type, "p_tanh"),
        )

        # Flatten output
        self.last = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, obs, std):
        # Flatten everything into channel dim and unflatten
        if isinstance(obs, esnn.GeometricTensor):
            inp = obs.tensor
        else:
            inp = obs
        inp = self.first(inp)
        inp = esnn.GeometricTensor(inp, self.in_type)

        h = self.trunk(inp)
        mu_reg = self.policy(h)

        # Flatten everything into channel dim
        mu_reg = self.last(mu_reg.tensor)
        mu_reg = torch.tanh(mu_reg)

        if not self.cup_catch:
            mu_reg = mu_reg.view(obs.shape[0], -1, 2)

        # Quotient/regular to sign repr
        mu = mu_reg @ self.sign_mat
        mu = mu.squeeze(-1)

        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class EquiCritic(nn.Module):
    def __init__(
        self, repr_dim, action_shape, feature_dim, hidden_dim, gspace, cup_catch=False
    ):
        super().__init__()

        self.gspace = gspace
        self.cup_catch = cup_catch

        self.in_type = esnn.FieldType(
            self.gspace, repr_dim * [self.gspace.regular_repr]
        )
        self.feat_type = esnn.FieldType(
            self.gspace, feature_dim * [self.gspace.regular_repr]
        )
        if self.gspace.fibergroup.name == "C2":
            if self.cup_catch:
                # Cup catch
                self.act_type = esnn.FieldType(
                    self.gspace, [self.gspace.regular_repr] + [self.gspace.trivial_repr]
                )
                # To get [a,b] -> [a, -a, b]
                self.register_buffer(
                    "sign_mat", torch.tensor([[1, -1, 0], [0, 0, 1]], dtype=torch.float)
                )
            else:
                # CartPole/Acrobot
                self.act_type = esnn.FieldType(
                    self.gspace,
                    action_shape[0] * [self.gspace.regular_repr],
                )

        elif self.gspace.fibergroup.name == "D2":
            # Reacher
            self.act_type = esnn.FieldType(
                self.gspace,
                action_shape[0]
                * [self.gspace.quotient_repr((None, self.gspace.rotations_order))],
            )
        else:
            raise NotImplementedError("only implemented for groups C2,D2")

        self.hid_type = esnn.FieldType(
            self.gspace, hidden_dim * [self.gspace.regular_repr]
        )
        self.out_type = esnn.FieldType(
            self.gspace,
            1 * [self.gspace.trivial_repr],
        )

        # Unflatten obs
        self.first = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Unflatten(dim=-1, unflattened_size=(-1, 1, 1)),
        )

        self.trunk = esnn.SequentialModule(
            esnn.R2Conv(self.in_type, self.feat_type, 1),
            esnn.InnerBatchNorm(self.feat_type),
            esnn.PointwiseNonLinearity(self.feat_type, "p_tanh"),
        )

        self.Q1 = esnn.SequentialModule(
            esnn.R2Conv(self.feat_type + self.act_type, self.hid_type, 1),
            esnn.ReLU(self.hid_type, inplace=True),
            esnn.R2Conv(self.hid_type, self.hid_type, 1),
            esnn.ReLU(self.hid_type, inplace=True),
            esnn.R2Conv(self.hid_type, self.out_type, 1),
        )

        self.Q2 = esnn.SequentialModule(
            esnn.R2Conv(self.feat_type + self.act_type, self.hid_type, 1),
            esnn.ReLU(self.hid_type, inplace=True),
            esnn.R2Conv(self.hid_type, self.hid_type, 1),
            esnn.ReLU(self.hid_type, inplace=True),
            esnn.R2Conv(self.hid_type, self.out_type, 1),
        )

        self.last = nn.Flatten(start_dim=1)

    def forward(self, obs, action):
        # Flatten everything into channel dim and unflatten
        if isinstance(obs, esnn.GeometricTensor):
            inp = obs.tensor
        else:
            inp = obs
        inp = self.first(inp)
        inp = esnn.GeometricTensor(inp, self.in_type)

        h = self.trunk(inp)

        # Sign to quotient/regular repr
        if self.cup_catch:
            act = action @ self.sign_mat
        else:
            act = torch.stack([action, -action], dim=-1)
        act = self.first(act)

        # Convert action to GeometricTensor
        act = esnn.GeometricTensor(act, self.act_type)
        h_action = esnn.tensor_directsum([h, act])

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        q1 = self.last(q1.tensor)
        q2 = self.last(q2.tensor)

        return q1, q2


class EquiDrQV2Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        encoder_hidden_dim,
        encoder_out_dim,
        gspace,
        mixed_precision,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = EquiEncoder(
            obs_shape, encoder_hidden_dim, encoder_out_dim, gspace, device
        ).to(device)
        self.actor = EquiActor(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, gspace, device
        ).to(device)

        self.critic = EquiCritic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, gspace
        ).to(device)
        self.critic_target = EquiCritic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, gspace
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # scaler for mixed precision
        self.scaler = GradScaler()
        self.mixed_precision = mixed_precision

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        self.training = False
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        # with autocast(enabled=self.mixed_precision):
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        # self.scaler.scale(critic_loss).backward()
        # self.scaler.step(self.critic_opt)
        # self.scaler.step(self.encoder_opt)

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        # with autocast(enabled=self.mixed_precision):
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)

        actor_loss.backward()
        self.actor_opt.step()

        # self.scaler.scale(actor_loss).backward()
        # self.scaler.step(self.actor_opt)

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        # self.scaler.update()

        return metrics

    def save(self):
        save_dict = dict()
        self.eval()
        self.critic_target.eval()

        # model
        save_dict["agent.encoder"] = self.encoder.state_dict()
        save_dict["agent.actor"] = self.actor.state_dict()
        save_dict["agent.critic"] = self.critic.state_dict()
        save_dict["agent.critic_target"] = self.critic_target.state_dict()

        # optimizers
        save_dict["agent.encoder_opt"] = self.encoder_opt.state_dict()
        save_dict["agent.actor_opt"] = self.actor_opt.state_dict()
        save_dict["agent.critic_opt"] = self.critic_opt.state_dict()

        self.train()
        self.critic_target.train()

        return save_dict

    def load(self, state_dict):
        self.eval()
        self.critic_target.eval()

        # model
        self.encoder.load_state_dict(state_dict["agent.encoder"])
        self.actor.load_state_dict(state_dict["agent.actor"])
        self.critic.load_state_dict(state_dict["agent.critic"])
        self.critic_target.load_state_dict(state_dict["agent.critic_target"])

        # optimizers
        self.encoder_opt.load_state_dict(state_dict["agent.encoder_opt"])
        self.actor_opt.load_state_dict(state_dict["agent.actor_opt"])
        self.critic_opt.load_state_dict(state_dict["agent.critic_opt"])

        self.encoder.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.train()
        self.critic_target.train()
