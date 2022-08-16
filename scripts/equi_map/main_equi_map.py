#  MIT License
#
#  Copyright (c) 2022 Dian Wang
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import copy
import collections
from tqdm import tqdm, trange
import os
import sys
sys.path.append('../..')
import matplotlib.pyplot as plt

from utils.create_agent import createAgent
from utils.parameters import *
from utils.env_wrapper import EnvWrapper
from utils.logger import Logger

import torch
import torch.nn.functional as F
from e2cnn import gspaces, nn
from utils.torch_utils import centerCrop
from networks.equivariant_sac_net import EquivariantEncoder64
from networks.sac_networks import SACEncoder

def collectData():
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)

    n_true = 8
    n_data = 1000
    buffer = []
    pbar = tqdm(total=n_data)
    while len(buffer) < n_data:
        states = []
        true_obs = []
        for _ in range(n_true):
            state, obs = envs.reset()
            true_obs.append(obs)
            states.append(state)

        d4 = gspaces.FlipRot2dOnR2(4)
        # d4 = gspaces.Rot2dOnR2(8)
        o0 = true_obs[0]
        state_tile = states[0].reshape(num_processes, 1, 1, 1).repeat(1, 1, o0.shape[2], o0.shape[3])
        o0 = torch.cat([o0, state_tile], dim=1)

        o0 = nn.GeometricTensor(o0, nn.FieldType(d4, 5 * [d4.trivial_repr]))

        trans_obs = []
        for tran in d4.testing_elements:
            trans_obs.append(o0.transform(tran).tensor)

        states = torch.stack(states, dim=1)
        true_obs = torch.stack(true_obs, dim=1)
        trans_obs = torch.stack(trans_obs, dim=1)
        state_tile = states.reshape(states.size(0), states.size(1), 1, 1, 1).repeat(1, 1, 1, true_obs.shape[-2], true_obs.shape[-1])
        true_obs = torch.cat([true_obs, state_tile], dim=2)

        for i in range(num_processes):
            buffer.append((true_obs[i], trans_obs[i]))
            pbar.update(1)
            if len(buffer) >= n_data:
                break

    torch.save(buffer, '{}_{}.pt'.format(env, n_data))

class ContrastiveModel(torch.nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.rand(z_dim, z_dim))

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

def calcLossBatch(network, contrastive, true_obs, trans_obs):
    losses = []
    for i in range(true_obs.shape[0]):
        true = true_obs[i]
        trans = trans_obs[i]
        # permutation = torch.rand(8).argsort().to(device)
        # true = true[permutation]
        # trans = trans[permutation]
        z_true = network(true)
        z_trans = network(trans)
        if type(z_true) == nn.GeometricTensor:
            z_true = z_true.tensor.squeeze(2).squeeze(2)
        if type(z_trans) == nn.GeometricTensor:
            z_trans = z_trans.tensor.squeeze(2).squeeze(2)
        logits = contrastive.compute_logits(z_true, z_trans)
        labels = torch.arange(logits.shape[0]).long().to(device)
        loss = F.cross_entropy(logits, labels)
        losses.append(loss)
    return torch.stack(losses).mean()

def train():
    max_epochs = 50
    holdout_ratio = 0.1
    pbar = tqdm(total=max_epochs)

    if model == 'equi_d':
        group = gspaces.FlipRot2dOnR2(4)
        network = EquivariantEncoder64(group, 5, 64).to(device)
    elif model == 'cnn_ssm':
        network = SACEncoder(obs_shape=(5, 64, 64), out_dim=64*8, ssm=True).to(device)
    else:
        raise NotImplementedError
    contrastive = ContrastiveModel(64 * 8).to(device)
    optimizer = torch.optim.Adam(list(network.parameters()) + list(contrastive.parameters()), lr=1e-3)

    log_dir = os.path.join(log_pre, 'equi_map_{}'.format(model))
    logger = Logger(log_dir, env, 'train', num_processes, max_train_step, gamma, log_sub)

    data = torch.load('{}_1000.pt'.format(env))
    num_holdout = int(len(data) * holdout_ratio)
    true_obss, trans_obss = zip(*data)
    true_obss = torch.stack(true_obss)
    trans_obss = torch.stack(trans_obss)
    data = torch.stack([true_obss, trans_obss], dim=1)
    data_permutation = np.random.permutation(len(data))
    data = data[data_permutation]
    train_data = data[num_holdout:]
    holdout_data = data[:num_holdout]
    for epoch in range(1, max_epochs+1):
        train_idx = np.random.permutation(train_data.shape[0])
        for start_pos in tqdm(range(0, train_data.shape[0], batch_size)):
            idx = train_idx[start_pos: start_pos + batch_size]
            batch = train_data[idx]
            true_obs = batch[:, 0].to(device)
            trans_obs = batch[:, 1].to(device)
            loss = calcLossBatch(network, contrastive, true_obs, trans_obs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.model_losses.append(loss.item())

        with torch.no_grad():
            holdout_true = holdout_data[:, 0].to(device)
            holdout_trans = holdout_data[:, 1].to(device)
            holdout_loss = calcLossBatch(network, contrastive, holdout_true, holdout_trans)
        logger.model_holdout_losses.append(holdout_loss.item())
        logger.saveModelLosses()
        logger.saveModelLossCurve()
        logger.saveModelHoldoutLossCurve()
        if epoch % 10 == 0:
            torch.save(network.state_dict(), os.path.join(logger.models_dir, 'model_{}.pt'.format(epoch)))
            # torch.save(contrastive.state_dict(), os.path.join(logger.models_dir, 'contrastive_{}.pt'.format(epoch)))
        pbar.set_description('epoch: {}, holdout loss: {:.03f}'.format(epoch, holdout_loss.item()))
        pbar.update()

if __name__ == '__main__':
    # collectData()
    from scripts.main import set_seed
    global seed
    for i in range(4):
        set_seed(i)
        train()

