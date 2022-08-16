import copy
import collections
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils.create_agent import createAgent
from utils.parameters import *
from utils.env_wrapper import EnvWrapper

import torch
from e2cnn import gspaces, nn
from networks.equivariant_sac_net import EquivariantEncoder64
from networks.sac_networks import SACEncoder

def calcErrorMatrix(t0: torch.Tensor, t1: torch.Tensor):
    error = np.zeros((t0.shape[0], t1.shape[0]))
    for i in range(t0.shape[0]):
        e = torch.abs(t0[i:i+1] - t1)
        e = e.reshape(e.shape[0], -1).sum(1)
        error[i] = e.cpu().numpy()
    return error

def test(network, envs):
    n_true = 8

    states = []
    true_obs = []
    for i in range(n_true):
        state, obs = envs.reset()
        true_obs.append(obs)
        states.append(state)

    d4 = gspaces.FlipRot2dOnR2(4)
    # d4 = gspaces.Rot2dOnR2(8)
    o0 = true_obs[0]
    state_tile = states[0].reshape(1, 1, 1, 1).repeat(1, 1, o0.shape[2], o0.shape[3])
    o0 = torch.cat([o0, state_tile], dim=1)

    o0 = nn.GeometricTensor(o0, nn.FieldType(d4, 5 * [d4.trivial_repr]))

    trans_obs = []
    for tran in d4.testing_elements:
        trans_obs.append(o0.transform(tran).tensor)

    states = torch.cat(states)
    true_obs = torch.cat(true_obs)
    trans_obs = torch.cat(trans_obs)
    state_tile = states.reshape(states.size(0), 1, 1, 1).repeat(1, 1, true_obs.shape[2], true_obs.shape[3])
    true_obs = torch.cat([true_obs, state_tile], dim=1)

    o0 = o0.to(device)
    true_obs = true_obs.to(device)
    trans_obs = trans_obs.to(device)

    true_obs = nn.GeometricTensor(true_obs, nn.FieldType(d4, 5 * [d4.trivial_repr]))
    trans_obs = nn.GeometricTensor(trans_obs, nn.FieldType(d4, 5 * [d4.trivial_repr]))

    layer0 = network.conv[:2]
    layer1 = network.conv[:5]
    layer2 = network.conv[:8]
    layer3 = network.conv[:11]
    layer4 = network

    o0_outs = []
    true_obs_outs = []
    trans_obs_outs = []

    with torch.no_grad():
        for layer in [layer0, layer1, layer2, layer3, layer4]:
            o0_outs.append(layer(o0.tensor))
            true_obs_outs.append(layer(true_obs.tensor))
            trans_obs_outs.append(layer(trans_obs.tensor))

    o0 = o0.tensor
    true_obs = true_obs.tensor
    trans_obs = trans_obs.tensor

    obs_error = calcErrorMatrix(true_obs, trans_obs)
    layer_errors = []
    for i in range(len(true_obs_outs)-1):
        layer_errors.append(calcErrorMatrix(true_obs_outs[i], trans_obs_outs[i]))
        # true = true_obs_outs[i].reshape(true_obs_outs[i].shape[0], -1, 8, true_obs_outs[i].shape[-2]*true_obs_outs[i].shape[-1]).sum(-1)
        # trans = trans_obs_outs[i].reshape(trans_obs_outs[i].shape[0], -1, 8, trans_obs_outs[i].shape[-2]*trans_obs_outs[i].shape[-1]).sum(-1)
        # layer_errors.append(calcErrorMatrix(true, trans))

    layer_errors.append(calcErrorMatrix(true_obs_outs[-1], trans_obs_outs[-1]))

    layer_errors = [obs_error] + layer_errors

    fig, axs = plt.subplots(2, int(np.ceil(len(layer_errors)/2)), figsize=(0.3*n_true*len(layer_errors), 5*2))
    axs = axs.reshape(-1)
    for i in range(len(layer_errors)):
        axs[i].imshow(layer_errors[i])
        min_cell = np.stack((layer_errors[i].argmin(1), np.arange(n_true))).T
        # axs[i].scatter(layer_errors[i].argmin(1), np.arange(n_true), color='r')
        diag_rank = layer_errors[i].argsort().argsort()[np.eye(n_true, dtype=np.bool_)]
        for j in range(n_true):
            rect = plt.Rectangle((min_cell[j][0]-0.5, min_cell[j][1]-0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            axs[i].add_patch(rect)
            axs[i].text(j, j, diag_rank[j], color='r', horizontalalignment='center',
        verticalalignment='center', size=18, fontdict=None)
        if i == 0:
            axs[i].set_title('obs')
        elif i == len(layer_errors)-1:
            axs[i].set_title('output')
        else:
            axs[i].set_title('layer {}'.format(i-1))
    if len(layer_errors) % 2 != 0:
        axs[-1].axis('off')
    plt.tight_layout()
    fig.show()

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.reshape(-1)
    for i in range(8):
        axs[i].imshow(true_obs[i, :3].permute(1, 2, 0).cpu())
    plt.tight_layout()
    fig.show()

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.reshape(-1)
    for i in range(8):
        axs[i].imshow(trans_obs[i, :3].permute(1, 2, 0).cpu())
    plt.tight_layout()
    fig.show()

if __name__ == '__main__':
    plt.style.use('default')
    # group = gspaces.FlipRot2dOnR2(4)
    # network = EquivariantEncoder64(group, 5, 64).to(device)
    if model == 'equi_d':
        group = gspaces.FlipRot2dOnR2(4)
        network = EquivariantEncoder64(group, 5, 64).to(device)
    elif model == 'cnn_ssm':
        network = SACEncoder(obs_shape=(5, 64, 64), out_dim=64*8, ssm=True).to(device)
    else:
        raise NotImplementedError
    network.load_state_dict(torch.load('/tmp/equi_map/train_close_loop_block_picking_2022-08-15.19:22:19/models/model_99'))
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)

    test(network, envs)
    print(0)