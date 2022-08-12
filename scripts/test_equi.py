import copy
import collections
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils.create_agent import createAgent
from utils.parameters import *
from utils.env_wrapper import EnvWrapper

import torch
from e2cnn import gspaces, nn
from utils.torch_utils import centerCrop

def calcErrorMatrix(t0: torch.Tensor, t1: torch.Tensor):
    error = np.zeros((t0.shape[0], t1.shape[0]))
    for i in range(t0.shape[0]):
        e = torch.abs(t0[i:i+1] - t1)
        e = e.reshape(e.shape[0], -1).sum(1)
        error[i] = e.cpu().numpy()
    return error

def test():
    plt.style.use('default')
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    agent = createAgent(test=True)
    agent.train()
    agent.loadModel(load_model_pre)

    n_true = 8

    states = []
    true_obs = []
    for i in range(n_true):
        state, obs = envs.reset()
        true_obs.append(obs)
        states.append(state)

    # d4 = gspaces.FlipRot2dOnR2(4)
    d4 = gspaces.Rot2dOnR2(8)
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

    o0 = centerCrop(o0)
    true_obs = centerCrop(true_obs)
    trans_obs = centerCrop(trans_obs)

    o0 = o0.to(device)
    true_obs = true_obs.to(device)
    trans_obs = trans_obs.to(device)

    true_obs = nn.GeometricTensor(true_obs, nn.FieldType(d4, 5 * [d4.trivial_repr]))
    trans_obs = nn.GeometricTensor(trans_obs, nn.FieldType(d4, 5 * [d4.trivial_repr]))

    layer0 = agent.actor.img_conv.conv[:2]
    layer1 = agent.actor.img_conv.conv[:5]
    layer2 = agent.actor.img_conv.conv[:8]
    layer3 = agent.actor.img_conv.conv[:11]
    layer4 = agent.actor.img_conv.conv[:13]
    layer5 = agent.actor.img_conv
    layer6 = torch.nn.Sequential(agent.actor.img_conv, agent.actor.conv)

    o0_outs = []
    true_obs_outs = []
    trans_obs_outs = []

    with torch.no_grad():
        for layer in [layer0, layer1, layer2, layer3, layer4, layer5, layer6]:
            o0_outs.append(layer(o0).tensor)
            true_obs_outs.append(layer(true_obs).tensor)
            trans_obs_outs.append(layer(trans_obs).tensor)

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

    fig, axs = plt.subplots(2, len(layer_errors)//2, figsize=(0.3*n_true*len(layer_errors), 5*2))
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

    print(0)
    # feature0 = layer0(obs_geo)
    # feature1 = layer1(obs_geo)
    # feature2 = layer2(obs_geo)
    # feature3 = layer3(obs_geo)
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3, 2, figsize=(10, 11))
    # axs[0, 0].imshow(feature0.tensor.sum(1).cpu()[0])
    # axs[0, 0].set_title('layer 1')
    # axs[0, 1].imshow(feature1.tensor.sum(1).cpu()[0])
    # axs[0, 1].set_title('layer 2')
    # axs[1, 0].imshow(feature2.tensor.sum(1).cpu()[0])
    # axs[1, 0].set_title('layer 3')
    # axs[1, 1].imshow(feature3.tensor.sum(1).cpu()[0])
    # axs[1, 1].set_title('layer 4')
    # axs[2, 0].imshow(enc_out.tensor.reshape(batch_size, 64, 8).sum(1).cpu())
    # axs[2, 0].set_title('conv out')
    # axs[2, 1].axis('off')
    # fig.show()

    print(0)

if __name__ == '__main__':
    test()