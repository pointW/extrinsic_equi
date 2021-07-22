from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

class BaseAgent:
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32):
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dr = dr

        self.networks = []
        self.target_networks = []
        self.optimizers = []

        self.loss_calc_dict = {}

    def update(self, batch):
        raise NotImplementedError

    def _loadBatchToDevice(self, batch):
        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs)
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.stack(states).long().to(self.device)
        image_tensor = torch.stack(images).to(self.device)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(1)
        xy_tensor = torch.stack(xys).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).long().to(self.device)
        next_obs_tensor = torch.stack(next_obs).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.stack(dones).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.stack(step_lefts).to(self.device)
        is_experts_tensor = torch.stack(is_experts).bool().to(self.device)

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = image_tensor
        self.loss_calc_dict['action_idx'] = xy_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = next_obs_tensor
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor

        return states_tensor, image_tensor, xy_tensor, rewards_tensor, next_states_tensor, \
               next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor

    def _loadLossCalcDict(self):
        """
        get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def train(self):
        for i in range(len(self.networks)):
            self.networks[i].train()
        for i in range(len(self.target_networks)):
            self.target_networks[i].train()

    def eval(self):
        for i in range(len(self.networks)):
            self.networks[i].eval()

    def getModelStr(self):
        return str(self.networks)

    def updateTarget(self):
        """
        hard update the target networks
        """
        for i in range(len(self.networks)):
            self.target_networks[i].load_state_dict(self.networks[i].state_dict())

    def loadModel(self, path_pre):
        """
        load the saved models
        :param path_pre: path prefix to the model
        """
        for i in range(len(self.networks)):
            path = path_pre + '_{}.pt'.format(i)
            print('loading {}'.format(path))
            self.networks[i].load_state_dict(torch.load(path))
        self.updateTarget()

    def saveModel(self, path_pre):
        """
        save the models with path prefix path_pre. a '_q{}.pt' suffix will be added to each model
        :param path_pre: path prefix
        """
        for i in range(len(self.networks)):
            torch.save(self.networks[i].state_dict(), '{}_{}.pt'.format(path_pre, i))

    def getSaveState(self):
        """
        get the save state for checkpointing. Include network states, target network states, and optimizer states
        :return: the saving state dictionary
        """
        state = {}
        for i in range(len(self.networks)):
            state['{}'.format(i)] = self.networks[i].state_dict()
            state['{}_target'.format(i)] = self.target_networks[i].state_dict()
            state['{}_optimizer'.format(i)] = self.optimizers[i].state_dict()
        return state

    def loadFromState(self, save_state):
        """
        load from a save_state
        :param save_state: the loading state dictionary
        """
        for i in range(len(self.networks)):
            self.networks[i].load_state_dict(save_state['{}'.format(i)])
            self.target_networks[i].load_state_dict(save_state['{}_target'.format(i)])
            self.optimizers[i].load_state_dict(save_state['{}_optimizer'.format(i)])