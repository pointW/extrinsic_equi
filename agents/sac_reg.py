from agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils.torch_utils import augmentTransition, perturb

class SACReg(SAC):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel'):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval,
                         automatic_entropy_tuning, obs_type)
        self.reward_model = None
        self.transition_model = None
        self.reward_optimizer = None
        self.transition_optimizer = None
        self.model_loss_w = 1

    def initNetwork(self, actor, critic, reward_model, transition_model, initialize_target=True):
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.transition_model = transition_model
        self.actor_optimizer = torch.optim.Adam(self.actor.conv.parameters(), lr=self.lr[0])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr[1])
        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.transition_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=1e-3)
        if initialize_target:
            self.critic_target = deepcopy(critic)
            self.target_networks.append(self.critic_target)
        self.networks.append(self.actor)
        self.networks.append(self.critic)
        self.networks.append(self.reward_model)
        self.networks.append(self.transition_model)
        self.optimizers.append(self.actor_optimizer)
        self.optimizers.append(self.critic_optimizer)
        self.optimizers.append(self.alpha_optim)
        self.optimizers.append(self.reward_optimizer)
        self.optimizers.append(self.transition_optimizer)

    def updateCritic(self):
        qf1_loss, qf2_loss, td_error = self.calcCriticLoss()
        qf_loss = qf1_loss + qf2_loss

        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        latent_state = self.critic.img_conv.forwardNormalTensor(obs)
        reward_pred = self.reward_model(latent_state, action)
        reward_model_loss = F.mse_loss(reward_pred, rewards)

        latent_next_state_pred = self.transition_model(latent_state, action)
        latent_next_state_gc = self.critic_target.img_conv.forwardNormalTensor(next_obs).tensor.reshape(batch_size, -1).detach()
        transition_model_loss = F.mse_loss(latent_next_state_pred, latent_next_state_gc)

        critic_loss = qf_loss + self.model_loss_w * (reward_model_loss + transition_model_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        model_loss = reward_model_loss + transition_model_loss
        self.reward_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        model_loss.backward()
        self.reward_optimizer.step()
        self.transition_optimizer.step()

        return qf1_loss, qf2_loss, reward_model_loss, transition_model_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        qf1_loss, qf2_loss, reward_model_loss, transition_model_loss, td_error = self.updateCritic()
        policy_loss, alpha_loss, alpha_tlogs = self.updateActorAndAlpha()

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), reward_model_loss.item(), transition_model_loss.item()), td_error

