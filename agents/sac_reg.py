from agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils.torch_utils import augmentTransition, perturb

class SACReg(SAC):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel', model_loss_w=0.1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval,
                         automatic_entropy_tuning, obs_type)
        self.actor_reward_model = None
        self.actor_transition_model = None
        self.actor_reward_optimizer = None
        self.actor_transition_optimizer = None
        self.critic_reward_model = None
        self.critic_transition_model = None
        self.critic_reward_optimizer = None
        self.critic_transition_optimizer = None
        self.model_loss_w = model_loss_w

        self.fix_trans_reward = False

    def initNetwork(self, actor, critic, actor_reward_model, actor_transition_model, critic_reward_model, critic_transition_model, initialize_target=True):
        super().initNetwork(actor, critic, initialize_target)
        self.actor_reward_model = actor_reward_model
        self.actor_transition_model = actor_transition_model
        self.critic_reward_model = critic_reward_model
        self.critic_transition_model = critic_transition_model

        self.networks.append(self.actor_reward_model)
        self.networks.append(self.actor_transition_model)
        self.networks.append(self.critic_reward_model)
        self.networks.append(self.critic_transition_model)

        self.actor_reward_optimizer = torch.optim.Adam(self.actor_reward_model.parameters(), lr=1e-3)
        self.actor_transition_optimizer = torch.optim.Adam(self.actor_transition_model.parameters(), lr=1e-3)
        self.critic_reward_optimizer = torch.optim.Adam(self.critic_reward_model.parameters(), lr=1e-3)
        self.critic_transition_optimizer = torch.optim.Adam(self.critic_transition_model.parameters(), lr=1e-3)

        self.optimizers.append(self.actor_reward_optimizer)
        self.optimizers.append(self.actor_transition_optimizer)
        self.optimizers.append(self.critic_reward_optimizer)
        self.optimizers.append(self.critic_transition_optimizer)

        if initialize_target:
            self.actor_target = deepcopy(actor)
            self.target_networks.append(self.actor_target)

    def saveModel(self, path_pre):
        """
        save the models with path prefix path_pre. a '_q{}.pt' suffix will be added to each model
        :param path_pre: path prefix
        """
        networks = self.networks + [self.actor.conv, self.critic.critic_1, self.critic.critic_2]
        for i in range(len(networks)):
            torch.save(self.networks[i].state_dict(), '{}_{}.pt'.format(path_pre, i))

    def loadTransAndRewardModel(self, path_pre):
        """
        load the saved models
        :param path_pre: path prefix to the model
        """
        for i in range(2, 6):
            path = path_pre + '_{}.pt'.format(i)
            print('loading {}'.format(path))
            self.networks[i].load_state_dict(torch.load(path))
        self.fix_trans_reward = True

    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def updateCritic(self):
        qf1_loss, qf2_loss, td_error = self.calcCriticLoss()
        qf_loss = qf1_loss + qf2_loss

        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        latent_state = self.critic.img_conv.forwardNormalTensor(obs)
        reward_pred = self.critic_reward_model(latent_state, action)
        reward_model_loss = F.mse_loss(reward_pred, rewards)

        latent_next_state_pred = self.critic_transition_model(latent_state, action)
        latent_next_state_gc = self.critic_target.img_conv.forwardNormalTensor(next_obs).tensor.reshape(batch_size, -1).detach()
        transition_model_loss = F.mse_loss(latent_next_state_pred, latent_next_state_gc)

        critic_loss = qf_loss + self.model_loss_w * (reward_model_loss + transition_model_loss)

        if self.fix_trans_reward:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        else:
            model_loss = reward_model_loss + transition_model_loss
            self.critic_reward_optimizer.zero_grad()
            self.critic_transition_optimizer.zero_grad()
            model_loss.backward(retain_graph=True)
            self.critic_reward_optimizer.step()
            self.critic_transition_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()



        return qf1_loss, qf2_loss, reward_model_loss, transition_model_loss, td_error

    def updateActorAndAlpha(self):
        policy_loss = self.calcActorLoss()
        log_pi = self.loss_calc_dict['log_pi']

        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        latent_state = self.actor.img_conv.forwardNormalTensor(obs)
        reward_pred = self.actor_reward_model(latent_state, action)
        reward_model_loss = F.mse_loss(reward_pred, rewards)

        latent_next_state_pred = self.actor_transition_model(latent_state, action)
        latent_next_state_gc = self.actor_target.img_conv.forwardNormalTensor(next_obs).tensor.reshape(batch_size, -1).detach()
        transition_model_loss = F.mse_loss(latent_next_state_pred, latent_next_state_gc)

        policy_loss = policy_loss + self.model_loss_w * (reward_model_loss + transition_model_loss)

        if self.fix_trans_reward:
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        else:
            model_loss = reward_model_loss + transition_model_loss
            self.actor_reward_optimizer.zero_grad()
            self.actor_transition_optimizer.zero_grad()
            model_loss.backward(retain_graph=True)
            self.actor_reward_optimizer.step()
            self.actor_transition_optimizer.step()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        return policy_loss, alpha_loss, alpha_tlogs, reward_model_loss, transition_model_loss

    def update(self, batch):
        self._loadBatchToDevice(batch)
        qf1_loss, qf2_loss, critic_reward_model_loss, critic_transition_model_loss, td_error = self.updateCritic()
        policy_loss, alpha_loss, alpha_tlogs, actor_reward_model_loss, actor_transition_model_loss = self.updateActorAndAlpha()

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), critic_reward_model_loss.item(), critic_transition_model_loss.item(), actor_reward_model_loss.item(), actor_transition_model_loss.item()), td_error

