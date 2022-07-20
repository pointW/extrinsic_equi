from agents.sac import SAC
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import itertools
from tqdm import tqdm

from utils.torch_utils import augmentTransition, perturb

class SACReg(SAC):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel', model_loss_w=0.1, train_reg=False):
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
        self.train_reg = train_reg

        self.fix_trans_reward = False

        self.r_model_loss_type = 'mse'
        assert self.r_model_loss_type in ['mse', 'bce']

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
            torch.save(networks[i].state_dict(), '{}_{}.pt'.format(path_pre, i))

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

    def updateCriticReg(self):
        qf1_loss, qf2_loss, td_error = self.calcCriticLoss()
        qf_loss = qf1_loss + qf2_loss

        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        latent_state = self.critic.img_conv.forwardNormalTensor(obs)
        reward_pred = self.critic_reward_model(latent_state, action)
        if self.r_model_loss_type == 'mse':
            reward_model_loss = F.mse_loss(reward_pred, rewards)
        else:
            reward_model_loss = F.binary_cross_entropy_with_logits(reward_pred, rewards)

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

    def updateActorAndAlphaReg(self):
        policy_loss = self.calcActorLoss()
        log_pi = self.loss_calc_dict['log_pi']

        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        latent_state = self.actor.img_conv.forwardNormalTensor(obs)
        reward_pred = self.actor_reward_model(latent_state, action)
        if self.r_model_loss_type == 'mse':
            reward_model_loss = F.mse_loss(reward_pred, rewards)
        else:
            reward_model_loss = F.binary_cross_entropy_with_logits(reward_pred, rewards)

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

    def updateReg(self, batch):
        self._loadBatchToDevice(batch)
        qf1_loss, qf2_loss, critic_reward_model_loss, critic_transition_model_loss, td_error = self.updateCriticReg()
        policy_loss, alpha_loss, alpha_tlogs, actor_reward_model_loss, actor_transition_model_loss = self.updateActorAndAlphaReg()

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), critic_reward_model_loss.item(), critic_transition_model_loss.item(), actor_reward_model_loss.item(), actor_transition_model_loss.item()), td_error

    def update(self, batch):
        if self.train_reg:
            return self.updateReg(batch)
        else:
            return super().update(batch)

    def trainModel(self, logger, data, batch_size, holdout_ratio=0.2, max_epochs_since_update=5, max_epochs=100):
        if self.fix_trans_reward:
            return
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0
        self._state = {}
        self._snapshots = (None, 1e10, 1e10, 1e10, 1e10)

        num_holdout = int(batch_size * holdout_ratio)
        permutation = np.random.permutation(len(data))

        data = np.array(data+[None], dtype=object)[:-1][permutation]
        train_data = data[num_holdout:]
        holdout_data = data[:num_holdout]

        def generator():
            while True:
                yield

        pbar = tqdm(generator())

        for epoch in range(max_epochs):
            train_idx = np.random.permutation(train_data.shape[0])
            for start_pos in range(0, train_data.shape[0], batch_size):
                idx = train_idx[start_pos: start_pos + batch_size]
                batch = train_data[idx]
                self._loadBatchToDevice(batch)
                mini_batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
                with torch.no_grad():
                    actor_latent_state = self.actor.img_conv.forwardNormalTensor(obs)
                actor_reward_pred = self.actor_reward_model(actor_latent_state, action)
                if self.r_model_loss_type == 'mse':
                    actor_reward_model_loss = F.mse_loss(actor_reward_pred, rewards)
                else:
                    actor_reward_model_loss = F.binary_cross_entropy_with_logits(actor_reward_pred, rewards)
                self.actor_reward_optimizer.zero_grad()
                actor_reward_model_loss.backward()
                self.actor_reward_optimizer.step()

                actor_latent_next_state_pred = self.actor_transition_model(actor_latent_state, action)
                actor_latent_next_state_gc = self.actor.img_conv.forwardNormalTensor(next_obs).tensor.reshape(mini_batch_size, -1).detach()
                actor_transition_model_loss = F.mse_loss(actor_latent_next_state_pred, actor_latent_next_state_gc)
                self.actor_transition_optimizer.zero_grad()
                actor_transition_model_loss.backward()
                self.actor_transition_optimizer.step()

                with torch.no_grad():
                    critic_latent_state = self.critic.img_conv.forwardNormalTensor(obs)
                critic_reward_pred = self.critic_reward_model(critic_latent_state, action)
                if self.r_model_loss_type == 'mse':
                    critic_reward_model_loss = F.mse_loss(critic_reward_pred, rewards)
                else:
                    critic_reward_model_loss = F.binary_cross_entropy_with_logits(critic_reward_pred, rewards)
                self.critic_reward_optimizer.zero_grad()
                critic_reward_model_loss.backward()
                self.critic_reward_optimizer.step()

                critic_latent_next_state_pred = self.critic_transition_model(critic_latent_state, action)
                critic_latent_next_state_gc = self.critic.img_conv.forwardNormalTensor(next_obs).tensor.reshape(mini_batch_size, -1).detach()
                critic_transition_model_loss = F.mse_loss(critic_latent_next_state_pred, critic_latent_next_state_gc)
                self.critic_transition_optimizer.zero_grad()
                critic_transition_model_loss.backward()
                self.critic_transition_optimizer.step()

                pbar.update()
                logger.model_losses.append((actor_reward_model_loss.item(), actor_transition_model_loss.item(), critic_reward_model_loss.item(), critic_transition_model_loss.item()))

            with torch.no_grad():
                self._loadBatchToDevice(holdout_data)
                mini_batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
                actor_latent_state = self.actor.img_conv.forwardNormalTensor(obs)
                actor_reward_pred = self.actor_reward_model(actor_latent_state, action)
                if self.r_model_loss_type == 'mse':
                    actor_holdout_r_loss = F.mse_loss(actor_reward_pred, rewards)
                else:
                    actor_holdout_r_loss = F.binary_cross_entropy_with_logits(actor_reward_pred, rewards)

                actor_latent_next_state_pred = self.actor_transition_model(actor_latent_state, action)
                actor_latent_next_state_gc = self.actor.img_conv.forwardNormalTensor(next_obs).tensor.reshape(mini_batch_size, -1).detach()
                actor_holdout_t_loss = F.mse_loss(actor_latent_next_state_pred, actor_latent_next_state_gc)

                critic_latent_state = self.critic.img_conv.forwardNormalTensor(obs)
                critic_reward_pred = self.critic_reward_model(critic_latent_state, action)
                if self.r_model_loss_type == 'mse':
                    critic_holdout_r_loss = F.mse_loss(critic_reward_pred, rewards)
                else:
                    critic_holdout_r_loss = F.binary_cross_entropy_with_logits(critic_reward_pred, rewards)

                critic_latent_next_state_pred = self.critic_transition_model(critic_latent_state, action)
                critic_latent_next_state_gc = self.critic.img_conv.forwardNormalTensor(next_obs).tensor.reshape(mini_batch_size, -1).detach()
                critic_holdout_t_loss = F.mse_loss(critic_latent_next_state_pred, critic_latent_next_state_gc)

                logger.model_holdout_losses.append((actor_holdout_r_loss.item(), actor_holdout_t_loss.item(), critic_holdout_r_loss.item(), critic_holdout_t_loss.item()))
                logger.saveModelLossCurve()
                logger.saveModelHoldoutLossCurve()

                pbar.set_description('epoch: {}, ar loss: {:.03f}, at loss: {:.03f}, cr loss: {:.03f}, ct loss: {:.03f}'.format(epoch, actor_holdout_r_loss.item(), actor_holdout_t_loss.item(), critic_holdout_r_loss.item(), critic_holdout_t_loss.item()))

                break_train = self._save_best(epoch, actor_holdout_t_loss, actor_holdout_r_loss, critic_holdout_t_loss, critic_holdout_r_loss)

                if break_train:
                    break

    def _save_best(self, epoch, actor_holdout_t_loss, actor_holdout_r_loss, critic_holdout_t_loss, critic_holdout_r_loss):
        updated = False
        current_actor_t_loss = actor_holdout_t_loss
        current_actor_r_loss = actor_holdout_r_loss
        current_critic_t_loss = critic_holdout_t_loss
        current_critic_r_loss = critic_holdout_r_loss
        _, best_actor_t, best_actor_r, best_critic_t, best_critic_r = self._snapshots
        improvement_actor_t = (best_actor_t - current_actor_t_loss) / best_actor_t
        improvement_actor_r = (best_actor_r - current_actor_r_loss) / best_actor_r
        improvement_critic_t = (best_critic_t - current_critic_t_loss) / best_critic_t
        improvement_critic_r = (best_critic_r - current_critic_r_loss) / best_critic_r

        if improvement_actor_t > 0.01 or improvement_actor_r > 0.01 or improvement_critic_t > 0.01 or improvement_critic_r > 0.01:
            self._snapshots = (epoch, current_actor_t_loss, current_actor_r_loss, current_critic_t_loss, current_critic_r_loss)
            updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False
