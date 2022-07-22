from agents.sac_share_enc import SACShareEnc
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import itertools
from tqdm import tqdm

from utils.torch_utils import augmentTransition, perturb

class SACShareEncReg(SACShareEnc):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel', model_loss_w=0.1, train_reg=False):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval,
                         automatic_entropy_tuning, obs_type)
        self.reward_model = None
        self.transition_model = None
        self.reward_optimizer = None
        self.transition_optimizer = None

        self.model_loss_w = model_loss_w
        self.train_reg = train_reg

        self.fix_trans_reward = False

        self.r_model_loss_type = 'mse'
        assert self.r_model_loss_type in ['mse', 'bce']

    def initNetwork(self, enc, actor, critic, reward_model, transition_model, initialize_target=True):
        super().initNetwork(enc, actor, critic, initialize_target)
        self.reward_model = reward_model
        self.transition_model = transition_model

        self.networks.append(self.reward_model)
        self.networks.append(self.transition_model)

        self.reward_optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.transition_optimizer = torch.optim.Adam(self.transition_model.parameters(), lr=1e-3)

        self.optimizers.append(self.reward_optimizer)
        self.optimizers.append(self.transition_optimizer)

    def loadTransAndRewardModel(self, path_pre):
        """
        load the saved models
        :param path_pre: path prefix to the model
        """
        for i in range(3, 5):
            path = path_pre + '_{}.pt'.format(i)
            print('loading {}'.format(path))
            self.networks[i].load_state_dict(torch.load(path))
        self.fix_trans_reward = True

    def calcModelLoss(self, require_enc_grad=True):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        if not require_enc_grad:
            with torch.no_grad():
                latent_state = self.enc(obs)
        else:
            latent_state = self.enc(obs)
        reward_pred = self.reward_model(latent_state, action)
        if self.r_model_loss_type == 'mse':
            reward_model_loss = F.mse_loss(reward_pred, rewards)
        else:
            reward_model_loss = F.binary_cross_entropy_with_logits(reward_pred, rewards)

        latent_next_state_pred = self.transition_model(latent_state, action)
        latent_next_state_gc = self.enc(next_obs).tensor.reshape(batch_size, -1).detach()
        transition_model_loss = F.mse_loss(latent_next_state_pred, latent_next_state_gc)

        return reward_model_loss, transition_model_loss

    def updateCriticReg(self):
        qf1_loss, qf2_loss, td_error = self.calcCriticLoss()
        reward_model_loss, transition_model_loss = self.calcModelLoss()

        qf_loss = qf1_loss + qf2_loss
        critic_loss = qf_loss + self.model_loss_w * (reward_model_loss + transition_model_loss)

        if self.fix_trans_reward:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        else:
            model_loss = reward_model_loss + transition_model_loss
            self.reward_optimizer.zero_grad()
            self.transition_optimizer.zero_grad()
            model_loss.backward(retain_graph=True)
            self.reward_optimizer.step()
            self.transition_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        return qf1_loss, qf2_loss, reward_model_loss, transition_model_loss, td_error

    def updateReg(self, batch):
        self._loadBatchToDevice(batch)
        qf1_loss, qf2_loss, critic_reward_model_loss, critic_transition_model_loss, td_error = self.updateCriticReg()
        policy_loss, alpha_loss, alpha_tlogs = self.updateActorAndAlpha()

        self.num_update += 1
        if self.num_update % self.target_update_interval == 0:
            self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), critic_reward_model_loss.item(), critic_transition_model_loss.item()), td_error

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
        self._snapshots = (None, 1e10, 1e10)

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
                reward_model_loss, transition_model_loss = self.calcModelLoss(require_enc_grad=False)

                self.reward_optimizer.zero_grad()
                reward_model_loss.backward()
                self.reward_optimizer.step()

                self.transition_optimizer.zero_grad()
                transition_model_loss.backward()
                self.transition_optimizer.step()

                pbar.update()
                logger.model_losses.append((reward_model_loss.item(), transition_model_loss.item()))

            with torch.no_grad():
                self._loadBatchToDevice(holdout_data)
                holdout_reward_loss, holdout_transition_loss = self.calcModelLoss()

                logger.model_holdout_losses.append((holdout_reward_loss.item(), holdout_transition_loss.item()))
                logger.saveModelLossCurve()
                logger.saveModelHoldoutLossCurve()

                pbar.set_description('epoch: {}, reward loss: {:.03f}, transition loss: {:.03f}'.format(epoch, holdout_reward_loss.item(), holdout_transition_loss.item()))

                break_train = self._save_best(epoch, holdout_reward_loss, holdout_transition_loss)

                if break_train:
                    break

    def _save_best(self, epoch, holdout_reward_loss, holdout_transition_loss):
        updated = False
        current_reward_loss = holdout_reward_loss
        current_transition_loss = holdout_transition_loss
        _, best_reward_loss, best_transition_loss = self._snapshots
        improvement_reward = (best_reward_loss - current_reward_loss) / best_reward_loss
        improvement_transition = (best_transition_loss - current_transition_loss) / best_transition_loss

        if improvement_reward > 0.01 or improvement_transition > 0.01:
            self._snapshots = (epoch, current_reward_loss, current_transition_loss)
            updated = True

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False
