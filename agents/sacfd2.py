from agents.sacfd import SACfD
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

class SACfD2(SACfD):
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_a=5, tau=0.001,
                 alpha=0.01, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=False,
                 obs_type='pixel', demon_w=0.1, demon_l='pi', critic_demo_loss='margin', critic_n_neg=2048,
                 critic_demo_w=0.1, critic_margin_l=0.01):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_a, tau, alpha, policy_type, target_update_interval,
                         automatic_entropy_tuning, obs_type, demon_w, demon_l)
        self.critic_demo_loss = critic_demo_loss # margin or ce
        self.critic_n_neg = critic_n_neg
        self.critic_demo_w = critic_demo_w
        self.critic_margin_l = critic_margin_l

    def calcCriticDemoLossMargin(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        states, obs, action = states[is_experts], obs[is_experts], action[is_experts]
        batch_size = states.shape[0]
        if batch_size == 0:
            return torch.tensor(0., device=self.device)

        # Generate N negatives, one for each element in the batch: (B, N, D).
        size = (batch_size * self.critic_n_neg, self.n_a)
        samples = np.random.uniform(-1, 1, size=size)
        negatives = torch.as_tensor(samples, dtype=torch.float32, device=self.device).reshape(batch_size, self.critic_n_neg, -1)

        with torch.no_grad():
            qe1, qe2 = self.critic(obs, action)
            qe1 = qe1.reshape(batch_size)
            qe2 = qe2.reshape(batch_size)
            qe = torch.min(qe1, qe2)

        q_neg1, q_neg2 = self.critic(obs, negatives)
        losses = []

        for i in range(batch_size):
            q_neg1_over = q_neg1[i][q_neg1[i] > qe[i] - self.critic_margin_l]
            if q_neg1_over.shape[0] > 0:
                losses.append((q_neg1_over - (qe[i] - self.critic_margin_l)).mean())
            q_neg2_over = q_neg1[i][q_neg1[i] > qe[i] - self.critic_margin_l]
            if q_neg2_over.shape[0] > 0:
                losses.append((q_neg2_over - (qe[i] - self.critic_margin_l)).mean())
        if len(losses) == 0:
            return torch.tensor(0., device=self.device)
        else:
            return torch.stack(losses).mean()

    def calcCriticDemoLossCE(self):
        batch_size, states, obs, action, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        states, obs, action = states[is_experts], obs[is_experts], action[is_experts]
        batch_size = states.shape[0]
        if batch_size == 0:
            return torch.tensor(0., device=self.device)

        # Generate N negatives, one for each element in the batch: (B, N, D).
        size = (batch_size * self.critic_n_neg, self.n_a)
        samples = np.random.uniform(-1, 1, size=size)
        negatives = torch.as_tensor(samples, dtype=torch.float32, device=self.device).reshape(batch_size, self.critic_n_neg, -1)

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([action.unsqueeze(dim=1), negatives], dim=1)

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = torch.nonzero(permutation == 0)[:, 1].to(self.device)

        # For every element in the mini-batch, there is 1 positive for which the critic
        # should output a high value, and N negatives for which the critic should
        # output low values.
        q1, q2 = self.critic(obs, targets)

        loss = F.cross_entropy(q1, ground_truth) + F.cross_entropy(q2, ground_truth)

        return loss

    def updateCritic(self):
        qf1_loss, qf2_loss, td_error = self.calcCriticLoss()
        if self.critic_demo_loss == 'margin':
            q_demo_loss = self.calcCriticDemoLossMargin()
        elif self.critic_demo_loss == 'ce':
            q_demo_loss = self.calcCriticDemoLossCE()
        else:
            raise NotImplementedError
        qf_loss = qf1_loss + qf2_loss + self.critic_demo_w * q_demo_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        return qf1_loss, qf2_loss, td_error

