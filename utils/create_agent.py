from utils.parameters import *
from agents.dqn_agent_fac import DQNAgentFac
from agents.dqn_agent_com import DQNAgentCom
from agents.dqn_agent_com_drq import DQNAgentComDrQ
from agents.sdqfd_agent_com import SDQfDCom
from agents.sdqfd_agent_com_drq import SDQfDComDrQ
from agents.curl_dqn_com import CURLDQNCom
from agents.curl_sdqfd_com import CURLSDQfDCom
from networks.cnn import CNNFac, CNNCom, CNNCom2
from networks.equivariant import EquivariantCNNFac, EquivariantCNNFac2, EquivariantCNNFac3, EquivariantCNNCom, EquivariantCNNCom2

from agents.ddpg import DDPG
from agents.ddpgfd import DDPGfD
from networks.cnn import Actor, Critic

from agents.sac import SAC
from agents.sacfd import SACfD
from agents.curl_sac import CURLSAC
from agents.curl_sacfd import CURLSACfD
from agents.sac_aug import SACAug
from agents.bc_continuous import BehaviorCloningContinuous
from agents.sac_drq import SACDrQ
from agents.sacfd_drq import SACfDDrQ
from agents.sac_aux import SACAux
from networks.sac_networks import SACDeterministicPolicy, SACGaussianPolicy, SACCritic, SACVecCritic, SACVecGaussianPolicy, SACCritic2, SACGaussianPolicy2
from networks.equivariant_sac_net import EquivariantSACActor, EquivariantSACCritic, EquivariantSACActor2, EquivariantPolicy, EquivariantSACVecCritic, EquivariantSACVecGaussianPolicy, EquivariantSACCriticNoGP, EquivariantSACActor3, EquivariantSACActorDihedral, EquivariantSACCriticDihedral, EquivariantSACActorDihedralShareEnc, EquivariantSACCriticDihedralShareEnc, EquivariantEncoder128Dihedral
from networks.equivariant_sac_net import EquivariantSACActorSO2_1, EquivariantSACCriticSO2_1, EquivariantSACActorSO2_2, EquivariantSACCriticSO2_2, EquivariantSACActorSO2_3, EquivariantSACCriticSO2_3, EquivariantPolicySO2, EquivariantSACActorO2, EquivariantSACCriticO2, EquivariantPolicyO2, EquivariantSACActorO2_2, EquivariantSACCriticO2_2, EquivariantSACActorO2_3, EquivariantSACCriticO2_3
from networks.equivariant_ddpg_net import EquivariantDDPGActor, EquivariantDDPGCritic
from networks.curl_sac_net import CURLSACEncoder, CURLSACCritic, CURLSACGaussianPolicy, CURLSACEncoderOri, CURLSACEncoder2
from networks.curl_equi_sac_net import CURLEquiSACEncoder, CURLEquiSACCritic, CURLEquiSACGaussianPolicy
from networks.cnn import DQNComCURL, DQNComCURLOri

from agents.sac_reg import SACReg
from agents.sac_share_enc import SACShareEnc
from networks.equivariant_dynamic_model import EquivariantRewardModelDihedral, EquivariantTransitionModelDihedral
from networks.equivariant_sac_net import EquivariantSACCriticDihedralWithNonEquiEnc, EquivariantSACActorDihedralWithNonEquiEnc
from networks.equivariant_sac_net import EquivariantPolicyDihedralWithNonEquiEnc, EquivariantPolicyDihedral

from networks.curl_equi_sac_net import CURLEquiSACActorDihedral, CURLEquiSACCriticDihedral

from agents.sacfd2 import SACfD2

from networks.equivariant_sac_net import EquivariantSACActorFlip, EquivariantSACCriticFlip
from networks.equivariant_sac_net import EquivariantSACActorTrivial, EquivariantSACCriticTrivial
from networks.sac_networks import SACGaussianPolicyFullyConv, SACCriticFullyConv

def createAgent(test=False):
    print('initializing agent')
    if view_type.find('rgbd') > -1:
        obs_channel = 5
    elif view_type.find('rgb') > -1:
        obs_channel = 4
    else:
        obs_channel = 2
    if load_sub is not None or load_model_pre is not None or test:
        initialize = False
    else:
        initialize = True
    n_p = 2
    if not random_orientation:
        n_theta = 1
    else:
        n_theta = 3

    # setup agent
    if alg == 'dqn_fac':
        agent = DQNAgentFac(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p, n_theta=n_theta)
        if model == 'cnn':
            net = CNNFac(n_p=n_p, n_theta=n_theta).to(device)
        elif model == 'equi':
            net = EquivariantCNNFac(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        elif model == 'equi_2':
            net = EquivariantCNNFac2(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        elif model == 'equi_3':
            net = EquivariantCNNFac3(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(net, initialize_target=not test)
    elif alg in ['dqn_com', 'sdqfd_com', 'dqn_com_drq', 'sdqfd_com_drq']:
        if alg == 'dqn_com':
            agent = DQNAgentCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p, n_theta=n_theta)
        elif alg == 'sdqfd_com':
            agent = SDQfDCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p,
                             n_theta=n_theta, l=margin_l, w=margin_weight)
        elif alg == 'dqn_com_drq':
            agent = DQNAgentComDrQ(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p,
                                   n_theta=n_theta)
        elif alg == 'sdqfd_com_drq':
            agent = SDQfDComDrQ(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p,
                                n_theta=n_theta, l=margin_l, w=margin_weight)
        if model == 'cnn':
            net = CNNCom((obs_channel, crop_size, crop_size), n_p=n_p, n_theta=n_theta).to(device)
        elif model == 'cnn_2':
            net = CNNCom2((obs_channel, crop_size, crop_size), n_p=n_p, n_theta=n_theta).to(device)
        elif model == 'equi':
            net = EquivariantCNNCom(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        elif model == 'equi_2':
            net = EquivariantCNNCom2(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(net, initialize_target=not test)

    elif alg in ['curl_dqn_com', 'curl_sdqfd_com']:
        if alg == 'curl_dqn_com':
            agent = CURLDQNCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p,
                               n_theta=n_theta, crop_size=crop_size)
        else:
            raise NotImplementedError
        if model == 'cnn':
            net = DQNComCURL((obs_channel, crop_size, crop_size), n_p, n_theta).to(device)
        # network from curl paper
        elif model == 'cnn_curl':
            net = DQNComCURLOri((obs_channel, crop_size, crop_size), n_p, n_theta).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(net)

    elif alg in ['ddpg', 'ddpgfd']:
        ddpg_lr = (actor_lr, critic_lr)
        if alg == 'ddpg':
            agent = DDPG(lr=ddpg_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                         n_a=len(action_sequence), tau=tau)
        elif alg == 'ddpgfd':
            agent = DDPGfD(lr=ddpg_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                           n_a=len(action_sequence), tau=tau, demon_w=demon_w)
        else:
            raise NotImplementedError
        if model == 'cnn':
            actor = Actor(len(action_sequence)).to(device)
            critic = Critic(len(action_sequence)).to(device)
        elif model == 'equi_both':
            actor = EquivariantDDPGActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden,
                                        initialize=initialize, N=equi_n).to(device)
            critic = EquivariantDDPGCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden,
                                        initialize=initialize, N=equi_n).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic, initialize_target=not test)

    elif alg in ['sac', 'sacfd', 'sacfd_mean', 'sacfd2', 'sac_drq', 'sacfd_drq', 'sac_aux']:
        sac_lr = (actor_lr, critic_lr)
        if alg == 'sac':
            agent = SAC(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                        n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                        target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        elif alg == 'sacfd':
            agent = SACfD(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                          n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                          target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                          demon_w=demon_w)
        elif alg == 'sacfd_mean':
            agent = SACfD(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                          n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                          target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                          demon_w=demon_w, demon_l='mean')
        elif alg == 'sacfd2':
            agent = SACfD2(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                           n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                           target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                           demon_w=demon_w, critic_demo_loss=critic_demo_loss, critic_n_neg=critic_n_neg,
                           critic_demo_w=critic_demo_w, critic_margin_l=critic_margin_l)
        elif alg == 'sac_drq':
            agent = SACDrQ(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                           n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                           target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        elif alg == 'sacfd_drq':
            agent = SACfDDrQ(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                             n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                             target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                             demon_w=demon_w)
        elif alg == 'sac_aux':
            agent = SACAux(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                           n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                           target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        else:
            raise NotImplementedError
        # pixel observation
        if obs_type == 'pixel':
            if model == 'cnn':
                actor = SACGaussianPolicy((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = SACCritic((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            elif model == 'cnn_ssm':
                actor = SACGaussianPolicy((obs_channel, crop_size, crop_size), len(action_sequence), ssm=True).to(device)
                critic = SACCritic((obs_channel, crop_size, crop_size), len(action_sequence), ssm=True).to(device)
            elif model == 'cnn_fully_conv':
                actor = SACGaussianPolicyFullyConv((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = SACCriticFullyConv((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            elif model == 'cnn_2':
                actor = SACGaussianPolicy2((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = SACCritic2((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            elif model == 'equi_actor':
                actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                critic = SACCritic2((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            elif model == 'equi_critic':
                actor = SACGaussianPolicy2((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            elif model == 'equi_both':
                actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            elif model == 'equi_both_d':
                actor = EquivariantSACActorDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=3).to(device)
                critic = EquivariantSACCriticDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=3).to(device)
            elif model == 'equi_both_f':
                actor = EquivariantSACActorFlip((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
                critic = EquivariantSACCriticFlip((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
            elif model == 'equi_both_t':
                actor = EquivariantSACActorTrivial((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
                critic = EquivariantSACCriticTrivial((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
            elif model == 'equi_both_d_share_enc':
                enc = EquivariantEncoder128Dihedral(obs_channel, n_hidden, initialize, equi_n).to(device)
                actor = EquivariantSACActorDihedralShareEnc(enc, (obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=3).to(device)
                critic = EquivariantSACCriticDihedralShareEnc(enc, (obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=3).to(device)
            elif model == 'equi_both_so2_1':
                actor = EquivariantSACActorSO2_1((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
                critic = EquivariantSACCriticSO2_1((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
            elif model == 'equi_both_so2_2':
                actor = EquivariantSACActorSO2_2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
                critic = EquivariantSACCriticSO2_2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
            elif model == 'equi_both_so2_3':
                actor = EquivariantSACActorSO2_3((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
                critic = EquivariantSACCriticSO2_3((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
            elif model == 'equi_both_o2':
                actor = EquivariantSACActorO2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
                critic = EquivariantSACCriticO2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
            elif model == 'equi_both_o2_2':
                actor = EquivariantSACActorO2_2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
                critic = EquivariantSACCriticO2_2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
            elif model == 'equi_both_o2_3':
                actor = EquivariantSACActorO2_3((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
                critic = EquivariantSACCriticO2_3((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
            elif model == 'equi_both_d_k5':
                actor = EquivariantSACActorDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=5).to(device)
                critic = EquivariantSACCriticDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=5).to(device)
            elif model == 'equi_both_2':
                actor = EquivariantSACActor2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize,
                                             N=equi_n).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize,
                                              N=equi_n).to(device)
            elif model == 'equi_both_3':
                actor = EquivariantSACActor3((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            elif model == 'equi_both_enc_2':
                actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence),
                                            n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_id=2).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence),
                                              n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_id=2).to(device)
            elif model == 'equi_both_enc_3':
                actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence),
                                            n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_id=3).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence),
                                                  n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_id=3).to(device)
            elif model == 'equi_both_enc_4':
                actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence),
                                            n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_id=4).to(device)
                critic = EquivariantSACCritic((obs_channel, crop_size, crop_size), len(action_sequence),
                                                  n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_id=4).to(device)
            elif model == 'equi_both_nogp':
                actor = EquivariantSACActor((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                critic = EquivariantSACCriticNoGP((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            else:
                if model == 'equi_both_d_w_enc' or model == 'equi_both_d_w_enc_fc':
                    enc_type = 'fc'
                    backbone = 'cnn'
                elif model == 'equi_both_d_w_enc_equi':
                    enc_type= 'equi'
                    backbone = 'cnn'
                elif model == 'equi_both_d_w_enc_ssm':
                    enc_type = 'ssm'
                    backbone = 'cnn'
                elif model == 'equi_both_d_w_enc_ssmstd':
                    enc_type = 'ssmstd'
                    backbone = 'cnn'
                elif model == 'equi_both_d_w_enc_res_fc':
                    enc_type = 'fc'
                    backbone = 'res'
                elif model == 'equi_both_d_w_enc_ssm_equi':
                    enc_type = 'ssm+equi'
                    backbone = 'cnn'
                else:
                    raise NotImplementedError
                actor = EquivariantSACActorDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                                  initialize=initialize, N=equi_n, enc_type=enc_type,
                                                                  backbone=backbone, n_channels=actor_channels).to(device)
                critic = EquivariantSACCriticDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                                    initialize=initialize, N=equi_n, enc_type=enc_type,
                                                                    backbone=backbone, n_channels=critic_channels).to(device)
        # vector observation
        elif obs_type == 'vec':
            if model == 'cnn':
                actor = SACVecGaussianPolicy(obs_dim, len(action_sequence)).to(device)
                critic = SACVecCritic(obs_dim, len(action_sequence)).to(device)
            elif model == 'equi_both':
                actor = EquivariantSACVecGaussianPolicy(obs_dim=obs_dim, action_dim=len(action_sequence),
                                                        n_hidden=n_hidden, N=equi_n, initialize=initialize).to(device)
                critic = EquivariantSACVecCritic(obs_dim=obs_dim, action_dim=len(action_sequence), n_hidden=n_hidden,
                                                 N=equi_n, initialize=initialize).to(device)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic, not test)

    elif alg in ['sac_reg']:
        sac_lr = (actor_lr, critic_lr)
        if alg == 'sac_reg':
            agent = SACReg(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                           n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                           target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                           model_loss_w=model_loss_w, train_reg=train_reg)
            if model == 'equi_both_d':
                actor = EquivariantSACActorDihedral((obs_channel, crop_size, crop_size), len(action_sequence),
                                                    n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=3).to(device)
                critic = EquivariantSACCriticDihedral((obs_channel, crop_size, crop_size), len(action_sequence),
                                                      n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=3).to(device)
            elif model == 'equi_both_d_w_enc':
                actor = EquivariantSACActorDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                    n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_type='fc').to(device)
                critic = EquivariantSACCriticDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                      n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_type='fc').to(device)

            actor_reward_model = EquivariantRewardModelDihedral(n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            actor_transition_model = EquivariantTransitionModelDihedral(n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            critic_reward_model = EquivariantRewardModelDihedral(n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            critic_transition_model = EquivariantTransitionModelDihedral(n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)

            agent.initNetwork(actor, critic, actor_reward_model, actor_transition_model, critic_reward_model, critic_transition_model)

    elif alg in ['sac_share_enc']:
        sac_lr = (actor_lr, critic_lr)
        if alg == 'sac_share_enc':
            agent = SACShareEnc(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                                n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                                target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
            enc = EquivariantEncoder128Dihedral(obs_channel, n_hidden, initialize, equi_n).to(device)
            actor = EquivariantSACActorDihedralShareEnc(enc, (obs_channel, crop_size, crop_size), len(action_sequence),
                                                        n_hidden=n_hidden, initialize=initialize, N=equi_n,
                                                        kernel_size=3).to(device)
            critic = EquivariantSACCriticDihedralShareEnc(enc, (obs_channel, crop_size, crop_size),
                                                          len(action_sequence), n_hidden=n_hidden,
                                                          initialize=initialize, N=equi_n, kernel_size=3).to(device)
            agent.initNetwork(actor, critic)

    elif alg in ['bc_con']:
        agent = BehaviorCloningContinuous(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                                          n_a=len(action_sequence))

        if model == 'equi':
            policy = EquivariantPolicy((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        elif model == 'equi_enc_2':
            policy = EquivariantPolicy((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, enc_id=2).to(device)
        elif model == 'cnn':
            policy = Actor(len(action_sequence)).to(device)
        elif model == 'equi_both_so2':
            policy = EquivariantPolicySO2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
        elif model == 'equi_both_o2':
            policy = EquivariantPolicyO2((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, kernel_size=3).to(device)
        elif model == 'equi_d':
            policy = EquivariantPolicyDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n, kernel_size=3).to(device)
        elif model == 'equi_both_d_w_enc_ssm':
            policy = EquivariantPolicyDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                             n_hidden=n_hidden, initialize=initialize, N=equi_n,
                                                             enc_type='ssm', backbone='cnn').to(device)

        else:
            raise NotImplementedError
        agent.initNetwork(policy)

    elif alg in ['curl_sac', 'curl_sacfd', 'curl_sacfd_mean']:
        curl_sac_lr = [actor_lr, critic_lr, lr, lr]
        if model == 'equi_both':
            z_dim = n_hidden * equi_n
        elif model == 'equi_both_d_w_enc_ssm':
            z_dim = n_hidden * equi_n * 2
        else:
            z_dim = 50

        if alg == 'curl_sac':
            agent = CURLSAC(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_a=len(action_sequence),
                            tau=tau, alpha=init_temp, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=True,
                            crop_size=crop_size, z_dim=z_dim)
        elif alg == 'curl_sacfd':
            agent = CURLSACfD(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                              n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                              target_update_interval=1, automatic_entropy_tuning=True, crop_size=crop_size,
                              demon_w=demon_w, demon_l='pi', z_dim=z_dim)
        elif alg == 'curl_sacfd_mean':
            agent = CURLSACfD(lr=curl_sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                              n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                              target_update_interval=1, automatic_entropy_tuning=True, crop_size=crop_size,
                              demon_w=demon_w, demon_l='mean', z_dim=z_dim)
        else:
            raise NotImplementedError
        if model == 'cnn':
            actor = CURLSACGaussianPolicy(CURLSACEncoder((obs_channel, crop_size, crop_size)).to(device), action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoder((obs_channel, crop_size, crop_size)).to(device), action_dim=len(action_sequence)).to(device)
        elif model == 'cnn_ssm':
            actor = CURLSACGaussianPolicy(CURLSACEncoder((obs_channel, crop_size, crop_size), output_dim=z_dim, ssm=True).to(device), encoder_output_dim=z_dim, action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoder((obs_channel, crop_size, crop_size), output_dim=z_dim, ssm=True).to(device), encoder_output_dim=z_dim, action_dim=len(action_sequence)).to(device)
        elif model == 'equi_both_d_w_enc_ssm':
            actor = CURLEquiSACActorDihedral(CURLSACEncoder((obs_channel, crop_size, crop_size),
                                                            output_dim=n_hidden * equi_n * 2, ssm=True).to(device),
                                             action_dim=len(action_sequence), n_hidden=n_hidden, initialize=initialize,
                                             N=equi_n).to(device)
            critic = CURLEquiSACCriticDihedral(CURLSACEncoder((obs_channel, crop_size, crop_size),
                                                              output_dim=n_hidden * equi_n * 2, ssm=True).to(device),
                                               action_dim=len(action_sequence), n_hidden=n_hidden, initialize=initialize,
                                               N=equi_n).to(device)
        elif model == 'cnn_2':
            actor = CURLSACGaussianPolicy(CURLSACEncoder2((obs_channel, crop_size, crop_size)).to(device), action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoder2((obs_channel, crop_size, crop_size)).to(device), action_dim=len(action_sequence)).to(device)
        # ferm paper network
        elif model == 'cnn_ferm':
            actor = CURLSACGaussianPolicy(CURLSACEncoderOri((obs_channel, crop_size, crop_size)).to(device),
                                          action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoderOri((obs_channel, crop_size, crop_size)).to(device),
                                   action_dim=len(action_sequence)).to(device)
        elif model == 'equi_both':
            actor = CURLEquiSACGaussianPolicy(CURLEquiSACEncoder((obs_channel, crop_size, crop_size), n_hidden, initialize, equi_n),
                                              n_hidden, len(action_sequence), initialize, equi_n).to(device)
            critic = CURLEquiSACCritic(CURLEquiSACEncoder((obs_channel, crop_size, crop_size), n_hidden, initialize, equi_n),
                                       n_hidden, len(action_sequence), initialize, equi_n).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic)

    elif alg in ['sac_aug']:
        sac_lr = (actor_lr, critic_lr)
        agent = SACAug(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_a=len(action_sequence),
                       tau=tau, alpha=init_temp, policy_type='gaussian', target_update_interval=1, automatic_entropy_tuning=True)
        if model == 'cnn':
            actor = SACGaussianPolicy((obs_channel, 64, 64), len(action_sequence)).to(device)
            critic = SACCritic((obs_channel, 64, 64), len(action_sequence)).to(device)
        elif model == 'equi_both':
            actor = EquivariantSACActor((obs_channel, 64, 64), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            critic = EquivariantSACCritic((obs_channel, 64, 64), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic, not test)


    else:
        raise NotImplementedError
    agent.aug = aug
    agent.aug_type = aug_type
    print('initialized agent')
    return agent