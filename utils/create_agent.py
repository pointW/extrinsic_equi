from utils.parameters import *

from agents.sac import SAC
from agents.sacfd import SACfD
from agents.curl_sac import CURLSAC
from agents.curl_sacfd import CURLSACfD
from agents.sac_drq import SACDrQ
from agents.sacfd_drq import SACfDDrQ
from networks.equivariant_sac_net import EquivariantSACActorDihedral, EquivariantSACCriticDihedral
from networks.curl_sac_net import CURLSACCritic, CURLSACGaussianPolicy, CURLSACEncoderOri, CURLSACEncoder2, CURLSACEncoder3
from networks.equivariant_sac_net import EquivariantSACCriticDihedralWithNonEquiEnc, EquivariantSACActorDihedralWithNonEquiEnc
from networks.sac_networks import SACGaussianPolicyFullyConv, SACCriticFullyConv
from networks.sac_networks import SACCriticSimFC, SACGaussianPolicySimFC

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

    # setup agent
    if alg in ['sac', 'sacfd', 'sac_drq', 'sacfd_drq']:
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
        elif alg == 'sac_drq':
            agent = SACDrQ(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                           n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                           target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type)
        elif alg == 'sacfd_drq':
            agent = SACfDDrQ(lr=sac_lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot,
                             n_a=len(action_sequence), tau=tau, alpha=init_temp, policy_type='gaussian',
                             target_update_interval=1, automatic_entropy_tuning=True, obs_type=obs_type,
                             demon_w=demon_w)
        else:
            raise NotImplementedError
        # pixel observation
        if obs_type == 'pixel':
            if model == 'cnn_sim':
                actor = SACGaussianPolicyFullyConv((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = SACCriticFullyConv((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            elif model == 'cnn_sim_fc':
                actor = SACGaussianPolicySimFC((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
                critic = SACCriticSimFC((obs_channel, crop_size, crop_size), len(action_sequence)).to(device)
            elif model == 'cnn_sim_fc_2':
                actor = SACGaussianPolicySimFC((obs_channel, crop_size, crop_size), len(action_sequence), enc_id=2).to(device)
                critic = SACCriticSimFC((obs_channel, crop_size, crop_size), len(action_sequence), enc_id=2).to(device)
            elif model == 'equi_both_d':
                actor = EquivariantSACActorDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
                critic = EquivariantSACCriticDihedral((obs_channel, crop_size, crop_size), len(action_sequence), n_hidden=n_hidden, initialize=initialize, N=equi_n).to(device)
            elif model == 'sen_fc':
                actor = EquivariantSACActorDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                                  n_hidden=64, initialize=initialize, N=equi_n, enc='fc').to(device)
                critic = EquivariantSACCriticDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                                    n_hidden=64, initialize=initialize, N=equi_n, enc='fc').to(device)
            elif model == 'sen_fc_2':
                actor = EquivariantSACActorDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                                  n_hidden=32, initialize=initialize, N=equi_n, enc='fc_2').to(device)
                critic = EquivariantSACCriticDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                                    n_hidden=32, initialize=initialize, N=equi_n, enc='fc_2').to(device)
            elif model == 'sen_conv':
                actor = EquivariantSACActorDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                                  n_hidden=64, initialize=initialize, N=equi_n, enc='conv').to(device)
                critic = EquivariantSACCriticDihedralWithNonEquiEnc((obs_channel, crop_size, crop_size), len(action_sequence),
                                                                    n_hidden=64, initialize=initialize, N=equi_n, enc='conv').to(device)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic, not test)

    elif alg in ['curl_sac', 'curl_sacfd', 'curl_sacfd_mean']:
        curl_sac_lr = [actor_lr, critic_lr, lr, lr]
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
         # more weights in conv
        if model == 'cnn_sim':
            actor = CURLSACGaussianPolicy(CURLSACEncoder2((obs_channel, crop_size, crop_size)).to(device), hidden_dim=256, action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoder2((obs_channel, crop_size, crop_size)).to(device), hidden_dim=256, action_dim=len(action_sequence)).to(device)
        # more weights in fc
        elif model == 'cnn_sim_2':
            actor = CURLSACGaussianPolicy(CURLSACEncoder3((obs_channel, crop_size, crop_size)).to(device), hidden_dim=512, action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoder3((obs_channel, crop_size, crop_size)).to(device), hidden_dim=512, action_dim=len(action_sequence)).to(device)
        # ferm paper network
        elif model == 'cnn_ferm':
            actor = CURLSACGaussianPolicy(CURLSACEncoderOri((obs_channel, crop_size, crop_size)).to(device),
                                          action_dim=len(action_sequence)).to(device)
            critic = CURLSACCritic(CURLSACEncoderOri((obs_channel, crop_size, crop_size)).to(device),
                                   action_dim=len(action_sequence)).to(device)
        else:
            raise NotImplementedError
        agent.initNetwork(actor, critic)
    else:
        raise NotImplementedError
    agent.aug = aug
    agent.aug_type = aug_type
    print('initialized agent')
    return agent