from tqdm import tqdm
import sys
sys.path.append('..')
from rot_est.parameters import *
from rot_est.env_wrapper import EnvWrapper

import torch

def collectData():
    envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)

    n_data = 2000
    buffer = []
    pbar = tqdm(total=n_data)
    while len(buffer) < n_data:
        state, obs = envs.reset()
        for i in range(num_processes):
            buffer.append((obs[i], state[i]))
            pbar.update(1)
            if len(buffer) >= n_data:
                break

    torch.save(buffer, 'trans_est_C8_{}_{}_{}_{}.pt'.format(env, corrupt, heightmap_size, n_data))
    envs.close()
    del buffer

if __name__ == '__main__':
    global corrupt
    # for corrupt in [[''], ['grid'], ['occlusion'], ['random_light_color'], ['light_effect'], ['two_light_color'],
    #                 ['two_specular'], ['squeeze'], ['reflect'], ['reflect', 'grid'], ['random_reflect'],
    #                 ['random_reflect', 'grid'], ['side'], ['side' ,'grid'], ['side', 'light_effect'],
    #                 ['side', 'squeeze'], ['side', 'reflect'], ['side', 'reflect', 'grid'], ['side', 'random_reflect'],
    #                 ['side', 'random_reflect', 'grid']]:
    # for corrupt in [['side']]:
    #     env_config['corrupt'] = corrupt
    env_config['seed'] = 0
    collectData()

