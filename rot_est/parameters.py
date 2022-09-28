import numpy as np
import torch
import argparse

def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def noneOrStr(value):
    if value == 'None':
        return None
    return value

parser = argparse.ArgumentParser()
env_group = parser.add_argument_group('environment')
env_group.add_argument('--num_processes', type=int, default=20)
env_group.add_argument('--render', type=strToBool, default=False)
env_group.add_argument('--workspace_size', type=float, default=0.3)
env_group.add_argument('--heightmap_size', type=int, default=152)
env_group.add_argument('--corrupt', type=str, default='')

training_group = parser.add_argument_group('training')
training_group.add_argument('--model', type=str, default='resucat')
training_group.add_argument('--lr', type=float, default=1e-4)
training_group.add_argument('--seed', type=int, default=None)
training_group.add_argument('--crop_size', type=int, default=128)
training_group.add_argument('--aug', type=strToBool, default=False)
training_group.add_argument('--device', type=str, default='cuda')
training_group.add_argument('--n_data', type=int, default=1000)
training_group.add_argument('--n_img_trans', type=int, default=0)
training_group.add_argument('--batch_size', type=int, default=64)
training_group.add_argument('--load_buffer', type=str, default=None)

logging_group = parser.add_argument_group('logging')
logging_group.add_argument('--log_pre', type=str, default='/tmp')
logging_group.add_argument('--log_sub', type=str, default=None)
logging_group.add_argument('--no_bar', action='store_true')
logging_group.add_argument('--time_limit', type=float, default=10000)
logging_group.add_argument('--load_sub', type=str, default=None)

args = parser.parse_args()
# env
env = 'close_loop_duck'
simulator = 'pybullet'
num_processes = args.num_processes
render = args.render

workspace_size = args.workspace_size
workspace = np.asarray([[0.45-workspace_size/2, 0.45+workspace_size/2],
                        [0-workspace_size/2, 0+workspace_size/2],
                        [0.01, 0.25]])
heightmap_size = args.heightmap_size

heightmap_resolution = workspace_size/heightmap_size

corrupt = args.corrupt
corrupt = [str(c) for c in corrupt.split(',')]

######################################################################################
# training
model = args.model
lr = args.lr
seed = args.seed
crop_size = args.crop_size
aug = args.aug
device = args.device
n_data = args.n_data
n_img_trans = args.n_img_trans
batch_size = args.batch_size
load_buffer = args.load_buffer

# logging
log_pre = args.log_pre
log_sub = args.log_sub
no_bar = args.no_bar
time_limit = args.time_limit
load_sub = args.load_sub
if load_sub == 'None':
    load_sub = None

######################################################################################
env_config = {'workspace': workspace, 'obs_size': heightmap_size, 'render': render,  'robot': 'empty',
              'workspace_check': 'point', 'hard_reset_freq': 1000, 'physics_mode' : 'fast', 'corrupt': corrupt}
planner_config = {}
if seed is not None:
    env_config['seed'] = seed
######################################################################################
hyper_parameters = {}
for key in sorted(vars(args)):
    hyper_parameters[key] = vars(args)[key]

for key in hyper_parameters:
    print('{}: {}'.format(key, hyper_parameters[key]))