import numpy as np
import torch
import argparse

def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()
env_group = parser.add_argument_group('environment')
env_group.add_argument('--env', type=str, default='block_stacking', help='block_picking, block_stacking, brick_stacking, '
                                                                         'brick_inserting, block_cylinder_stacking')
env_group.add_argument('--reward_type', type=str, default='sparse')
env_group.add_argument('--simulator', type=str, default='pybullet')
env_group.add_argument('--robot', type=str, default='kuka')
env_group.add_argument('--num_objects', type=int, default=3)
env_group.add_argument('--max_episode_steps', type=int, default=10)
env_group.add_argument('--fast_mode', type=strToBool, default=True)
env_group.add_argument('--action_sequence', type=str, default='pxyzr')
env_group.add_argument('--random_orientation', type=strToBool, default=True)
env_group.add_argument('--num_processes', type=int, default=5)
env_group.add_argument('--render', type=strToBool, default=False)
env_group.add_argument('--workspace_size', type=float, default=0.4)
env_group.add_argument('--heightmap_size', type=int, default=128)

training_group = parser.add_argument_group('training')
training_group.add_argument('--alg', default='dqn')
training_group.add_argument('--model', type=str, default='resucat')
training_group.add_argument('--lr', type=float, default=5e-5)
training_group.add_argument('--gamma', type=float, default=0.9)
training_group.add_argument('--explore', type=int, default=10000)
training_group.add_argument('--fixed_eps', action='store_true')
training_group.add_argument('--init_eps', type=float, default=1.0)
training_group.add_argument('--final_eps', type=float, default=0.)
training_group.add_argument('--training_iters', type=int, default=1)
training_group.add_argument('--training_offset', type=int, default=1000)
training_group.add_argument('--max_episode', type=int, default=50000)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--target_update_freq', type=int, default=100)
training_group.add_argument('--save_freq', type=int, default=500)
training_group.add_argument('--action_selection', type=str, default='egreedy')
training_group.add_argument('--load_model_pre', type=str, default=None)
training_group.add_argument('--planner_episode', type=int, default=0)
training_group.add_argument('--note', type=str, default=None)
training_group.add_argument('--seed', type=int, default=None)
training_group.add_argument('--perlin', type=float, default=0.0)
training_group.add_argument('--load_buffer', type=str, default=None)
training_group.add_argument('--load_n', type=int, default=1000000)
training_group.add_argument('--pre_train_step', type=int, default=0)

margin_group = parser.add_argument_group('margin')
margin_group.add_argument('--margin', default='l', choices=['ce', 'bce', 'bcel', 'l', 'oril'])
margin_group.add_argument('--margin_l', type=float, default=0.1)
margin_group.add_argument('--margin_weight', type=float, default=0.1)
margin_group.add_argument('--margin_beta', type=float, default=100)

buffer_group = parser.add_argument_group('buffer')
buffer_group.add_argument('--buffer', default='normal', choices=['normal', 'per', 'expert', 'per_expert'])
buffer_group.add_argument('--per_eps', type=float, default=1e-6, help='Epsilon parameter for PER')
buffer_group.add_argument('--per_alpha', type=float, default=0.6, help='Alpha parameter for PER')
buffer_group.add_argument('--per_beta', type=float, default=0.4, help='Initial beta parameter for PER')
buffer_group.add_argument('--per_expert_eps', type=float, default=0.1)
buffer_group.add_argument('--batch_size', type=int, default=32)
buffer_group.add_argument('--buffer_size', type=int, default=100000)

logging_group = parser.add_argument_group('logging')
logging_group.add_argument('--log_pre', type=str, default='/tmp')
logging_group.add_argument('--log_sub', type=str, default=None)
logging_group.add_argument('--no_bar', action='store_true')
logging_group.add_argument('--time_limit', type=float, default=10000)
logging_group.add_argument('--load_sub', type=str, default=None)

test_group = parser.add_argument_group('test')
test_group.add_argument('--test', action='store_true')

args = parser.parse_args()
# env
random_orientation = args.random_orientation
reward_type = args.reward_type
env = args.env
simulator = args.simulator
num_objects = args.num_objects
max_episode_steps = args.max_episode_steps
fast_mode = args.fast_mode
action_sequence = args.action_sequence
num_processes = args.num_processes
render = args.render
robot = args.robot


workspace_size = args.workspace_size
workspace = np.asarray([[0.5-workspace_size/2, 0.5+workspace_size/2],
                        [0-workspace_size/2, 0+workspace_size/2],
                        [0.01, 0.25]])
heightmap_size = args.heightmap_size

heightmap_resolution = workspace_size/heightmap_size
action_space = [0, heightmap_size]

######################################################################################
# training
alg = args.alg
model = args.model
lr = args.lr
gamma = args.gamma
explore = args.explore
fixed_eps = args.fixed_eps
init_eps = args.init_eps
final_eps = args.final_eps
training_iters = args.training_iters
training_offset = args.training_offset
max_episode = args.max_episode
device = torch.device(args.device_name)
target_update_freq = args.target_update_freq
save_freq = args.save_freq
action_selection = args.action_selection
planner_episode = args.planner_episode

load_model_pre = args.load_model_pre
is_test = args.test
note = args.note
seed = args.seed
perlin = args.perlin

# pre train
load_buffer = args.load_buffer
load_n = args.load_n
pre_train_step = args.pre_train_step

# buffer
buffer_type = args.buffer
per_eps = args.per_eps
per_alpha = args.per_alpha
per_beta = args.per_beta
per_expert_eps = args.per_expert_eps
batch_size = args.batch_size
buffer_size = args.buffer_size

# margin
margin = args.margin
margin_l = args.margin_l
margin_weight = args.margin_weight
margin_beta = args.margin_beta

# logging
log_pre = args.log_pre
log_sub = args.log_sub
no_bar = args.no_bar
time_limit = args.time_limit
load_sub = args.load_sub
if load_sub == 'None':
    load_sub = None

######################################################################################
env_config = {'workspace': workspace, 'max_steps': max_episode_steps, 'obs_size': heightmap_size,
              'fast_mode': fast_mode,  'action_sequence': action_sequence, 'render': render, 'num_objects': num_objects,
              'random_orientation':random_orientation, 'reward_type': reward_type, 'robot': robot,
              'workspace_check': 'point', 'object_scale_range': (1, 1),
              'hard_reset_freq': 1000, 'physics_mode' : 'fast'}
planner_config = {'random_orientation':random_orientation,}
if seed is not None:
    env_config['seed'] = seed
######################################################################################
hyper_parameters = {}
for key in sorted(vars(args)):
    hyper_parameters[key] = vars(args)[key]

for key in hyper_parameters:
    print('{}: {}'.format(key, hyper_parameters[key]))