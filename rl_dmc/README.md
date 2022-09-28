# RL in DeepMind Control Suite
This folder holds the code for the RL experiments in DeepMind Control Suite (DMC).

The code was forked from the original [DrQv2 repository](https://github.com/facebookresearch/drqv2) and modified to include an equivariant version. The folder `drqv2` contains the code for equivariant and non-equivariant DrQv2.

[DeepMind Control Suite](https://github.com/deepmind/dm_control) was also modified to include different camera angles of the original domains. 

## Requirements
1. Python 3.7+ (tested on 3.9)
2. Install [MuJoCo](https://github.com/deepmind/mujoco) and required libraries
3. Install package dependencies
   ```
   pip install -r requirements.txt
   ```
4. Install the custom dm_control library
   ```
   pip install dm_control
   ```

## Experiments
Navigate into the `drqv2` folder and run commands there. See the `cfgs` folder to see all hyperparameters.

### Original domains
There are 7 tasks: `cartpole_balance, cartpole_swingup, pendulum_swingup, cup_catch, acrobot_swingup, reacher_easy, reacher_hard`. Substite the task you want to run as the argument to `task`.
- Non-equivariant DrQ: 
   ```
   python train.py task=cartpole_balance device=cuda
   ```
- Equivariant DrQv2: 

   For the D_1 environments `cartpole_swingup, cartpole_balance, pendulum_swingup, cup_catch, acrobot_swingup`, run the following command:
   ```
   python train.py task=cartpole_balance device=cuda agent._target_=eq_drqv2.EquiDrQV2Agent agent.encoder_hidden_dim=22 agent.encoder_out_dim=22 agent.hidden_dim=720
   ```

   For the D_2 environments `reacher_easy, reacher_hard`, run the following command:
   ```
   python train.py task=reacher_easy device=cuda agent._target_=eq_drqv2.EquiDrQV2Agent agent.encoder_hidden_dim=16 agent.encoder_out_dim=16 agent.hidden_dim=512
   ```

### Camera angles
- To run camera angle experiments for `cartpole_swingup, cup_catch`, append `_no_grid, _roll_camera1, _roll_camera2, _roll_camera3, _roll_camera4` to the task name.
- To run camera angle experiments for `reacher_hard`, append `_no_grid, _roll_camera1, _roll_camera2, _roll_camera3, _roll_camera4` to the task name.

Example:
```
   python train.py task=cartpole_swingup_roll_camera1 device=cuda agent._target_=eq_drqv2.EquiDrQV2Agent agent.encoder_hidden_dim=22 agent.encoder_out_dim=22 agent.hidden_dim=720
```
