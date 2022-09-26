# RL in Manipulation
This folder holds the code for the manipulation RL experiment
## Requirement
Please skip if you already installed the required packages when running the code under `rot_est`
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
1. Create and activate conda environment
    ```
    conda create --name equi python=3.9
    conda activate equi
    ```
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.10.2, torchvision==0.11.3):
   ```
   conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 -c pytorch
   ```
1. Install other required packages 
   ```
   pip3 install -r requirements.txt
   ```

## Running Equivariant SAC in Block Picking
```
python main.py --env=close_loop_block_picking --num_objects=1 --alg=sacfd --model=equi_both_d --max_train_step=5000 --planner_episode=50 --device=cuda:1 --view_type=camera_side_rgbd --aug=t --seed=1
```
### Running the baselines
- CNN + RAD: 
   ```
   python main.py --env=close_loop_block_picking --num_objects=1 --alg=sacfd --model=cnn_sim --max_train_step=5000 --planner_episode=50 --device=cuda:1 --view_type=camera_side_rgbd --aug=t --seed=1
   ```
- CNN + DrQ: 
   ```
   python main.py --env=close_loop_block_picking --num_objects=1 --alg=sacfd_drq --model=cnn_sim --max_train_step=5000 --planner_episode=50 --device=cuda:1 --view_type=camera_side_rgbd --aug=f --seed=1
   ```
- FERM: 
   ```
   python main.py --env=close_loop_block_picking --num_objects=1 --alg=curl_sacfd --model=cnn_sim_2 --max_train_step=5000 --planner_episode=50 --device=cuda:1 --view_type=camera_side_rgbd --aug=f --seed=1
   ```
- SEN + RAD: 
   ```
   python main.py --env=close_loop_block_picking --num_objects=1 --alg=sacfd --model=sen_fc --max_train_step=5000 --planner_episode=50 --device=cuda:1 --view_type=camera_side_rgbd --aug=t --seed=1
   ```
### Running in Other Environments
Replace `--env=close_loop_block_picking --num_objects=1` with:
- Block Pushing: `--env=close_loop_block_pushing --num_objects=1`. Also replace `--planner_episode=50` with `--planner_episode=20` to run with 20 expert demos.
- Block Pulling: `--env=close_loop_block_pulling --num_objects=2`. Also replace `--planner_episode=50` with `--planner_episode=20` to run with 20 expert demos.
- Drawer Opening: `--env=close_loop_drawer_opening --num_objects=1`
- Block in Bowl: `--env=close_loop_block_in_bowl --num_objects=2`. Also replace `--max_train_step=5000` with `--max_train_step=10000` to run 10k time steps.