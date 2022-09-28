# Rotation Estimation
This folder holds the code for the rotation estimation experiment
## Requirement
Please skip if you already installed the required packages when running the code under `rl_manipulation`
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
## Data Collection
`python collect_data.py --corrupt=side`
### Selecting Symmetry Corruption
replace `side` with any of
` ` (leave empty for no corruption), `grid`, `occlusion`, `random_light_color`, `light_effect`, `two_light_color`, `two_specular`, `squeeze`, `reflect`, `condition_reverse`, `side`, `side,grid`, `side,light_effect`, `side,squeeze`, `side,reflect`
for other symmetry corruptions.

## Training
` python main.py --device=cuda:1 --aug=t --model=equi_c --n_data=100 --corrupt=side`
### Selecting Symmetry Corruption
replace `side` with any of
` ` (leave empty for no corruption), `grid`, `occlusion`, `random_light_color`, `light_effect`, `two_light_color`, `two_specular`, `squeeze`, `reflect`, `condition_reverse`, `side`, `side,grid`, `side,light_effect`, `side,squeeze`, `side,reflect`
for other symmetry corruptions.
### Running The CNN baseline
replace `equi_c` with `cnn_sim`