# Heteromorphic Humanoid Imitation

Code accompanying the paper:
"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" \
(https://xbpeng.github.io/projects/ASE/index.html)


### Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```

## Train

python ase/run.py --task HumanoidHHI --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/ --headless

python ase/run.py --task HumanoidHHI --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/003628_female_093098f0.pkl --headless

<!-- test transfer learning -->
python ase/run.py --task HumanoidHHI --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/003628_female_093098f0.pkl --checkpoint ~/Downloads/film_256_32_288_64.pth --headless

python ase/run.py --task HumanoidHHITraj --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/003628_female_093098f0.pkl --headless

python ase/run.py --task HumanoidTransfer --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/transfer_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/003628_female_093098f0.pkl --headless

python ase/run.py --task HumanoidTransfer --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/transfer_humanoid.yaml --motion_file /home/hlz/datasets/humos_results/002175_female_0e26b88d.pkl --checkpoint ~/Downloads/transfer_256_32_288_64.pth --headless

--

## Test

python ase/run.py --test --task HumanoidHHI --num_envs 16 --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/hlz/datasets/288_64_test --checkpoint /home/hlz/Downloads/film_256_32_288_64.pth

python ase/run.py --test --task HumanoidHHI --num_envs 16 --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/hlz/datasets/simple_walk_motions --checkpoint /home/hlz/Downloads/film_simple_walk_new_reward.pth

python ase/run.py --test --task HumanoidTransfer --num_envs 16 --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/transfer_humanoid.yaml --motion_file /home/hlz/datasets/288_64_test --checkpoint /home/hlz/Downloads/transfer_256_32_288_64.pth

# This is the lates, looks pretty solid, only a bit twitching
`
python ase/run.py --test --task HumanoidHHI --num_envs 16 --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/hlz/datasets/humos_results_test --checkpoint /home/hlz/Downloads/hhi_film_0419.pth
`


## Visual HUMOS results in PHC format

`
python ase/run.py --task HumanoidViewMotion --num_envs 4 --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/hlz/datasets/humos_results_visual/
`

## Tensrolog

# 1) Create a new conda env with Python 3.8.10
conda create -n tb_env python=3.8.10 -y

# 2) Activate the environment
conda activate tb_env

# 3) Install tensorboard (and pin protobuf to avoid your previous error)
pip install "tensorboard" "protobuf<5"


### note


- load 128 of humanoid, see if their betas covers them all.


- /humanoid_hhi.py: computes *imitation reward* (compute_imitation_reward) + optional power penalty; writes rew_buf`

- learning/hhi_agent.py: overwrites the learning signal by mixing `task_reward_w` and `disc_reward_w` in _combine_rewards().

- ase/data/cfg/train/rlg/hhi_humanoid.yaml: shows `task_reward_w`: 0.0 and `disc_reward_w`: 1.0 (so PPO is not optimizing imitation reward unless you changed this).
    - (PHC use task_reward_w: 0.5, disc_reward_w: 0.5)

    - Log raw reward components (reward_raw): pos/rot/vel/ang_vel + power. Is pos/rot high (~0.8-1.0) but vel/ang_vel low? Indicates static bias.

    - Experiment: Scale weights (e.g., increase w_vel to 0.2 in config). Retrain short run; if reward improves, tune.

- utils/motion_lib_humos.py: dataset loading (load_data) and GPU-side motion loading (load_motions).

- env/tasks/humanoid_hhi.py: calls _motion_lib.sample_motions() at reset; builds reference at time `t`

    Print/log at runtime (once per epoch):

    _motion_lib._num_unique_motions (dataset size)

    _motion_lib._num_motions (actually loaded motions available for sampling)

    distribution of _sampled_motion_ids over time (should cover all 256 variants)

-------

## reward

Adjust the scaling constants (k values) and weights (w values) in self.reward_specs to match PHC's softer penalties.

self.reward_specs = cfg["env"].get("reward_specs", {
    "k_pos": 2.0,       # Lower from 100; PHC uses ~2 for positions
    "k_rot": 0.2,       # Lower from 10; PHC uses ~0.2 for rotations
    "k_vel": 0.1,       # Keep or slightly adjust; PHC uses ~0.1
    "k_ang_vel": 0.1,   # Keep or slightly adjust; PHC uses ~0.1
    "w_pos": 0.5,       # PHC: 0.5 for positions
    "w_rot": 0.3,       # PHC: 0.3 for rotations (or 0.2 if adding end-effector term)
    "w_vel": 0.15,      # PHC: 0.15 for velocities
    "w_ang_vel": 0.05   # PHC: 0.05 for angular velocities
})

-------

### Help

`export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH`

`python ase/run.py --test --task HumanoidAMPGetup --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth`

`python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth`

`python ase/run.py --task HumanoidAMPGetup --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --headless`

`pip install "protobuf==3.20.*"`

------------------

find out how amp save motion result,
try to play the motion in AMP settings.
copy the red ball logic from phc

use smpl sim to generate humanoid of different shapes and density, (shape follow the density)
common_player -> amp_player; humanoid -> humanoid_amp; find out the initial position settings

----
