"""
ase/data/motions/amp_humanoid_walk.npy
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
tensor([ 7,  3, 22, 17], device='cuda:0')
"""

import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import torch
from easydict import EasyDict

from utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree

device = 'cuda:0'

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
motion_file = os.path.join(project_dir, "data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl")

# _dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# _dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]
# _key_body_ids = torch.tensor([ 7,  3, 22, 17], device=device)

# in phc setting, we have 24 keypoints, maybe for humos training, we reduce to 22, exclude hands.
_key_body_ids = torch.tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23], device=device)

asset_file = os.path.join(project_dir, "./data/assets/mjcf/smpl_humanoid.xml")

sk_tree = SkeletonTree.from_mjcf(asset_file)

gender_beta = np.zeros(17)
num_envs = 1

humanoid_shapes = torch.tensor(np.array([gender_beta] * num_envs)).float().to(device)


motion_lib_cfg = EasyDict({
                "motion_file": motion_file,
                "device": torch.device("cpu"),
                "fix_height": 1,
                "min_length": -1,
                "max_length": -1,
                "im_eval": True,
                "multi_thread": False,
                "smpl_type": "smpl",
                "randomrize_heading": True,
                "device": device,
                "min_length": -1, 
                "step_dt": 1/60,
                "key_body_ids": _key_body_ids
            })



motion_lib = MotionLibSMPL(motion_lib_cfg=motion_lib_cfg)
motion_lib.load_motions(skeleton_trees=[sk_tree], 
                        gender_betas=humanoid_shapes.cpu(), 
                        random_sample=True)

# tensor([0], device='cuda:0')
motion_ids = motion_lib.sample_motions(num_envs)
# tensor([2.9572], device='cuda:0')
motion_times = motion_lib.sample_time(motion_ids, truncate_time=0.0)

motion_res = motion_lib.get_motion_state(motion_ids, motion_times)

root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
        = (motion_res["root_pos"],
        motion_res["root_rot"],
        motion_res["dof_pos"],
        motion_res["root_vel"],
        motion_res["root_ang_vel"],
        motion_res["dof_vel"],
        motion_res["key_pos"])

# root_pos torch.Size([1, 3])
print("root_pos", root_pos.shape)
# root_rot torch.Size([1, 4])
print("root_rot", root_rot.shape)
# dof_pos torch.Size([1, 69])
print("dof_pos", dof_pos.shape)
# root_vel torch.Size([1, 3])
print("root_vel", root_vel.shape)
# root_ang_vel torch.Size([1, 3])
print("root_ang_vel", root_ang_vel.shape)
# dof_vel torch.Size([1, 69])
print("dof_vel", dof_vel.shape)
# key_pos torch.Size([1, 24, 3])
print("key_pos", key_pos.shape)


rg_pos = motion_res["rg_pos"]
rb_rot = motion_res["rb_rot"]
body_vel = motion_res["body_vel"]
body_ang_vel = motion_res["body_ang_vel"]

print("rg_pos", rg_pos.shape)
print("rb_rot", rb_rot.shape)
print("body_vel", body_vel.shape)
print("body_ang_vel", body_ang_vel.shape)
