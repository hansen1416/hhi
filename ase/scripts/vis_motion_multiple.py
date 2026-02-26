"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library


Module: vis_motion.py
Description: Visualizes SMPL-based human motions using Isaac Gym. 
Loads multiple SMPL robot models, simulates physics,
and animates motions from a library. Supports keyboard controls for motion switching and debugging.
Dependencies: isaacgym, torch, numpy, joblib, etc.
Usage: Run directly to start the simulation viewer.

"""
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from easydict import EasyDict
from utils.motion_lib_base import FixHeightMode

custom_parameters = [
    {
        "name": "--exclude_faulty",
        "action": "store_true",
        "dest": "exclude_faulty",
        "default": False,
        "help": "Include faulty humanoids (disable default exclusion).",
    },
]

# parse arguments
args = gymutil.parse_arguments(description="Joint monkey: Animate collision box",
                               custom_parameters=custom_parameters,)


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu
sim_params.use_gpu_pipeline = args.use_gpu_pipeline

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# viewer: render collision geometry (equivalent to toggling collision-mesh view in the Viewer tab)
cam_props = gymapi.CameraProperties()
cam_props.use_collision_geometry = True

# create viewer
viewer = gym.create_viewer(sim, cam_props)
if viewer is None:
    print("*** Failed to create viewer")
    quit()

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

asset_root = os.path.join(project_dir, "data", "assets", "mjcf", "smpl")

# Find all SMPL assets
asset_file_list = sorted(glob.glob(os.path.join(asset_root, "*_smpl.xml")))
if not asset_file_list:
    print("*** No SMPL assets found in", asset_root)
    quit()

if args.exclude_faulty:
    exclude_set = [os.path.join(asset_root,"aaab922b_smpl.xml"), os.path.join(asset_root,"6803e1fa_smpl.xml")]

    asset_file_list = [s for s in asset_file_list if s not in exclude_set]

print("Found", len(asset_file_list), "SMPL assets")

num_actors = len(asset_file_list)

# Grid layout (as square as possible)
actor_spacing = 5.0
num_cols = int(np.ceil(np.sqrt(num_actors))) if num_actors > 0 else 1
num_rows = int(np.ceil(num_actors / num_cols)) if num_actors > 0 else 1

# Dynamic camera positioning
half_span_x = (num_cols - 1) * actor_spacing / 2.0 if num_cols > 1 else 0.0
half_span_y = (num_rows - 1) * actor_spacing / 2.0 if num_rows > 1 else 0.0
max_half_span = max(half_span_x, half_span_y)

cam_pos = gymapi.Vec3(0.0, -(max_half_span + 15.0), max_half_span + 8.0)
cam_target = gymapi.Vec3(0.0, 0.0, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

asset_options = gymapi.AssetOptions()
asset_options.angular_damping = 0.01
asset_options.max_angular_velocity = 100.0
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

# set up the env grid (single large env)
num_envs = len(asset_file_list)
env_lower = gymapi.Vec3(-100.0, -100.0, 0.0)
env_upper = gymapi.Vec3(100.0, 100.0, 100.0)

# create env
env = gym.create_env(sim, env_lower, env_upper, 1)

# Create multiple actors
num_actors = len(asset_file_list)
actor_spacing = 4.0  # distance between actors

actor_handles = []
rb_sim_ids_per_actor = []

filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(num_actors):
    asset_path = asset_file_list[i]
    asset_file = os.path.basename(asset_path)
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # Grid position
    col = i % num_cols
    row = i // num_cols
    x_pos = (col - (num_cols - 1) / 2.0) * actor_spacing
    y_pos = (row - (num_rows - 1) / 2.0) * actor_spacing

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(x_pos, y_pos, 1.3)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_name = asset_file.replace(".xml", "")
    actor_handle = gym.create_actor(env, asset, pose, actor_name, 0, 1)
    actor_handles.append(actor_handle)

    # Set default DOF positions (zero / T-pose)
    num_dofs = gym.get_actor_dof_count(env, actor_handle)
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    # Set DOF drive mode to position control
    dof_props = gym.get_actor_dof_properties(env, actor_handle)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    # dof_props["stiffness"].fill(0.0)
    # dof_props["damping"].fill(0.0)

    gym.set_actor_dof_properties(env, actor_handle, dof_props)

    # Set rigid shape properties (collision filters)
    rigid_props = gym.get_actor_rigid_shape_properties(env, actor_handle)

    # SELF_MASK = 1  # any nonzero bit
    # for sp in rigid_props:
    #     sp.filter = SELF_MASK

    # gym.set_actor_rigid_shape_properties(env, actor_handle, rigid_props)
    
    # if num_rb == len(filter_ints):
    #     for p_idx in range(num_rb):
    #         rigid_props[p_idx].filter = filter_ints[p_idx]
    #     gym.set_actor_rigid_shape_properties(env, actor_handle, rigid_props)
    # else:
    #     print(f"Warning: Rigid body count mismatch ({num_rb} vs {len(filter_ints)}) for {asset_file}. Skipping filter setting.")

    num_rb = len(rigid_props)

    # Initialize collision visualization to green
    for rb_idx in range(num_rb):
        gym.set_rigid_body_color(env, actor_handle, rb_idx, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

    # Cache SIM-domain rigid body indices for contact visualization
    rb_sim_ids = [
        gym.get_actor_rigid_body_index(env, actor_handle, rb_idx, gymapi.DOMAIN_SIM)
        for rb_idx in range(num_rb)
    ]
    rb_sim_ids_per_actor.append(rb_sim_ids)

# Acquire contact force tensor after all actors are created (shape depends on total rigid bodies)
net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf_t = gymtorch.wrap_tensor(net_cf)  # shape: (total_rigid_bodies, 3)

gym.prepare_sim(sim)

device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))

rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(actor_root_state)

motion_id = 0
time_step = 0

env_ids = torch.arange(num_envs).int().to(args.sim_device)

gym.refresh_actor_root_state_tensor(sim)
gym.refresh_rigid_body_state_tensor(sim)
gym.refresh_dof_state_tensor(sim)

t_idx = 0
tota_steps = 1000

contact_thresh = 1.0

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_net_contact_force_tensor(sim)
    mags = torch.linalg.vector_norm(net_cf_t, dim=1)

    # Update contact visualization for all actors
    for a_idx, actor_handle in enumerate(actor_handles):
        rb_sim_ids = rb_sim_ids_per_actor[a_idx]
        for rb_local, rb_sim in enumerate(rb_sim_ids):
            in_contact = mags[rb_sim].item() > contact_thresh
            color = gymapi.Vec3(1, 0, 0) if in_contact else gymapi.Vec3(0, 1, 0)
            gym.set_rigid_body_color(env, actor_handle, rb_local, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    gym.sync_frame_time(sim)
    
    t_idx += 1

    if t_idx >= tota_steps:
        t_idx = 0

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)