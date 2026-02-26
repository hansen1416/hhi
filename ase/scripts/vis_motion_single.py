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
oads an SMPL robot model, simulates physics,
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

import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonTree

# parse arguments
args = gymutil.parse_arguments(description="Joint monkey: Animate collision box",
                               custom_parameters=[])
    # parser.add_argument("--contact_thresh", type=float, default=1.0, help="N; color red if ||F_contact|| > thresh")


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim_params.physx.solver_type = 1
sim_params.physx.num_threads = 0
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0

# sim_params.physx.contact_offset = 0.005  # Default 0.02; smaller detects sooner
# sim_params.physx.rest_offset = 0.001  # Default 0.0; small positive reduces jitter
# sim_params.physx.bounce_threshold_velocity = 0.2  # Reduce bouncing
# sim_params.physx.max_depenetration_velocity = 0.0001  # Limit penetration correction speed
# sim_params.physx.default_buffer_size_multiplier = 5.0  # Increase for complex contacts
# sim_params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP  # CC_LAST (more accurate but slower)

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

asset_file = "aaab922b_smpl.xml"
asset_file = "6803e1fa_smpl.xml"

# asset_file = "638a4fb7_smpl.xml"


sk_tree = SkeletonTree.from_mjcf(osp.join(asset_root, asset_file))

asset_options = gymapi.AssetOptions()
asset_options.angular_damping = 0.01
asset_options.max_angular_velocity = 100.0
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
num_per_row = 5
spacing = 5
env_lower = gymapi.Vec3(-spacing, spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(3.0, 0.0, 2)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []

num_dofs = gym.get_asset_dof_count(asset)

# create env
env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)

# add actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.,  0.,  1.0)
pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

actor_handle = gym.create_actor(env, asset, pose, "actor", 0, 1)

# visulize collision box -----------
nb = gym.get_actor_rigid_body_count(env, actor_handle)
rb_sim_ids = [
    gym.get_actor_rigid_body_index(env, actor_handle, i, gymapi.DOMAIN_SIM)
    for i in range(nb)
]

# initialize to green
for i in range(nb):
    gym.set_rigid_body_color(env, actor_handle, i, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))

# net contact forces (one Vec3 per rigid body in SIM domain)
net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf_t = gymtorch.wrap_tensor(net_cf)  # shape: (num_rigid_bodies, 3)
# visulize collision box -----------


# set default DOF positions
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_POS)            # PD position mode
# props["stiffness"].fill(10.0)
# props["damping"].fill(2.0)
# Reasonable generic gains (tune to match training if needed)

gym.set_actor_dof_properties(env, actor_handle, props) 

rigid_props = gym.get_actor_rigid_shape_properties(env, actor_handle)

filter_ints = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for p_idx in range(len(rigid_props)):
    rigid_props[p_idx].filter = filter_ints[p_idx]

gym.set_actor_rigid_shape_properties(env, actor_handle, rigid_props)

gym.prepare_sim(sim)


device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))

rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(actor_root_state)

motion_id = 0
time_step = 0

# tensor([0], device='cuda:0', dtype=torch.int32)
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

    for rb_local, rb_sim in enumerate(rb_sim_ids):
        in_contact = mags[rb_sim].item() > contact_thresh
        color = gymapi.Vec3(1, 0, 0) if in_contact else gymapi.Vec3(0, 1, 0)
        gym.set_rigid_body_color(env, actor_handle, rb_local, gymapi.MESH_VISUAL_AND_COLLISION, color)


    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)


    gym.sync_frame_time(sim)
    
    # time_step += dt
    t_idx += 1

    if t_idx >= tota_steps:
        t_idx = 0


print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
