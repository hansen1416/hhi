import numpy as np
import os
import torch
import getpass

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils
from env.tasks.base_task import BaseTask
from env.features.target_marker import TargetMarkerFeature
from env.features.follow_camera import FollowCameraFeature

USER = getpass.getuser()

SMPL_MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 
                     'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
SMPLH_MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 
                      'L_Wrist', 'L_Index1', 'L_Index2', 'L_Index3', 'L_Middle1', 'L_Middle2', 'L_Middle3', 'L_Pinky1', 'L_Pinky2', 'L_Pinky3', 'L_Ring1', 'L_Ring2', 'L_Ring3', 'L_Thumb1', 'L_Thumb2', 'L_Thumb3', 
                      'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Index1', 'R_Index2', 'R_Index3', 'R_Middle1', 'R_Middle2', 'R_Middle3', 'R_Pinky1', 'R_Pinky2', 'R_Pinky3', 'R_Ring1', 'R_Ring2', 'R_Ring3', 'R_Thumb1', 'R_Thumb2', 'R_Thumb3']

def mat33_to_np(m):
    # m: gymapi.Mat33
    return np.array([
        [m.x.x, m.x.y, m.x.z],
        [m.y.x, m.y.y, m.y.z],
        [m.z.x, m.z.y, m.z.z],
    ], dtype=np.float32)

def encode_gender(gender_str: str, device='cpu', dtype=torch.float32):
    if gender_str == 'male':
        return torch.tensor([1], device=device, dtype=dtype)
    elif gender_str == 'female':
        return torch.tensor([-1], device=device, dtype=dtype)
    
    return torch.tensor([0], device=device, dtype=dtype)

class Humanoid(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        if not hasattr(self, "target_marker_enabled"):
            self.target_marker_enabled = False

        if not hasattr(self, "follow_camera_enabled"):
            self.follow_camera_enabled = True

        # fetures plugin -------------
        self._features = [TargetMarkerFeature(enabled=self.target_marker_enabled), FollowCameraFeature(enabled=self.follow_camera_enabled)]
        # fetures plugin -------------

        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]
        self.all_betas = torch.load(os.path.join(self.asset_root, self.cfg["env"]["asset"]["assetFileName"]), weights_only=False)
         
        super().__init__(cfg=self.cfg)
        
        self.dt = self.control_freq_inv * sim_params.dt
        
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # multi humanoid template change ===============
        self.force_sensor_joints = cfg["env"].get("force_sensor_joints", ["L_Ankle", "R_Ankle"]) # force tensor joints
        sensors_per_env = len(self.force_sensor_joints)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()
        
        # is a view (no copy) that reshapes self._root_states to [num_envs, num_actors_per_env, 13] 
        # and then selects the 0-th actor per env (the humanoid):
        # Because it is a view, in-place writes to self._humanoid_root_states[:, 0:3/3:7/…] 
        # directly modify the corresponding slice of self._root_states.
        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        
        # fetures plugin -------------
        for f in self._features: f.on_post_init_tensors(self)
        # fetures plugin -------------

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]
   
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = rigid_body_state_reshaped[..., :self.num_bodies, 10:13]

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]
        
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        
        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contactBodies"]
        # `self._key_body_ids` later used to compute `_compute_amp_observations`
        # also MotionLib is configured to output only those key bodies `self._key_body_ids`
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        return

    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions

    def get_num_actors_per_env(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        return num_actors

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        return

    def reset(self, env_ids=None):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._reset_envs(env_ids)
        return

    def set_char_color(self, col, env_ids):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.humanoid_handles[env_id]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _reset_envs(self, env_ids):
        raise NotImplementedError(
            "Base Humanoid reset removed. Use HumanoidPHC."
        )
    
    def _reset_actors(self, env_ids):
        raise NotImplementedError(
        "Base Humanoid reset removed. Use HumanoidPHC."
    )

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _setup_character_props(self, key_bodies):
        # multi humanoid template change ===============
        self._body_names = SMPL_MUJOCO_NAMES
        self._dof_names = self._body_names[1:]

        # ankle joints are the lowest articulated joints
        # self.force_sensor_joints = ["L_Ankle", "R_Ankle"]
        self._dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self._dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]

        self._dof_obs_size = len(self._dof_names) * 6
        self._dof_size = len(self._dof_names) * 3

        self._num_actions = len(self._dof_names) * 3

        # some conditions for `self._num_obs`, burrowed from PHC
        self._root_height_obs = True
        self.self_obs_v = 1

        # height + num_bodies * 15 (pos + vel + rot + ang_vel) - root_pos
        self._num_obs = 1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3
        # gender + betas
        self._num_obs += 11

        # load beta into observation ===============
        if not self._root_height_obs:
            self._num_obs -= 1
        
        if self.self_obs_v == 3:
            self._num_obs += 6 * len(self.force_sensor_joints)

        return

    def _build_termination_heights(self):
        head_term_height = 0.3

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.humanoid_handles[0], "head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])
        
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        # fetures plugin -------------
        for f in self._features: f.on_create_envs(self, num_envs)
        # fetures plugin -------------

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # # ---- 1211 actions robust convex decomposition & inertia overrides ----
        # # Use VHACD (volumetric hierarchical approximate convex decomposition)
        # # so that arbitrary meshes are approximated by a set of convex shapes.
        # # This tends to produce more stable contact behavior than raw triangle meshes.
        # asset_options.vhacd_enabled = True

        # # Ignore the center-of-mass from the imported asset and recompute it
        # # from the (possibly VHACD-processed) collision geometry. This keeps
        # # COM consistent with the new convex shapes and improves balance.
        # asset_options.override_com = True

        # # Ignore the inertia tensor from the imported asset and recompute it
        # # from the processed collision geometry. This avoids pathological
        # # inertia values when the original mesh or scaling is irregular.
        # asset_options.override_inertia = True

        # # Merge rigid bodies that are connected by fixed joints into a single
        # # rigid body where possible. This reduces joint count and can remove
        # # tiny, jitter-prone segments, leading to more stable simulation.
        # asset_options.collapse_fixed_joints = True

        # # Automatically replace cylinders in the collision geometry with
        # # capsules, which generally have more robust contact behavior and
        # # fewer edge cases in PhysX than cylinders.
        # asset_options.replace_cylinder_with_capsule = True
        # # ---- 1211 actions ---------------------------------------------------------

        # multi humanoid template change ===============
        motor_efforts = None

        humanoid_assets = []
        # load beta into observation ===============
        template_betas = []   # <--- add this
        # load beta into observation ===============

        for beta_key, betas in self.all_betas.items():
            for gender in ['male', 'female']:

                gender_tensor = encode_gender(gender)
                curr_gender_beta = torch.cat([gender_tensor.view(1), betas], dim=0)

                template_betas.append(curr_gender_beta)

                af = os.path.join("mjcf", "smpl", f"{gender}_{beta_key}_smpl.xml")

                humanoid_asset = self.gym.load_asset(self.sim, self.asset_root, af, asset_options)

                # load beta into observation ===============
                actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
                curr_motor_efforts = [prop.motor_effort for prop in actuator_props]

                right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "R_Ankle")
                left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "L_Ankle")

                sensor_pose = gymapi.Transform()

                self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
                self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

                # sensor_count = self.gym.get_asset_force_sensor_count(humanoid_asset)
                # if sensors_per_env is None:
                #     sensors_per_env = sensor_count
                # elif sensor_count != sensors_per_env:
                #     raise ValueError("All humanoid assets must expose the same number of force sensors")

                curr_num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
                curr_num_dof = self.gym.get_asset_dof_count(humanoid_asset)
                curr_num_joints = self.gym.get_asset_joint_count(humanoid_asset)

                if motor_efforts is None:
                    # the smpl type are of same rigid body and joints, so only take info from the first one

                    motor_efforts = curr_motor_efforts

                    self.max_motor_effort = max(motor_efforts)
                    self.motor_efforts = to_torch(motor_efforts, device=self.device)

                    self.torso_index = 0
                    self.num_bodies = curr_num_bodies
                    self.num_dof = curr_num_dof
                    self.num_joints = curr_num_joints

                else:

                    assert curr_num_bodies == self.num_bodies, f"diff num_bodies: {curr_num_bodies}, {self.num_bodies}, {af}, {i}"
                    assert curr_num_dof == self.num_dof, f"diff num_bodies: {curr_num_dof}, {self.num_dof}"
                    assert curr_num_joints == self.num_joints, f"diff num_bodies: {curr_num_joints}, {self.num_joints}"

                    if len(curr_motor_efforts) != len(motor_efforts):
                        raise ValueError("All humanoid assets must expose the same number of actuators")
                    if not np.allclose(curr_motor_efforts, motor_efforts):
                        raise ValueError("All humanoid assets must share identical actuator effort limits")

                humanoid_assets.append(humanoid_asset)
        
        # load beta into observation ===============
        # torch.Size([64, 10])
        self._template_betas = torch.stack(template_betas, dim=0)    # [T, B]
        # load beta into observation ===============
    
        # multi humanoid template change ===============

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # load beta into observation ===============
        # allocate per-env betas for smpl
        beta_dim = self._template_betas.shape[1]
        # torch.Size([2, 10]) [number_actors, betas]
        self._betas_env = torch.zeros(self.num_envs, beta_dim, device=self.device)
        # load beta into observation ===============


        # local batch testing ===============
        if USER == "hlz":
            # change batch id to load differnt batches
            batch_id = 0

            start = batch_id * self.num_envs

            for i in range(self.num_envs):
                # create env instance
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                # multi humanoid template change ===============
                m = len(humanoid_assets)

                asset_idx = start + i

                if asset_idx >= m:
                    asset_idx = asset_idx % m

                h_asset = humanoid_assets[asset_idx]

                # load beta into observation ===============
                # assign beta for this env when smpl is used
                self._betas_env[i] = self._template_betas[asset_idx]
                # load beta into observation ===============

                self._build_env(i, env_ptr, h_asset)
                # multi humanoid template change ===============
                self.envs.append(env_ptr)
            
        # local batch testing ===============
        else:
            for i in range(self.num_envs):
                # create env instance
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                # multi humanoid template change ===============
                m = len(humanoid_assets)

                h_asset = humanoid_assets[i % m]

                # load beta into observation ===============
                # assign beta for this env when smpl is used
                template_id = i % m
                self._betas_env[i] = self._template_betas[template_id]
                # load beta into observation ===============

                self._build_env(i, env_ptr, h_asset)
                # multi humanoid template change ===============
                self.envs.append(env_ptr)

        print(f"Loaded {torch_utils.count_unique_tensors_approx_abs(self._betas_env)} unique gender beta combination")

        # collect per-actor dof limits (lower, upper already corrected for swapped bounds)
        dof_lowers_all = []
        dof_uppers_all = []

        for env, handle in zip(self.envs, self.humanoid_handles):
            dof_prop = self.gym.get_actor_dof_properties(env, handle)

            # fix swapped bounds per DOF
            lower = np.minimum(dof_prop['lower'], dof_prop['upper'])
            upper = np.maximum(dof_prop['lower'], dof_prop['upper'])

            dof_lowers_all.append(lower)
            dof_uppers_all.append(upper)

        # shape: [num_actors, num_dof]
        dof_lowers_all = to_torch(np.stack(dof_lowers_all, axis=0), device=self.device)
        dof_uppers_all = to_torch(np.stack(dof_uppers_all, axis=0), device=self.device)

        # global per-DOF limits across all actors
        self.dof_limits_lower, _ = torch.min(dof_lowers_all, dim=0)  # [num_dof]
        self.dof_limits_upper, _ = torch.max(dof_uppers_all, dim=0)  # [num_dof]

        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        col_group = env_id
        col_filter = self._get_humanoid_collision_filter()
        segmentation_id = 0

        start_pose = gymapi.Transform()
        char_h = 1.3

        start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        # this should take no effect at reset actors, we are using the motion root_pos
        humanoid_handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", col_group, col_filter, segmentation_id)

        # fetures plugin -------------
        for f in self._features: f.on_humanoid_actor_created(self, env_id, env_ptr)
        # fetures plugin -------------

        self.gym.enable_actor_dof_force_sensors(env_ptr, humanoid_handle)

        for j in range(self.num_bodies):
            self.gym.set_rigid_body_color(env_ptr, humanoid_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.54, 0.85, 0.2))

        if (self._pd_control):
            dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
            dof_prop["driveMode"] = gymapi.DOF_MODE_POS
            self.gym.set_actor_dof_properties(env_ptr, humanoid_handle, dof_prop)

        self.humanoid_handles.append(humanoid_handle)

        return

    def _build_pd_action_offset_scale(self):
        num_joints = len(self._dof_offsets) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = self._dof_offsets[j]
            dof_size = self._dof_offsets[j + 1] - self._dof_offsets[j]

            if (dof_size == 3):
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale
                
                #lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                #lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _get_humanoid_collision_filter(self):
        return 0

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        raise NotImplementedError(
            "Base Humanoid reset removed. Use HumanoidPHC."
        )

    def render(self, sync_frame_time=False):
        # if self.viewer:
        #     self._update_camera()

        # fetures plugin -------------
        for f in self._features: f.on_render(self)
        # fetures plugin -------------

        super().render(sync_frame_time)
        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif (dof_size == 1):
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            joint_pose_q = None
            assert(False), "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs



