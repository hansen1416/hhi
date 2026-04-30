import os
from enum import Enum
import numpy as np
import torch
from easydict import EasyDict

from isaacgym import gymtorch

from env.tasks.humanoid import Humanoid, dof_to_obs
# from utils.motion_lib_smpl import MotionLibSMPL
from utils.motion_lib_humos import MotionLibHUMOS
from isaacgym.torch_utils import *
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from utils import torch_utils

class HumanoidHHI(Humanoid):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        self.follow_camera_enabled = True

        if cfg['args'].test and cfg['args'].play and not cfg['args'].headless:
            self.target_marker_enabled = True
        else:
            self.target_marker_enabled = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        self.reward_specs = {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1, "w_pos": 0.5, "w_rot": 0.3, "w_vel": 0.1, "w_ang_vel": 0.1}

        self.power_reward = True
        # self.reward_raw is [num_envs, [r_body_pos, r_body_rot, r_vel, r_ang_vel, power_reward]], and its for debug purpose
        self.reward_raw = torch.zeros((self.num_envs, 5 if self.power_reward else 4)).to(self.device)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.00005)

        # load_motion must happen here
        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        # ---- target motion observation ----
        # spawn anchor for each env (keeps envs separated in world)
        self._spawn_root_pos = self._humanoid_root_states[:, 0:3].clone()

        # PHC-style reference bookkeeping
        self._sampled_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._motion_start_times = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # self._global_offset = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        # ---- target motion observation ----

        # jiter fix 0423 =======================
        if self.smoothness_reward:
            self.smooth_action_coef = cfg["env"].get("smoothActionCoef", 0.005)
            self._prev_actions_raw = torch.zeros(
                (self.num_envs, self.get_action_size()),
                device=self.device, dtype=torch.float
            )
        # jiter fix 0423 =======================

        self.reward_vesion = 1

        # reward v1 -----------
        self.reward_specs_v1 = {
            # tracking sharpness
            "k_root_pos": 10.0,
            "k_root_rot": 10.0,
            "k_pose": 30.0,
            "k_com_pos": 20.0,
            "k_com_vel": 0.5,
            "k_ee": 40.0,
            "k_vel": 0.05,

            # positive reward weights, max roughly 1.0
            "w_root": 0.25,
            "w_pose": 0.30,
            "w_com": 0.10,
            "w_ee": 0.25,
            "w_vel": 0.10,

            # negative smoothness/contact penalties
            "lambda_action_smooth": 0.01,
            "lambda_dof_vel_delta": 0.005,
            "lambda_foot_slide": 0.05,

            # contact/slide threshold
            "foot_height_threshold": 0.08,
        }

        # End-effectors used for stronger coarse body tracking.
        ee_names = cfg["env"].get(
            "overallPoseEndEffectors",
            ["L_Wrist", "R_Wrist", "L_Ankle", "R_Ankle", "Head"]
        )
        ee_names = [n for n in ee_names if n in self._body_names]
        self._overall_pose_ee_ids = self._build_key_body_ids_tensor(ee_names)

        # State buffers for temporal smoothness.
        self.actions = torch.zeros(
            (self.num_envs, self.get_action_size()),
            device=self.device,
            dtype=torch.float
        )
        self._prev_actions = torch.zeros_like(self.actions)
        self._prev_dof_vel = self._dof_vel.clone()
        # reward v1 -----------

        return

    def post_physics_step(self):
        # computes obs_buf inside the base class
        # super().post_physics_step()

        self.progress_buf += 1

        self._refresh_sim_tensors()

        motion_res, motion_res_next, task_obs = self._compute_task_obs_v7()

        # offset = self._global_offset.unsqueeze(1)# [N,1,3]

        # ref_body_pos = motion_res["rg_pos"] + offset

        ref_body_pos = motion_res["rg_pos"]
        ref_body_rot = motion_res["rb_rot"]
        ref_body_vel = motion_res["body_vel"]
        ref_body_ang_vel = motion_res["body_ang_vel"]

        # we removed all the gloabl_offset logic, so maybe no need to offset here either
        ref_key_pos_next = motion_res_next["key_pos"]

        self._compute_observations(task_obs=task_obs)
        self._compute_reward(ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel)

        # if self.cfg["env"].get("debug_reward", True):
        #     local_env = 3
        #     print(self.progress_buf[local_env], self.rew_buf[local_env], self.rew_buf[local_env])

        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())

        # each row of `amp_obs_flat`, `amp_shape` corresponds to one env,
        # disc-shape-condition
        """
        fake_amp_obs   = batch_dict['amp_obs'][idx]
        fake_amp_shape = batch_dict['amp_shape'][idx]

        real_amp_obs = _fetch_amp_obs_demo(fake_amp_shape)

        then use fake_amp_obs[i] with real_amp_obs[i], both conditioned on the same fake_amp_shape[i]
        """
        self.extras["amp_obs"] = amp_obs_flat # [num_envs, get_num_amp_obs()]
        self.extras["amp_shape"] = self._betas_env# [num_envs, 11]

        # fetures plugin -------------
        for f in self._features: f.on_post_physics_step(self, ref_key_pos_next)
        # fetures plugin -------------

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step

    # def fetch_amp_obs_demo(self, num_samples):
    def fetch_amp_obs_demo(self):
        """
        num_sample is the same as HHIAgent._amp_batch_size, 
        controled by cfg_train['params']['config']['amp_batch_size']

        disc-shape-condition
        """
  
        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(self.num_envs)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == self.num_envs)
        # typical self._amp_obs_demo_buf shape, [1, 10, 292]
        # 1 = num_samples; 10 = self._num_amp_obs_steps; 292 = self._num_amp_obs_per_step

        gender_beta_key = []

        for i in range(self.num_envs):
            gender_beta_key.append(self.env_id_beta_keys_map[i])
        
        # [num_envs,]
        motion_ids = self._motion_lib.sample_motions(self.num_envs, gender_beta_key)
        
        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        # [num_envs x 10, 292]
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times0)

        # [num_envs, 10, 292]
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        # [num_envs, 2920]
        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())

        amp_shape_demo = self._motion_lib.get_motion_shape(motion_ids)

        return amp_obs_demo_flat, amp_shape_demo

    def build_amp_obs_demo(self, motion_ids, motion_times0):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps, device=self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = (motion_res["root_pos"],
                motion_res["root_rot"],
                motion_res["dof_pos"],
                motion_res["root_vel"],
                motion_res["root_ang_vel"],
                motion_res["dof_vel"],
                motion_res["key_pos"])

        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        # multi humanoid template change ===============
        num_key_bodies = len(key_bodies)

        # asset_file.startswith("mjcf/smpl_"), it's a list of asset files
        # some conditions borrowed from PHC
        # Use AMP observation version 1 (basic key-body positions only)
        self.amp_obs_v = 1
        # Include root height in AMP observations
        self._amp_root_height_obs = True
        # Use a subset of degrees of freedom instead of all joints
        self._has_dof_subset = False
        # Do not include discrete shape parameters
        self._has_shape_obs_disc = False
        # Do not include discrete limb length/weight features
        self._has_limb_weight_obs_disc = False

        remove_names = ["L_Hand", "R_Hand", "L_Toe", "R_Toe"]
        disc_idxes = []

        for idx, name in enumerate(self._dof_names):
            if not name in remove_names:
                disc_idxes.append(np.arange(idx * 3, (idx + 1) * 3))
        
        if len(disc_idxes) > 0:
            self.dof_subset = torch.from_numpy(np.concatenate(disc_idxes)) 
        else: 
            torch.tensor([]).long()

        if self.amp_obs_v == 1:
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + len(self._dof_names) * 3 + 3 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + len(self._dof_names) * 3 + 6 * num_key_bodies  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, key_body_vel]

        if not self._amp_root_height_obs:
            self._num_amp_obs_per_step -= 1

        if self._has_dof_subset:
            self._num_amp_obs_per_step -= (6 + 3) * int((len(self._dof_names) * 3 - len(self.dof_subset)) / 3)

        if self._has_shape_obs_disc:
            # todo-disc-shape-condition, change this flag and pass gender,beta to amp_obs
            self._num_amp_obs_per_step += 11
        
        if self._has_limb_weight_obs_disc:
            self._num_amp_obs_per_step += 10

        # 196
        # print(self._num_amp_obs_per_step)

        # ---- target motion observation ----
        self._enable_task_obs = True
        self._task_obs_v = 7
        self._num_task_obs = 9 * len(key_bodies)   # [Δp_local, Δv_local, p*_rel_local]
        self._num_obs += self._num_task_obs

        # here self._num_obs == 585. 
        # 1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3 + 9 * len(key_bodies)
        # ---- target motion observation ----

        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_dof)

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
            "device": self.device,
            "min_length": -1, 
            "step_dt": 1/60,
            "key_body_ids": self._key_body_ids
        })

        self._motion_lib = MotionLibHUMOS(motion_lib_cfg=motion_lib_cfg, loaded_gender_beta_key=self.loaded_gender_beta_key)

        self._motion_lib.load_motions()

        return
    
    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None, task_obs=None):
        # [num_env, 358]
        obs = self._compute_humanoid_obs(env_ids)

        # load beta into observation ===============
        # append shape betas for smpl assets
        if env_ids is None:
            betas = self._betas_env                        # [num_envs, B]
        else:
            betas = self._betas_env[env_ids]               # [len(env_ids), B]

        # simple normalisation to keep magnitudes modest
        # betas is [num_env, 11].
        betas_norm = betas.clone()
        betas_norm[:, 1:] /= 3.0

        # torch.Size([num_envs, 358]) -> torch.Size([num_envs, 368]), 10 betas
        # load beta into observation ===============

        # ---- target motion observation ----
        if task_obs is not None:
            # [num_env, 574]
            obs = torch.cat([obs, task_obs], dim=-1)
        # ---- target motion observation ----

        # [num_env, 585]
        obs = torch.cat([obs, betas], dim=-1)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
        
        obs = compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs)
        return obs

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_heights)
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # super()._reset_envs(env_ids)

        if (len(env_ids) > 0):
            # 
            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self._terminate_buf[env_ids] = 0

            num_envs = env_ids.shape[0]

            # get the gender,beta by env id
            gender_beta_key = []

            for eid in env_ids:
                gender_beta_key.append(self.env_id_beta_keys_map[eid.item()])

            self._sampled_motion_ids[env_ids] = self._motion_lib.sample_motions(num_envs, gender_beta_key)

            truncate_time = self.dt * (self._num_amp_obs_steps - 1)


            # force full-motion playback when in test/play/non-headless mode
            if self.target_marker_enabled:
                # HumanoidViewMotion / visualization mode:
                # always start at t=0 so every reset plays the COMPLETE motion
                # (works perfectly with your 128 HUMOS variations + AMASS clips)
                motion_times = torch.zeros(num_envs, dtype=torch.float, device=self.device)
            else:
                # Original training behavior (random start with AMP history buffer)
                truncate_time = self.dt * (self._num_amp_obs_steps - 1)
                motion_times = self._motion_lib.sample_time(self._sampled_motion_ids[env_ids], truncate_time=truncate_time)
                motion_times = motion_times + truncate_time

            self._reset_ref_env_ids = env_ids
            self._reset_ref_motion_ids = self._sampled_motion_ids[env_ids]
            self._reset_ref_motion_times = motion_times

            self._motion_start_times[env_ids] = motion_times

            motion_res, motion_res_next, task_obs = self._compute_task_obs_v7(env_ids=env_ids)

            target_root_pos     = motion_res["root_pos"]
            target_root_rot     = motion_res["root_rot"]
            target_dof_pos      = motion_res["dof_pos"]
            target_root_vel     = motion_res["root_vel"]
            target_root_ang_vel = motion_res["root_ang_vel"]
            target_dof_vel      = motion_res["dof_vel"]
            target_key_pos      = motion_res["key_pos"]

            self._reset_actors(env_ids, target_root_pos, target_root_rot, target_dof_pos, target_root_vel, target_root_ang_vel, target_dof_vel)
            self._reset_env_tensors(env_ids)
            self._refresh_sim_tensors()

            # reward v1 -----------
            self._prev_actions[env_ids] = 0.0
            self._prev_dof_vel[env_ids] = self._dof_vel[env_ids]
            # reward v1 -----------

            self._compute_observations(env_ids=env_ids, task_obs=task_obs)

            # jiter fix 0423 =======================
            if self.action_filtering:
                self._filtered_actions[env_ids] = 0.0

            if self.smoothness_reward:
                self._prev_actions_raw[env_ids] = 0.0
            # jiter fix 0423 =======================

            # fetures plugin -------------
            for f in self._features: f.on_reset_envs(self, env_ids, target_key_pos)
            # fetures plugin -------------

        self._init_amp_obs(env_ids)

        return

    def _reset_actors(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):

        # set env actor state
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        # self._reset_ref_env_ids = env_ids
        # self._reset_ref_motion_ids = motion_ids
        # self._reset_ref_motion_times = motion_times


        # ---- target motion observation ----
        # self._sampled_motion_ids[env_ids] = motion_ids
        # self._motion_start_times[env_ids] = motion_times
        # self._global_offset[env_ids] = global_offset
        # ---- target motion observation ----
        return

    def _reset_env_tensors(self, env_ids):

        # # root: overwrite vel
        # self._humanoid_root_states[env_ids, 7:10]  = 0.0  # or root_vel
        # self._humanoid_root_states[env_ids, 10:13] = 0.0  # or root_ang_vel

        # # dof: overwrite vel
        # self._dof_vel[env_ids, :] = 0.0  # or dof_vel

        # print(self._humanoid_root_states[env_ids, 7:10])
        # print(self._humanoid_root_states[env_ids, 10:13])
        # print("=================================")

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        dof_pos = self._dof_state[..., :, 0]
        dof_pos = dof_pos.contiguous()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                      gymtorch.unwrap_tensor(dof_pos),
                                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = (motion_res["root_pos"],
                motion_res["root_rot"],
                motion_res["dof_pos"],
                motion_res["root_vel"],
                motion_res["root_ang_vel"],
                motion_res["dof_vel"],
                motion_res["key_pos"])

        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._dof_offsets)
        return

    # ---- target motion observation ----
    def _compute_task_obs_v7(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        root_pos = self._rigid_body_pos[env_ids, 0, :]
        root_rot = self._rigid_body_rot[env_ids, 0, :]

        body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
        body_vel = self._rigid_body_vel[env_ids][:, self._key_body_ids, :]

        motion_ids = self._sampled_motion_ids[env_ids]
        t = self.progress_buf[env_ids].float() * self.dt + \
            self._motion_start_times[env_ids]

        # reference key positions (and finite-diff key velocities)
        motion_res      = self._motion_lib.get_motion_state(motion_ids, t)
        motion_res_next = self._motion_lib.get_motion_state(motion_ids, t + self.dt)

        # # anchor into env
        # offset = self._global_offset[env_ids].unsqueeze(1)
        # # motion_res["key_pos"].shape is [num_envs, the number of key bodies, 3]
        # ref_pos = motion_res["key_pos"] + offset

        ref_pos = motion_res["key_pos"]
        ref_vel = ( motion_res_next["key_pos"] - motion_res["key_pos"]) / self.dt

        task_obs = compute_task_obs_v7_1step(root_pos, root_rot, body_pos, body_vel, ref_pos, ref_vel)

        return motion_res, motion_res_next, task_obs
    # ---- target motion observation ----

    def _compute_reward(self, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel

        if self.reward_vesion == 1:
            self.rew_buf[:], self.reward_raw = compute_imitation_reward_v1(
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                ref_body_pos,
                ref_body_rot,
                ref_body_vel,
                ref_body_ang_vel,
                self.actions,
                self._prev_actions,
                self._dof_vel,
                self._prev_dof_vel,
                self._key_body_ids,
                self._overall_pose_ee_ids,
                self._contact_body_ids,
                self.progress_buf,
                self.reward_specs_v1,
            )
        else:
            self.rew_buf[:], self.reward_raw = compute_imitation_reward(body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, self.reward_specs)

        if self.power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1) 
            # power_reward = -0.00005 * (power ** 2)
            power_reward = -self.power_coefficient * power
            power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

            self.rew_buf[:] += power_reward
            self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)

        # jiter fix 0423 =======================
        if self.smoothness_reward:
            # smoothness penalty on raw policy action change
            action_diff = self.actions_raw - self._prev_actions_raw
            smooth_reward = -self.smooth_action_coef * torch.sum(action_diff * action_diff, dim=-1)
            smooth_reward[self.progress_buf <= 1] = 0

            self.rew_buf[:] += smooth_reward
            self.reward_raw = torch.cat([self.reward_raw, smooth_reward[:, None]], dim=-1)

            self._prev_actions_raw[:] = self.actions_raw
        # jiter fix 0423 =======================

        # reward v1 -----------
        # Update temporal buffers after reward computation.
        self._prev_actions[:] = self.actions
        self._prev_dof_vel[:] = self._dof_vel
        # reward v1 -----------

        return
    
    def get_env_state(self):
        """Called automatically during checkpoint save."""
        return {
            "root_states": self._root_states.clone(),
            "dof_state": self._dof_state.clone(),

            # Motion bookkeeping — this is what was missing
            "sampled_motion_ids": self._sampled_motion_ids.clone(),
            "motion_start_times": self._motion_start_times.clone(),
            # "global_offset": self._global_offset.clone(),

            # Episode control
            "progress_buf": self.progress_buf.clone(),
            "reset_buf": self.reset_buf.clone(),
            "terminate_buf": self._terminate_buf.clone(),
        }

    def set_env_state(self, state):
        """Called automatically on resume."""
        if state is None:
            return

        # Physics
        if "root_states" in state:
            self._root_states[:] = state["root_states"]
        if "dof_state" in state:
            self._dof_state[:] = state["dof_state"]

        # Motion bookkeeping (safe for old checkpoints)
        self._sampled_motion_ids[:] = state.get("sampled_motion_ids", self._sampled_motion_ids)
        self._motion_start_times[:] = state.get("motion_start_times", self._motion_start_times)
        # self._global_offset[:] = state.get("global_offset", self._global_offset)

        self.progress_buf[:] = state.get("progress_buf", self.progress_buf)
        self.reset_buf[:] = state.get("reset_buf", 0)
        self._terminate_buf[:] = state.get("terminate_buf", 0)

        # Recompute everything so observations / AMP / task_obs are consistent
        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_amp_observations()
        self._init_amp_obs(torch.arange(self.num_envs, device=self.device))

        print(f"✅ Restored full motion state | progress_buf mean = {self.progress_buf.float().mean():.1f}")

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

# ---- target motion observation ----
@torch.jit.script
def compute_task_obs_v7_1step(root_pos, root_rot, body_pos, body_vel, ref_pos, ref_vel):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # root_pos: (B,3), root_rot:(B,4)
    # body_pos/body_vel/ref_pos/ref_vel: (B,J,3)
    B, J, _ = body_pos.shape

    heading_inv = torch_utils.calc_heading_quat_inv(root_rot)         # (B,4)
    heading_inv = heading_inv.unsqueeze(1).repeat(1, J, 1)            # (B,J,4)

    dp = ref_pos - body_pos
    dv = ref_vel - body_vel
    ref_rel = ref_pos - root_pos.unsqueeze(1)

    dp_l = quat_rotate(heading_inv.reshape(-1,4), dp.reshape(-1,3)).view(B, J * 3)
    dv_l = quat_rotate(heading_inv.reshape(-1,4), dv.reshape(-1,3)).view(B, J * 3)
    rr_l = quat_rotate(heading_inv.reshape(-1,4), ref_rel.reshape(-1,3)).view(B, J * 3)

    return torch.cat([dp_l, dv_l, rr_l], dim=-1)   # (B, 9*J)
# ---- target motion observation ----


@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos,
                                  local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)

    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


@torch.jit.script
def compute_imitation_reward(body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, rwd_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)
    # import ipdb
    # ipdb.set_trace()
    return reward, reward_raw

# reward v1 -----------
@torch.jit.script
def compute_imitation_reward_v1(
    body_pos,
    body_rot,
    body_vel,
    body_ang_vel,
    ref_body_pos,
    ref_body_rot,
    ref_body_vel,
    ref_body_ang_vel,
    actions,
    prev_actions,
    dof_vel,
    prev_dof_vel,
    key_body_ids,
    ee_body_ids,
    contact_body_ids,
    progress_buf,
    rwd_specs,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]

    k_root_pos = rwd_specs["k_root_pos"]
    k_root_rot = rwd_specs["k_root_rot"]
    k_pose = rwd_specs["k_pose"]
    k_com_pos = rwd_specs["k_com_pos"]
    k_com_vel = rwd_specs["k_com_vel"]
    k_ee = rwd_specs["k_ee"]
    k_vel = rwd_specs["k_vel"]

    w_root = rwd_specs["w_root"]
    w_pose = rwd_specs["w_pose"]
    w_com = rwd_specs["w_com"]
    w_ee = rwd_specs["w_ee"]
    w_vel = rwd_specs["w_vel"]

    lambda_action_smooth = rwd_specs["lambda_action_smooth"]
    lambda_dof_vel_delta = rwd_specs["lambda_dof_vel_delta"]
    lambda_foot_slide = rwd_specs["lambda_foot_slide"]
    foot_height_threshold = rwd_specs["foot_height_threshold"]

    B = body_pos.shape[0]

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    root_vel = body_vel[:, 0, :]

    ref_root_pos = ref_body_pos[:, 0, :]
    ref_root_rot = ref_body_rot[:, 0, :]
    ref_root_vel = ref_body_vel[:, 0, :]

    heading_inv = torch_utils.calc_heading_quat_inv(root_rot)
    ref_heading_inv = torch_utils.calc_heading_quat_inv(ref_root_rot)

    # ------------------------------------------------------------
    # 1. Root trajectory/orientation reward
    # ------------------------------------------------------------
    root_pos_err = ((ref_root_pos - root_pos) ** 2).mean(dim=-1)

    diff_root_rot = torch_utils.quat_mul(ref_root_rot, torch_utils.quat_conjugate(root_rot))
    diff_root_angle = torch_utils.quat_to_angle_axis(diff_root_rot)[0]
    root_rot_err = diff_root_angle ** 2

    r_root = torch.exp(-k_root_pos * root_pos_err - k_root_rot * root_rot_err)

    # ------------------------------------------------------------
    # 2. Root-relative key-body pose reward
    #    This is the main "overall body pose" term.
    # ------------------------------------------------------------
    key_pos = body_pos[:, key_body_ids, :]
    ref_key_pos = ref_body_pos[:, key_body_ids, :]

    J = key_pos.shape[1]

    key_rel = key_pos - root_pos.unsqueeze(1)
    ref_key_rel = ref_key_pos - ref_root_pos.unsqueeze(1)

    heading_inv_expand = heading_inv.unsqueeze(1).repeat(1, J, 1)
    ref_heading_inv_expand = ref_heading_inv.unsqueeze(1).repeat(1, J, 1)

    local_key_rel = quat_rotate(
        heading_inv_expand.reshape(B * J, 4),
        key_rel.reshape(B * J, 3)
    ).view(B, J, 3)

    local_ref_key_rel = quat_rotate(
        ref_heading_inv_expand.reshape(B * J, 4),
        ref_key_rel.reshape(B * J, 3)
    ).view(B, J, 3)

    pose_err = ((local_ref_key_rel - local_key_rel) ** 2).mean(dim=-1).mean(dim=-1)
    r_pose = torch.exp(-k_pose * pose_err)

    # ------------------------------------------------------------
    # 3. Proxy COM reward
    #    This uses mean rigid-body position/velocity as a lightweight COM proxy.
    # ------------------------------------------------------------
    com_pos = body_pos.mean(dim=1)
    ref_com_pos = ref_body_pos.mean(dim=1)

    com_vel = body_vel.mean(dim=1)
    ref_com_vel = ref_body_vel.mean(dim=1)

    local_com_rel = quat_rotate(heading_inv, com_pos - root_pos)
    local_ref_com_rel = quat_rotate(ref_heading_inv, ref_com_pos - ref_root_pos)

    local_com_vel = quat_rotate(heading_inv, com_vel)
    local_ref_com_vel = quat_rotate(ref_heading_inv, ref_com_vel)

    com_pos_err = ((local_ref_com_rel - local_com_rel) ** 2).mean(dim=-1)
    com_vel_err = ((local_ref_com_vel - local_com_vel) ** 2).mean(dim=-1)

    r_com = torch.exp(-k_com_pos * com_pos_err - k_com_vel * com_vel_err)

    # ------------------------------------------------------------
    # 4. End-effector reward: hands, feet, head
    # ------------------------------------------------------------
    if ee_body_ids.numel() > 0:
        ee_pos = body_pos[:, ee_body_ids, :]
        ref_ee_pos = ref_body_pos[:, ee_body_ids, :]

        E = ee_pos.shape[1]

        ee_rel = ee_pos - root_pos.unsqueeze(1)
        ref_ee_rel = ref_ee_pos - ref_root_pos.unsqueeze(1)

        heading_ee = heading_inv.unsqueeze(1).repeat(1, E, 1)
        ref_heading_ee = ref_heading_inv.unsqueeze(1).repeat(1, E, 1)

        local_ee = quat_rotate(
            heading_ee.reshape(B * E, 4),
            ee_rel.reshape(B * E, 3)
        ).view(B, E, 3)

        local_ref_ee = quat_rotate(
            ref_heading_ee.reshape(B * E, 4),
            ref_ee_rel.reshape(B * E, 3)
        ).view(B, E, 3)

        ee_err = ((local_ref_ee - local_ee) ** 2).mean(dim=-1).mean(dim=-1)
        r_ee = torch.exp(-k_ee * ee_err)
    else:
        r_ee = torch.ones_like(r_root)

    # ------------------------------------------------------------
    # 5. Coarse key-body velocity reward
    # ------------------------------------------------------------
    key_vel = body_vel[:, key_body_ids, :]
    ref_key_vel = ref_body_vel[:, key_body_ids, :]

    local_key_vel = quat_rotate(
        heading_inv_expand.reshape(B * J, 4),
        key_vel.reshape(B * J, 3)
    ).view(B, J, 3)

    local_ref_key_vel = quat_rotate(
        ref_heading_inv_expand.reshape(B * J, 4),
        ref_key_vel.reshape(B * J, 3)
    ).view(B, J, 3)

    vel_err = ((local_ref_key_vel - local_key_vel) ** 2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * vel_err)

    # ------------------------------------------------------------
    # 6. Smoothness penalties
    # ------------------------------------------------------------
    action_delta = ((actions - prev_actions) ** 2).mean(dim=-1)
    dof_vel_delta = ((dof_vel - prev_dof_vel) ** 2).mean(dim=-1)

    p_action_smooth = -lambda_action_smooth * action_delta
    p_dof_vel_delta = -lambda_dof_vel_delta * dof_vel_delta

    p_action_smooth = torch.where(
        progress_buf <= 3,
        torch.zeros_like(p_action_smooth),
        p_action_smooth
    )

    p_dof_vel_delta = torch.where(
        progress_buf <= 3,
        torch.zeros_like(p_dof_vel_delta),
        p_dof_vel_delta
    )

    # ------------------------------------------------------------
    # 7. Foot sliding penalty
    # ------------------------------------------------------------
    if contact_body_ids.numel() > 0:
        foot_pos = body_pos[:, contact_body_ids, :]
        foot_vel = body_vel[:, contact_body_ids, :]

        foot_height = foot_pos[:, :, 2]
        foot_speed_xy = (foot_vel[:, :, 0:2] ** 2).sum(dim=-1)

        near_ground = foot_height < foot_height_threshold
        foot_slide = (near_ground.float() * foot_speed_xy).mean(dim=-1)

        p_foot_slide = -lambda_foot_slide * foot_slide
        p_foot_slide = torch.where(
            progress_buf <= 3,
            torch.zeros_like(p_foot_slide),
            p_foot_slide
        )
    else:
        p_foot_slide = torch.zeros_like(r_root)

    # ------------------------------------------------------------
    # Final task reward
    # ------------------------------------------------------------
    reward = (
        w_root * r_root
        + w_pose * r_pose
        + w_com * r_com
        + w_ee * r_ee
        + w_vel * r_vel
        + p_action_smooth
        + p_dof_vel_delta
        + p_foot_slide
    )

    reward_raw = torch.stack(
        [
            r_root,
            r_pose,
            r_com,
            r_ee,
            r_vel,
            p_action_smooth,
            p_dof_vel_delta,
            p_foot_slide,
        ],
        dim=-1,
    )

    return reward, reward_raw
# reward v1 -----------