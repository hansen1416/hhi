import torch

from isaacgym import gymtorch

from env.tasks.humanoid_hhi import HumanoidHHI


class HumanoidViewMotion(HumanoidHHI):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        """
        HumanoidViewMotion does not use physics-based joint torques. `cfg["env"]["pdControl"] = False`,

        each pre_physics_step, it writes a zero force vector to the simulator 
        `self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)`, meaning no torques are applied to the joints. 
        
        After physics advances, post_physics_step calls _motion_sync, which directly sets root pose and all joint positions from the motion library and zeroes velocities, 
        `self.gym.set_actor_root_state_tensor_indexed`, `self.gym.set_dof_state_tensor_indexed`
        """
        control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self._motion_dt = control_freq_inv * sim_params.dt

        cfg["env"]["controlFrequencyInv"] = 1
        cfg["env"]["pdControl"] = False

        self.target_marker_enabled = False
        self.follow_camera_enabled = False

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = self._motion_lib.sample_motions(self.num_envs)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)

        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = torch.zeros_like(self.actions)
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        return

    def post_physics_step(self):
        super().post_physics_step()

        # if just want to visualize the humanodid model, see if they are stable, comment out this line.
        self._motion_sync()
        return
    
    def _get_humanoid_collision_filter(self):
        return 1 # disable self collisions

    def _motion_sync(self):
        # num_motions = self._motion_lib.num_motions()
        motion_ids = self._motion_ids
        motion_times = self.progress_buf * self._motion_dt

        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times)

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = (motion_res["root_pos"],
                motion_res["root_rot"],
                motion_res["dof_pos"],
                motion_res["root_vel"],
                motion_res["root_ang_vel"],
                motion_res["dof_vel"],
                motion_res["key_pos"])
        
        root_vel = torch.zeros_like(root_vel)
        root_ang_vel = torch.zeros_like(root_ang_vel)
        dof_vel = torch.zeros_like(dof_vel)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_actors(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_reset(self):
        motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self.reset_buf[:], self._terminate_buf[:] = compute_view_motion_reset(self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt)
        return

    def _reset_env_tensors(self, env_ids):
        num_motions = self._motion_lib.num_motions()
        self._motion_ids[env_ids] = torch.remainder(self._motion_ids[env_ids] + self.num_envs, num_motions)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

@torch.jit.script
def compute_view_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    motion_times = progress_buf * dt
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), terminated)
    return reset, terminated