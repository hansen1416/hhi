import os
import torch
from isaacgym import gymtorch
from env.tasks.humanoid_hhi import HumanoidHHI


class HumanoidReplayActions(HumanoidHHI):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        """
        Plays back a SINGLE recorded PD action sequence saved by HumanoidRecordActions.
        
        • Only one motion/action file (you pass exactly one .pt via cfg["env"]["action_file"])
        • Starts from the exact initial state saved in the recording
        • Stops completely when the recorded sequence ends (no reset/loop)
        • Fully physics-based PD control (no neural net, no kinematic sync)
        • Fixes the Gym tensor shape error you saw ("expected (25, 13), received (1, 13)")
        • Compatible with your HUMOS 128 variations (single motion + shape + gender)
        
        This is the exact companion to your recording class.
        """
        cfg = dict(cfg)  # don't mutate original
        cfg["env"]["pdControl"] = True
        cfg["env"]["controlFrequencyInv"] = cfg["env"].get("controlFrequencyInv", 1)

        # Enforce single-env playback
        if cfg["env"].get("numEnvs", 1) != 1:
            print("WARNING: HumanoidPlayRecordedActions forces numEnvs=1")
            cfg["env"]["numEnvs"] = 1

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        
        record_dir = os.path.join("/home/hlz/datasets/recorded_policy_actions_hhi/")

        self.action_file = os.path.join(record_dir, "motion_000000_recorded.pt")
        
        assert os.path.exists(self.action_file), f"Recorded file not found: {self.action_file}"

        data = torch.load(self.action_file, map_location=self.device)
        self.recorded_actions = data["actions"].to(self.device)          # [T, 207]
        self.num_playback_steps = int(data.get("num_steps", self.recorded_actions.shape[0]))
        self.current_playback_step = 0

        # Exact initial state from the recording
        self.initial_root_pos = data["initial_root_pos"].to(self.device).unsqueeze(0)
        self.initial_root_rot = data["initial_root_rot"].to(self.device).unsqueeze(0)
        self.initial_dof_pos = data["initial_dof_pos"].to(self.device).unsqueeze(0)

        print(f"✅ PLAYBACK MODE (single HUMOS motion)")
        print(f"   Loaded {self.num_playback_steps} PD actions from: {self.action_file}")
        print(f"   Will STOP (no reset) when sequence ends — final pose held by last PD target.")

        # Apply exact initial state using the correct Isaac Gym pattern
        self._apply_recorded_initial_state()

    def _apply_recorded_initial_state(self):
        """Use the exact same pattern as HumanoidViewMotion + _reset_actors to avoid shape error."""
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # Update internal buffers (this is what _reset_actors does internally)
        self._reset_actors(env_ids=env_ids,
                           root_pos=self.initial_root_pos,
                           root_rot=self.initial_root_rot,
                           dof_pos=self.initial_dof_pos,
                           root_vel=torch.zeros_like(self.initial_root_pos),
                           root_ang_vel=torch.zeros((self.num_envs, 3), device=self.device),
                           dof_vel=torch.zeros_like(self.initial_dof_pos))

        # Apply to simulator — CRITICAL: use self._root_states (full tensor), not _humanoid_root_states
        env_ids_int32 = self._humanoid_actor_ids[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def pre_physics_step(self, actions):
        """Ignore RL policy — feed the recorded PD targets instead."""
        if self.current_playback_step < self.num_playback_steps:
            playback_action = self.recorded_actions[self.current_playback_step].unsqueeze(0)  # [1, action_dim]
            super().pre_physics_step(playback_action)
            self.current_playback_step += 1
        else:
            # Sequence ended → hold last action (zero torques) so the humanoid freezes
            super().pre_physics_step(torch.zeros_like(self.actions))

    def post_physics_step(self):
        super().post_physics_step()

        # When recorded sequence ends, STOP (no reset/loop)
        if self.current_playback_step >= self.num_playback_steps:
            self.reset_buf[:] = 0          # block any further reset
            self._terminate_buf[:] = 1     # signal termination to viewer/training loop

            # Optional: zero velocities on final frame so it really "freezes"
            self._dof_vel[:] = 0.0
            self._humanoid_root_states[:, 7:13] = 0.0

        return

    def _compute_reset(self):
        """Override to never auto-reset during playback (we stop manually)."""
        # We do NOT call the motion-length check here — the recorded sequence controls termination
        self.reset_buf[:] = 0
        self._terminate_buf[:] = 0
        return

    def _reset_env_tensors(self, env_ids):
        """Only allow manual reset (e.g. you pressing R in the viewer)."""
        self.current_playback_step = 0
        if hasattr(self, 'initial_root_pos'):
            self._apply_recorded_initial_state()
        super()._reset_env_tensors(env_ids)

# import os
# import torch
# from isaacgym import gymtorch
# from env.tasks.humanoid_hhi import HumanoidHHI


# class HumanoidPlayRecordedActions(HumanoidHHI):
#     def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
#         """
#         HumanoidPlayRecordedActions replays a single sequence of recorded PD actions
#         (saved by HumanoidRecordActions) without any RL policy/model.
        
#         - Only one motion/action sequence (you pass a single .pt file via cfg["env"]["action_file"]).
#         - The motion file passed to the env (via cfg["env"]["motion_file"]) should contain only
#           the corresponding HUMOS motion (you already ensured this).
#         - Stops (no reset/cycling) when the recorded sequence reaches its end.
#         - Keeps PD control active so the physics sim behaves exactly as during recording.
#         - Designed for num_envs=1 (single playback).
        
#         This is the "play them without any model" counterpart to HumanoidRecordActions
#         and follows the same pattern as HumanoidViewMotion (kinematic sync) but uses
#         physics-based PD targets for faithful replay of the policy-driven motion.
#         """
#         # Force PD control mode (the recorded actions are PD targets)
#         cfg = dict(cfg)  # don't mutate original
#         cfg["env"]["pdControl"] = True
#         cfg["env"]["controlFrequencyInv"] = cfg["env"].get("controlFrequencyInv", 1)

#         # Enforce single-env playback for one motion
#         if cfg["env"].get("numEnvs", 1) != 1:
#             print("WARNING: HumanoidPlayRecordedActions is designed for numEnvs=1 (single motion playback). Forcing numEnvs=1.")
#             cfg["env"]["numEnvs"] = 1

#         super().__init__(cfg=cfg,
#                          sim_params=sim_params,
#                          physics_engine=physics_engine,
#                          device_type=device_type,
#                          device_id=device_id,
#                          headless=headless)

#         self.playback_enabled = True

#         record_dir = os.path.join("/home/hlz/datasets/recorded_policy_actions_hhi/")

#         self.action_file = os.path.join(record_dir, "motion_000000_recorded.pt")

#         assert os.path.exists(self.action_file), f"Recorded action file not found: {self.action_file}"

#         data = torch.load(self.action_file, map_location=self.device)
#         self.recorded_actions = data["actions"].to(self.device)          # [T, num_actions] (207 for HUMOS/SMPL)
#         self.num_playback_steps = int(data.get("num_steps", self.recorded_actions.shape[0]))
#         self.current_playback_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

#         # Optional: restore exact initial state from recording (highly recommended for perfect start)
#         self.initial_root_pos = data.get("initial_root_pos")
#         self.initial_root_rot = data.get("initial_root_rot")
#         self.initial_dof_pos = data.get("initial_dof_pos")

#         # If the recording was done with HUMOS, we can optionally set the motion_id for consistency
#         # (base class still loads the motion file, but we don't use it for control)
#         if "motion_id" in data:
#             self._sampled_motion_ids[:] = data["motion_id"]
#             print(f"Playback: set motion_id = {int(data['motion_id'])} (HUMOS motion + shape + gender)")

#         print(f"PLAYBACK MODE ENABLED (HUMOS 128 variations supported)")
#         print(f"   Loaded {self.num_playback_steps} policy actions from: {self.action_file}")
#         print(f"   Will STOP (no reset/loop) when sequence ends - final pose held by last PD target.")

#         # Apply recorded initial state once at startup
#         if self.initial_root_pos is not None:
#             self._apply_recorded_initial_state()

#     def _apply_recorded_initial_state(self):
#         """Set root pose + DOF positions from the recording (exact start of the saved episode)."""
#         env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
#         self._humanoid_root_states[env_ids, 0:3] = self.initial_root_pos
#         self._humanoid_root_states[env_ids, 3:7] = self.initial_root_rot
#         self._dof_pos[env_ids] = self.initial_dof_pos

#         # Zero velocities so physics starts cleanly
#         self._humanoid_root_states[env_ids, 7:13] = 0.0
#         self._dof_vel[env_ids] = 0.0

#         env_ids_int32 = self._humanoid_actor_ids[env_ids].to(torch.int32)
#         self.gym.set_actor_root_state_tensor_indexed(self.sim,
#                                                      gymtorch.unwrap_tensor(self._humanoid_root_states),
#                                                      gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
#         self.gym.set_dof_state_tensor_indexed(self.sim,
#                                               gymtorch.unwrap_tensor(self._dof_state),
#                                               gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

#     def pre_physics_step(self, actions):
#         """Ignore any external policy actions and feed the recorded PD targets instead."""
#         if self.playback_enabled:
#             # Clamp so we never go beyond the recorded length
#             step_idx = self.current_playback_step.clamp(max=self.num_playback_steps - 1)
#             playback_action = self.recorded_actions[step_idx]                     # [num_envs, action_dim] (since num_envs=1)
#             playback_action = playback_action.unsqueeze(0) if playback_action.ndim == 1 else playback_action

#             # Let base class handle PD target conversion + tensor write
#             super().pre_physics_step(playback_action)

#             # Advance only while still inside the sequence
#             self.current_playback_step += 1
#         else:
#             super().pre_physics_step(actions)

#     def post_physics_step(self):
#         super().post_physics_step()

#         # When the recorded sequence ends, STOP (no more resets)
#         finished = self.current_playback_step >= self.num_playback_steps
#         if finished.any():
#             self.reset_buf[finished] = 0          # prevent _reset_env_tensors from firing again
#             self._terminate_buf[finished] = 1     # signal termination (viewer / training loop respects this)

#             # Optional: zero velocities on the final frame to help the humanoid "freeze" at the end pose
#             self._dof_vel[finished] = 0.0
#             self._humanoid_root_states[finished, 7:13] = 0.0   # root linear + angular velocity

#         return

#     def _reset_env_tensors(self, env_ids):
#         """Only called on the very first frame (or if you manually trigger reset)."""
#         # Reset playback counter so a manual reset would replay from the start
#         self.current_playback_step[env_ids] = 0

#         # Re-apply recorded initial state if we ever reset
#         if self.initial_root_pos is not None:
#             self._apply_recorded_initial_state()

#         # Let base class do its normal motion / state bookkeeping
#         super()._reset_env_tensors(env_ids)