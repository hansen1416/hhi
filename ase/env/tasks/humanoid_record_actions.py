import os
import torch
from isaacgym import gymtorch
from env.tasks.humanoid_hhi import HumanoidHHI


@torch.jit.script
def compute_record_motion_reset(reset_buf, motion_lengths, progress_buf, dt):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    """
    Same logic as your HumanoidViewMotion: terminate the episode when the
    (single) HUMOS motion reaches its end. No auto-reset / looping.
    """
    motion_times = progress_buf * dt
    reset = torch.where(motion_times > motion_lengths, torch.ones_like(reset_buf), torch.zeros_like(reset_buf))
    terminated = torch.zeros_like(reset_buf)          # we will set terminate manually after saving
    return reset, terminated


class HumanoidRecordActions(HumanoidHHI):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        """
        Records the exact PD action targets predicted by your trained model/policy
        for a *single* HUMOS motion (you ensure only one motion file is passed via
        cfg["env"]["motion_file"]).

        Changes from the original version you linked:
        • Forces numEnvs=1 (clean single-motion recording)
        • Records policy outputs until the motion ends
        • Stops completely (no reset/cycling) when the motion reaches its end
        • Saves the full action sequence + initial state to a .pt file for later playback
        • Fully compatible with your HUMOS 128 variations (body-shape + gender) and
          your ultimate goal of turning kinematic AMASS/HUMOS motions into stable
          physics-based motions.

        This is the "save the predicted action from model to local file" class.
        The companion playback class (HumanoidPlayRecordedActions) I gave you earlier
        will replay these saved actions without any neural network.
        """
        # Force PD control (policy outputs PD targets, exactly what we want to record)
        cfg = dict(cfg)  # don't mutate original
        cfg["env"]["pdControl"] = True
        control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)
        self._motion_dt = control_freq_inv * sim_params.dt   # needed for accurate motion-end detection
        cfg["env"]["controlFrequencyInv"] = control_freq_inv

        # Enforce single-env for single-motion recording
        if cfg["env"].get("numEnvs", 1) != 1:
            print("WARNING: HumanoidRecordActions is designed for numEnvs=1 (single motion). Forcing numEnvs=1.")
            cfg["env"]["numEnvs"] = 1

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.record_dir = os.path.join("/home/hlz/datasets/recorded_policy_actions_hhi/")
        os.makedirs(self.record_dir, exist_ok=True)

        self.recorded_actions = []
        self.initial_state = None
        self.motion_id = None
        self.is_recording = True

        print(f"RECORDING MODE ENABLED - SINGLE HUMOS MOTION (128 variations supported)")
        print(f"   Saving policy PD actions + initial state to: {self.record_dir}")
        print(f"   Will STOP automatically when the motion reaches its end (no reset/loop).")

    def pre_physics_step(self, actions):
        # Let base class apply the PD targets to the simulator
        super().pre_physics_step(actions)

        if self.is_recording:
            # Record the exact action vector the model just predicted
            self.recorded_actions.append(self.actions[0].cpu().clone().detach())

    def post_physics_step(self):
        super().post_physics_step()

        # If the motion has ended (reset_buf was set by _compute_reset)
        if self.reset_buf[0].item() > 0 and self.is_recording:
            self._save_recorded_actions()
            self.is_recording = False
            # Prevent any further reset and signal termination so the viewer / loop stops
            self.reset_buf[:] = 0
            self._terminate_buf[:] = 1
            print("Motion ended → recording saved. Environment stopped.")

    def _compute_reset(self):
        """Override to stop at the end of the single HUMOS motion."""
        motion_lengths = self._motion_lib.get_motion_length(self._sampled_motion_ids)
        self.reset_buf[:], self._terminate_buf[:] = compute_record_motion_reset(
            self.reset_buf, motion_lengths, self.progress_buf, self._motion_dt)
        return

    def _reset_env_tensors(self, env_ids):
        """
        Only perform the very first reset (to set up the single motion).
        After that we block all resets so the motion plays exactly once.
        """
        if self.initial_state is None:
            # First-time initialization
            self.motion_id = int(self._sampled_motion_ids[0].item())
            super()._reset_env_tensors(env_ids)

            # Capture exact initial state (same as in your original recorder)
            self.initial_state = {
                "root_pos": self._humanoid_root_states[0, 0:3].clone().cpu(),
                "root_rot": self._humanoid_root_states[0, 3:7].clone().cpu(),
                "dof_pos": self._dof_pos[0].clone().cpu(),
            }
            print(f"Started recording motion {self.motion_id} (HUMOS + shape + gender)")
        else:
            # Motion has already ended → do nothing (we already set terminate_buf)
            pass

    def _save_recorded_actions(self):
        if len(self.recorded_actions) < 10:
            print("Warning: Recorded sequence too short, skipping save.")
            return

        actions_seq = torch.stack(self.recorded_actions)   # [T, 207] (HUMOS action dim)

        data = {
            "motion_id": self.motion_id,
            "initial_root_pos": self.initial_state["root_pos"],
            "initial_root_rot": self.initial_state["root_rot"],
            "initial_dof_pos": self.initial_state["dof_pos"],
            "actions": actions_seq,          # ← this is the key data for playback
            "num_steps": len(actions_seq),
        }

        save_path = os.path.join(self.record_dir, f"motion_{self.motion_id:06d}_recorded.pt")
        torch.save(data, save_path)
        print(f"✅ SAVED {len(actions_seq)} policy-predicted PD actions for motion {self.motion_id} → {save_path}")