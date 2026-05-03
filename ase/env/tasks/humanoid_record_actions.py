import os
import torch
from isaacgym import gymtorch
from env.tasks.humanoid_hhi import HumanoidHHI

class HumanoidRecordActions(HumanoidHHI):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        # Force PD control mode (the trained policy outputs PD targets)
        cfg = dict(cfg)  # don't mutate original
        cfg["env"]["pdControl"] = True
        cfg["env"]["controlFrequencyInv"] = cfg["env"].get("controlFrequencyInv", 1)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.record_actions_enabled = True
        self.record_dir = cfg["env"].get("record_dir", "./recorded_policy_actions_hhi")
        os.makedirs(self.record_dir, exist_ok=True)

        self.action_records = [[] for _ in range(self.num_envs)]
        self.initial_states = [None] * self.num_envs

        print(f"✅ RECORDING MODE ENABLED (HUMOS 128 variations supported)")
        print(f"   Saving policy actions + initial states to: {self.record_dir}")

    def pre_physics_step(self, actions):
        # Let base class apply PD control normally
        super().pre_physics_step(actions)

        if self.record_actions_enabled:
            # self.actions is the exact PD target vector the policy produced (shape [num_envs, 207])
            for i in range(self.num_envs):
                self.action_records[i].append(self.actions[i].cpu().clone().detach())

    def _reset_env_tensors(self, env_ids):
        # 1. Save completed episode for these envs
        if self.record_actions_enabled:
            env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
            for env_id in env_ids_list:
                if len(self.action_records[env_id]) > 10:  # skip tiny episodes
                    motion_id = int(self._sampled_motion_ids[env_id].item())  # HUMOS motion+shape+gender ID
                    actions_seq = torch.stack(self.action_records[env_id])   # [T, num_actions]

                    data = {
                        "motion_id": motion_id,
                        "initial_root_pos": self.initial_states[env_id]["root_pos"],
                        "initial_root_rot": self.initial_states[env_id]["root_rot"],
                        "initial_dof_pos": self.initial_states[env_id]["dof_pos"],
                        "actions": actions_seq,          # ← this is what we will replay
                        "num_steps": len(actions_seq),
                    }
                    save_path = os.path.join(self.record_dir, f"motion_{motion_id:06d}_env{env_id:03d}.pt")
                    torch.save(data, save_path)
                    print(f"💾 Saved {len(actions_seq)} policy actions for motion {motion_id} → {save_path}")

                self.action_records[env_id] = []  # clear for next episode

        # 2. Let base HHI do normal reset (samples new HUMOS motion, etc.)
        super()._reset_env_tensors(env_ids)

        # 3. Capture fresh initial state for the *new* episode
        if self.record_actions_enabled:
            env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
            for env_id in env_ids_list:
                self.initial_states[env_id] = {
                    "root_pos": self._humanoid_root_states[env_id, 0:3].clone().cpu(),
                    "root_rot": self._humanoid_root_states[env_id, 3:7].clone().cpu(),
                    "dof_pos": self._dof_pos[env_id].clone().cpu(),
                }