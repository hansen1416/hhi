import os
import torch
from isaacgym import gymtorch
from env.tasks.humanoid_hhi import HumanoidHHI

class HumanoidReplayActions(HumanoidHHI):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        cfg = dict(cfg)
        cfg["env"]["pdControl"] = True          # must use PD controller
        cfg["env"]["controlFrequencyInv"] = 1

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.replay_dir = cfg["env"].get("replay_dir", "./recorded_policy_actions_hhi")
        self.replay_data = {}  # motion_id → recorded dict

        # Load all saved recordings
        for fname in os.listdir(self.replay_dir):
            if fname.endswith(".pt"):
                data = torch.load(os.path.join(self.replay_dir, fname), map_location="cpu")
                mid = data["motion_id"]
                self.replay_data[mid] = data
                print(f"📼 Loaded replay data for motion {mid} ({data['num_steps']} steps)")

        self.replay_action_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.replay_actions = [None] * self.num_envs

        print(f"✅ REPLAY MODE: Loaded {len(self.replay_data)} motion sequences (HUMOS variations)")

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        for env_id in env_ids_list:
            motion_id = int(self._sampled_motion_ids[env_id].item())
            if motion_id in self.replay_data:
                data = self.replay_data[motion_id]
                self.replay_actions[env_id] = data["actions"].to(self.device)
                self.replay_action_idx[env_id] = 0

                # Force exact saved initial state
                root_pos = data["initial_root_pos"].to(self.device).unsqueeze(0)
                root_rot = data["initial_root_rot"].to(self.device).unsqueeze(0)
                dof_pos  = data["initial_dof_pos"].to(self.device).unsqueeze(0)

                self._reset_actors(env_ids=[env_id],
                                   root_pos=root_pos,
                                   root_rot=root_rot,
                                   dof_pos=dof_pos,
                                   root_vel=torch.zeros_like(root_pos),
                                   root_ang_vel=torch.zeros(1, 3, device=self.device),
                                   dof_vel=torch.zeros_like(dof_pos))

                print(f"🔄 Replaying saved policy motion {motion_id} (HUMOS variation)")
            else:
                print(f"⚠️ No replay data for motion {motion_id}")

    def pre_physics_step(self, actions):  # ignore RL policy output
        super().pre_physics_step(actions)

        # Feed recorded PD targets into the controller
        for i in range(self.num_envs):
            idx = self.replay_action_idx[i].item()
            if idx < len(self.replay_actions[i]):
                self.actions[i] = self.replay_actions[i][idx].clone()
                self.replay_action_idx[i] += 1
            else:
                self.actions[i].zero_()

        super().pre_physics_step(self.actions)  # base class converts to PD targets + applies