import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import numpy as np
from rl_games.common import a2c_common

import learning.amp_agent as amp_agent
from utils.running_mean_std import RunningMeanStd


@dataclass
class MultiMotionCfg:
    """
    Configuration for scaling AMP training from a single motion to many motions.

    This class mainly adds:
      (A) Visibility + control: evaluate performance and track per-motion failures.
      (B) Stability knobs: normalize discriminator reward; freeze observation RMS per epoch.
      (C) RNN support: AMP-aware rollout with terminate-masked bootstrapping.
    """

    # -------------------------
    # (A) Visibility + control
    # -------------------------

    # Track per-motion episode outcomes (success/termination) and report worst motions.
    track_motion_stats: bool = True

    # How many "worst" motion IDs to report (highest termination rate among seen motions).
    motion_stats_topk: int = 20

    # Optional: store per-step motion IDs in rollout buffers.
    # This is usually not needed; per-episode stats are much cheaper.
    store_motion_ids_in_rollout: bool = False

    # Enable periodic evaluation (separate from training rollouts).
    eval_enabled: bool = True

    # Evaluate every N epochs. Set 0 to disable auto-eval.
    eval_freq: int = 10

    # Number of episodes in eval. If 0, default to num_envs.
    eval_num_episodes: int = 0

    # Use deterministic policy (mu-actions) during eval to reduce noise.
    eval_deterministic: bool = True

    # Optional safety cap on the number of env steps during eval. 0 means no cap.
    eval_max_steps: int = 0

    # -------------------------
    # (B) Stability
    # -------------------------

    # Normalize discriminator reward with a running mean/std.
    # This can reduce reward scale drift when training on diverse motion sets.
    norm_disc_reward: bool = False

    # Optional affine transform applied after reward RMS normalization.
    # (PHC often used something like 0.5 * r + 0.25 after normalization.)
    disc_reward_norm_scale: float = 1.0
    disc_reward_norm_shift: float = 0.0

    # Freeze observation RMS for the whole epoch during gradient computation.
    # Motivation: avoid normalization drift between rollout-time targets and update-time inputs.
    freeze_obs_rms_per_epoch: bool = False


class MotionStats:
    """
    Per-motion episode statistics.

    We accumulate stats *per episode end*, which is cheap because:
      - 'done' indices are a small subset each step
      - we only update a handful of counts/sums on CPU

    Interpretation:
      - episodes[m]      = how many episodes ended while motion_id == m
      - terminations[m]  = how many of those ended by "terminate" (fall/early failure)
      - successes[m]     = episodes[m] - terminations[m] (not terminated)
      - return_sum[m]    = sum of episode returns for motion m
      - len_sum[m]       = sum of episode lengths for motion m
    """

    def __init__(self, num_motions: int):
        self.num_motions = int(num_motions)

        # Use CPU tensors to avoid GPU sync overhead for simple bookkeeping.
        self.episodes = torch.zeros(num_motions, dtype=torch.long)
        self.terminations = torch.zeros(num_motions, dtype=torch.long)
        self.successes = torch.zeros(num_motions, dtype=torch.long)

        self.return_sum = torch.zeros(num_motions, dtype=torch.float32)
        self.len_sum = torch.zeros(num_motions, dtype=torch.float32)

    def update(
        self,
        motion_ids: torch.Tensor,  # (K,) CPU long
        terminated: torch.Tensor,  # (K,) CPU bool
        ep_returns: torch.Tensor,  # (K,) CPU float
        ep_lens: torch.Tensor,     # (K,) CPU float
    ):
        """
        Update per-motion statistics for K finished episodes.

        Args:
            motion_ids: motion ID for each finished episode.
            terminated: True if the episode ended by failure (e.g., fall), not time-limit.
            ep_returns: episode return at termination.
            ep_lens: episode length at termination.
        """
        motion_ids = motion_ids.long()
        ones = torch.ones_like(motion_ids, dtype=torch.long)

        # episodes[m] += 1
        self.episodes.index_add_(0, motion_ids, ones)

        # terminations[m] += terminated
        term_long = terminated.long()
        self.terminations.index_add_(0, motion_ids, term_long)

        # successes[m] += (1 - terminated)
        self.successes.index_add_(0, motion_ids, (1 - term_long))

        # accumulate returns and lengths (useful for debugging and motion difficulty ranking)
        self.return_sum.index_add_(0, motion_ids, ep_returns.float())
        self.len_sum.index_add_(0, motion_ids, ep_lens.float())

    def summary(self, topk: int = 20) -> Dict[str, Any]:
        """
        Summarize overall coverage and worst motions by termination rate.

        Returns:
            coverage: number of motions that have been seen at least once
            total_episodes: total finished episodes across all motions
            termination_rate: global termination rate across all episodes
            top_fail_ids: motion IDs with highest termination rate among seen motions
            top_fail_rates: their termination rates
        """
        seen = self.episodes > 0
        coverage = int(seen.sum().item())
        total_eps = int(self.episodes.sum().item())

        if total_eps == 0:
            # nothing recorded yet
            return {
                "coverage": 0,
                "total_episodes": 0,
                "termination_rate": 0.0,
                "top_fail_ids": [],
                "top_fail_rates": [],
            }

        # global termination rate
        term_rate = (self.terminations.sum().float() / self.episodes.sum().float()).item()

        # per-motion termination rate; unseen motions are set to -1 to avoid dominating topk
        rates = torch.full_like(self.return_sum, -1.0)
        rates[seen] = self.terminations[seen].float() / self.episodes[seen].float()

        # top-k worst among seen motions
        k = min(int(topk), coverage)
        vals, idx = torch.topk(rates, k=k)

        return {
            "coverage": coverage,
            "total_episodes": total_eps,
            "termination_rate": float(term_rate),
            "top_fail_ids": idx.cpu().tolist(),
            "top_fail_rates": vals.cpu().tolist(),
        }


class PHCAgent(amp_agent.AMPAgent):
    """
    AMPAgent extension for multi-motion training.

    Adds three categories of functionality:

    (A) Visibility + control
        - Track per-motion termination/success statistics.
        - Provide a deterministic evaluation loop and periodic auto-eval.

    (B) Stability
        - Optional discriminator reward normalization (RunningMeanStd).
        - Optional "freeze observation RMS per epoch" snapshotting.

    (C) RNN policy support
        - Implements AMP-aware play_steps_rnn() so buffers contain amp_obs and
          next_values are masked by 'terminate' (failure) rather than only 'done'.

    Important assumption:
        Your environment/task must provide either:
          - infos["motion_ids"] each step, OR
          - task._sampled_motion_ids as a tensor-like of shape (num_envs,).
    """

    # -------------------------
    # config loading
    # -------------------------
    def _load_config_params(self, config):
        """
        Extend AMPAgent config parsing by reading a dedicated sub-config.
        This keeps the base agent config unchanged and makes features easy to toggle.
        """
        super()._load_config_params(config)

        mm = config.get("phc_agent", {}) or {}

        self._mm_cfg = MultiMotionCfg(
            track_motion_stats=bool(mm.get("track_motion_stats", True)),
            motion_stats_topk=int(mm.get("motion_stats_topk", 20)),
            store_motion_ids_in_rollout=bool(mm.get("store_motion_ids_in_rollout", False)),

            eval_enabled=bool(mm.get("eval_enabled", True)),
            eval_freq=int(mm.get("eval_freq", 10)),
            eval_num_episodes=int(mm.get("eval_num_episodes", 0)),
            eval_deterministic=bool(mm.get("eval_deterministic", True)),
            eval_max_steps=int(mm.get("eval_max_steps", 0)),

            norm_disc_reward=bool(mm.get("norm_disc_reward", False)),
            disc_reward_norm_scale=float(mm.get("disc_reward_norm_scale", 1.0)),
            disc_reward_norm_shift=float(mm.get("disc_reward_norm_shift", 0.0)),

            freeze_obs_rms_per_epoch=bool(mm.get("freeze_obs_rms_per_epoch", False)),
        )
        return

    def __init__(self, base_name, config):
        super().__init__(base_name, config)

        # (B1) Optional RunningMeanStd for discriminator reward.
        # Rationale: disc reward scale can drift as discriminator improves;
        # normalization can make PPO updates less sensitive to that drift.
        if self._mm_cfg.norm_disc_reward:
            self._disc_reward_mean_std = RunningMeanStd((1,)).to(self.ppo_device)
        else:
            self._disc_reward_mean_std = None

        # (B2) Observation RMS snapshot used during gradient computation.
        # When enabled, we take a snapshot at epoch start and use it
        # to preprocess observations in the training updates.
        self._obs_rms_snapshot = None
        self._use_obs_rms_snapshot = False

        # (A2) Per-motion stats (created lazily once motion_lib becomes available).
        self._motion_stats: Optional[MotionStats] = None

    # -------------------------
    # modes / checkpoint stats
    # -------------------------
    def set_eval(self):
        """
        Put networks (and RMS modules) into eval mode.
        For RMS, eval mode typically stops updates and uses fixed statistics.
        """
        super().set_eval()
        if self._disc_reward_mean_std is not None:
            self._disc_reward_mean_std.eval()

    def set_train(self):
        """
        Put networks (and RMS modules) into train mode.
        If disc reward normalization is enabled, its RMS should update in training.
        """
        super().set_train()
        if self._disc_reward_mean_std is not None:
            self._disc_reward_mean_std.train()

    def get_stats_weights(self):
        """
        Extend saved statistics to include discriminator reward RMS if enabled.
        This allows resume-training without disc reward scale discontinuities.
        """
        state = super().get_stats_weights()
        if self._disc_reward_mean_std is not None:
            state["disc_reward_mean_std"] = self._disc_reward_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        """
        Restore saved statistics. Safe-guarded: only loads disc RMS if present.
        """
        super().set_stats_weights(weights)
        if self._disc_reward_mean_std is not None and "disc_reward_mean_std" in weights:
            self._disc_reward_mean_std.load_state_dict(weights["disc_reward_mean_std"])

    # -------------------------
    # (A1) motion id access
    # -------------------------
    def _get_motion_ids(self, infos: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Obtain motion IDs for each environment.

        Preferred: infos["motion_ids"] (explicit and robust).
        Fallback: task._sampled_motion_ids (common pattern in motion imitation tasks).
        """
        if infos is not None and "motion_ids" in infos:
            mids = infos["motion_ids"]
            if torch.is_tensor(mids):
                return mids.to(self.device)
            return torch.as_tensor(mids, device=self.device, dtype=torch.long)

        task = self.vec_env.env.task
        if hasattr(task, "_sampled_motion_ids"):
            mids = task._sampled_motion_ids
            return mids.to(self.device) if torch.is_tensor(mids) else torch.as_tensor(
                mids, device=self.device, dtype=torch.long
            )

        raise RuntimeError(
            "Cannot obtain motion_ids: provide infos['motion_ids'] or task._sampled_motion_ids."
        )

    def _try_init_motion_stats(self):
        """
        Lazily initialize MotionStats. We only know num_motions once motion_lib exists.
        If motion_lib does not expose a motion count, disable stats instead of crashing.
        """
        if not self._mm_cfg.track_motion_stats:
            return
        if self._motion_stats is not None:
            return

        task = self.vec_env.env.task
        motion_lib = getattr(task, "_motion_lib", None)
        if motion_lib is None:
            return

        # Best-effort: common naming patterns used across motion libs.
        num_motions = None
        for attr in ["num_motions", "get_num_motions", "_num_motions"]:
            if hasattr(motion_lib, attr):
                v = getattr(motion_lib, attr)
                num_motions = int(v() if callable(v) else v)
                break

        if num_motions is None:
            # Disable silently because stats are a monitoring feature, not a core dependency.
            print("[PHCAgent] motion_lib has no num_motions; disabling per-motion stats.")
            self._mm_cfg.track_motion_stats = False
            return

        self._motion_stats = MotionStats(num_motions)

    # -------------------------
    # (B2) obs RMS snapshot
    # -------------------------
    def _begin_epoch_obs_rms_snapshot(self):
        """
        Snapshot observation RMS at the beginning of an epoch.

        Motivation:
            PPO computes targets (old logp, old values, advantages) using observations
            from rollout time. If normalization changes during the update phase, those
            targets become slightly inconsistent with the inputs used for gradient steps.

        Snapshotting makes training more stable in long runs and diverse motion sets.
        """
        if not self._mm_cfg.freeze_obs_rms_per_epoch:
            return
        if not hasattr(self, "running_mean_std") or self.running_mean_std is None:
            return

        # Deep copy so updates to the live RMS won't affect this snapshot.
        self._obs_rms_snapshot = copy.deepcopy(self.running_mean_std)

        # Ensure snapshot is "read-only": no updates.
        if hasattr(self._obs_rms_snapshot, "freeze"):
            self._obs_rms_snapshot.freeze()
        else:
            self._obs_rms_snapshot.eval()

    def _end_epoch_obs_rms_snapshot(self):
        """
        Clear snapshot state. Called at end of train_epoch().
        """
        self._obs_rms_snapshot = None
        self._use_obs_rms_snapshot = False

    def _preproc_obs(self, obs_batch):
        """
        Override observation preprocessing so we can optionally preprocess with the snapshot RMS.

        Implementation detail:
            We temporarily swap self.running_mean_std to the snapshot and call the base method.
            This avoids duplicating base preprocessing logic and keeps behavior consistent.
        """
        if not self._use_obs_rms_snapshot or self._obs_rms_snapshot is None:
            return super()._preproc_obs(obs_batch)

        old = self.running_mean_std
        self.running_mean_std = self._obs_rms_snapshot
        try:
            return super()._preproc_obs(obs_batch)
        finally:
            # Always restore even if an exception occurs.
            self.running_mean_std = old

    # -------------------------
    # (B1) disc reward normalization
    # -------------------------
    def _norm_disc_reward(self) -> bool:
        """
        Helper: whether disc reward normalization is enabled.
        """
        return self._disc_reward_mean_std is not None

    def _calc_disc_rewards(self, amp_obs):
        """
        Compute discriminator-based reward from AMP observations.

        AMP-style reward:
            r_disc = -log(max(1 - sigmoid(logit), eps))
        which increases as the discriminator becomes confident the sample is "real".

        Optional:
            Normalize r_disc with RMS before applying disc_reward_scale.
        """
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))

            # Numerically-safe: avoid log(0).
            disc_r = -torch.log(torch.maximum(
                1 - prob, torch.tensor(0.0001, device=self.ppo_device)
            ))

            if self._norm_disc_reward():
                # During training we want the RMS to track the current distribution.
                # During eval, set_eval() will put it into eval mode (no updates).
                self._disc_reward_mean_std.train()

                flat = disc_r.flatten()
                norm = self._disc_reward_mean_std(flat)
                disc_r = norm.reshape(disc_r.shape)

                # Optional affine re-scaling, useful to keep reward magnitude in a desired range.
                disc_r = self._mm_cfg.disc_reward_norm_scale * disc_r + self._mm_cfg.disc_reward_norm_shift

            # Global multiplier from AMP config.
            disc_r *= self._disc_reward_scale

        return disc_r

    # -------------------------
    # (A3) deterministic action + eval loop
    # -------------------------
    def get_action(self, obs_dict: Dict[str, torch.Tensor], deterministic: bool = True) -> torch.Tensor:
        """
        Inference-time action.

        - deterministic=True: returns policy mean (mu) for smooth playback and low-variance eval.
        - deterministic=False: returns sampled action for stochastic evaluation or exploration.
        """
        processed_obs = self._preproc_obs(obs_dict["obs"])
        self.model.eval()

        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)

        # Maintain RNN state for recurrent policies.
        if "rnn_states" in res_dict:
            self.rnn_states = res_dict["rnn_states"]

        return res_dict["mus"] if deterministic else res_dict["actions"]

    def env_eval_step(self, actions: torch.Tensor):
        """
        Single env step used by eval().

        Keeping a separate method makes it easy to:
          - change action conversion policy for eval
          - inject additional metrics / debug checks
        """
        return self.env_step(actions)

    def eval(self) -> Dict[str, float]:
        """
        Batched evaluation over many motions.

        We run episodes until collecting num_eps completed episodes, and report:
          - success rate (1 - termination rate)
          - termination rate
          - coverage (# unique motion ids observed in completed episodes)

        Note:
            This eval does not compute pose metrics (MPJPE). If you want those,
            expose them from the env via infos and aggregate similarly.
        """
        if not self._mm_cfg.eval_enabled:
            return {}

        self.set_eval()
        self._try_init_motion_stats()

        num_envs = self.num_actors
        num_eps = self._mm_cfg.eval_num_episodes if self._mm_cfg.eval_num_episodes > 0 else num_envs

        # Per-env running episode return/length.
        ep_ret = torch.zeros(num_envs, device=self.device)
        ep_len = torch.zeros(num_envs, device=self.device)

        done_indices = []
        finished = 0

        # Store termination + motion id for each completed episode.
        term_list = []
        mid_list = []

        steps = 0
        while finished < num_eps:
            # Optional safety cap to avoid extremely long loops if environment never terminates.
            if self._mm_cfg.eval_max_steps > 0 and steps >= self._mm_cfg.eval_max_steps:
                break
            steps += 1

            obs_dict = self.env_reset(done_indices)

            act = self.get_action(obs_dict, deterministic=self._mm_cfg.eval_deterministic)
            obs_dict, rewards, dones, infos = self.env_eval_step(act)

            # Accumulate per-env episode stats.
            ep_ret += rewards
            ep_len += 1

            terminated = infos.get("terminate", torch.zeros_like(dones)).bool()

            # done indices come in a flattened "agents x envs" scheme in rl-games.
            all_done = dones.nonzero(as_tuple=False)
            done_envs = all_done[:: self.num_agents][:, 0] if len(all_done) > 0 else None
            if done_envs is None or done_envs.numel() == 0:
                continue

            # Record per-episode termination and motion ids at episode end.
            mids = self._get_motion_ids(infos)
            term_done = terminated[done_envs].detach().cpu()
            mid_done = mids[done_envs].detach().cpu()

            term_list.append(term_done)
            mid_list.append(mid_done)

            # Reset per-env accumulators for completed envs.
            ep_ret[done_envs] = 0
            ep_len[done_envs] = 0

            # Reset recurrent states when an env finishes an episode.
            if self.is_rnn and self.rnn_states is not None:
                for s in self.rnn_states:
                    s[:, all_done, :] = 0.0

            finished += int(done_envs.numel())
            done_indices = done_envs

        if len(term_list) == 0:
            return {
                "eval/success_rate": 0.0,
                "eval/termination_rate": 0.0,
                "eval/coverage": 0.0,
            }

        term_all = torch.cat(term_list, dim=0).float()
        mids_all = torch.cat(mid_list, dim=0)

        term_rate = term_all.mean().item()
        succ_rate = 1.0 - term_rate

        # Coverage among completed episodes (not necessarily full dataset coverage).
        coverage = float(len(torch.unique(mids_all))) if mids_all.numel() > 0 else 0.0

        return {
            "eval/success_rate": float(succ_rate),
            "eval/termination_rate": float(term_rate),
            "eval/coverage": coverage,
        }

    # -------------------------
    # (A2) update motion stats on episode completion
    # -------------------------
    def _update_motion_stats_on_done(self, done_envs: torch.Tensor, infos: Dict[str, Any], terminated: torch.Tensor):
        """
        Update MotionStats when some environments finished episodes.

        We use agent-side accumulators:
          - self.current_rewards : episode return so far
          - self.current_lengths : episode length so far
        The base rl-games loop resets these after logging; we must update stats *before* reset.
        """
        if not self._mm_cfg.track_motion_stats:
            return

        self._try_init_motion_stats()
        if self._motion_stats is None:
            return

        mids = self._get_motion_ids(infos)[done_envs].detach().cpu().long()
        term = terminated[done_envs].detach().cpu().bool()

        # current_rewards is often shape (N, 1); squeeze to (N,)
        ep_r = self.current_rewards[done_envs].detach().cpu()
        if ep_r.ndim > 1:
            ep_r = ep_r.squeeze(-1)

        ep_l = self.current_lengths[done_envs].detach().cpu().float()

        self._motion_stats.update(mids, term, ep_r, ep_l)

    # -------------------------
    # (C) AMP-aware play_steps_rnn
    # -------------------------
    def play_steps_rnn(self):
        """
        RNN rollout with AMP bookkeeping.

        Key differences vs a generic rl-games RNN rollout:
          1) Store infos["amp_obs"] into the experience buffer.
          2) Compute next_values with critic and apply terminate-masking:
                 next_vals *= (1 - terminate)
             This avoids bootstrapping through failure transitions (falls).
          3) Store rand_action_mask if you use eps-greedy action mixing.
        """
        self.set_eval()

        mb_rnn_states = []

        # Reset buffer contents to avoid leftover values from previous epoch.
        self.experience_buffer.tensor_dict["values"].fill_(0)
        self.experience_buffer.tensor_dict["rewards"].fill_(0)
        self.experience_buffer.tensor_dict["dones"].fill_(1)

        update_list = self.update_list
        batch_size = self.num_agents * self.num_actors

        mb_rnn_masks = None
        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(
            batch_size, mb_rnn_states
        )

        done_indices = []

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)

            # Select the subset of envs that should step next (rl-games RNN scheduling).
            seq_indices, full_tensor = self.process_rnn_indices(
                mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states
            )
            if full_tensor:
                break

            # Get action/value/logp/mu/sigma/... for the current observation.
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)

            # Track RNN states for next timestep.
            self.rnn_states = res_dict["rnn_states"]

            # Store observation (obses) into the RNN-formatted experience buffer.
            self.experience_buffer.update_data_rnn("obses", indices, play_mask, self.obs["obs"])

            # Store standard PPO fields (actions, logp, values, mu, sigma, etc.).
            for k in update_list:
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data_rnn(
                    "states",
                    indices[:: self.num_agents],
                    play_mask[:: self.num_agents] // self.num_agents,
                    self.obs["states"],
                )

            # Step the environment with chosen actions.
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)

            # Store transition fields.
            self.experience_buffer.update_data_rnn("rewards", indices, play_mask, shaped_rewards)
            self.experience_buffer.update_data_rnn("next_obses", indices, play_mask, self.obs["obs"])
            self.experience_buffer.update_data_rnn("dones", indices, play_mask, self.dones.byte())

            # AMP-specific: store amp observations used by discriminator reward and loss.
            self.experience_buffer.update_data_rnn("amp_obs", indices, play_mask, infos["amp_obs"])

            # Eps-greedy bookkeeping: which envs used random actions vs deterministic mu.
            self.experience_buffer.update_data_rnn("rand_action_mask", indices, play_mask, res_dict["rand_action_mask"])

            # Terminate flag indicates failure termination (e.g., fall) distinct from time-limit.
            terminated = infos["terminate"].float().unsqueeze(-1)

            # Critic bootstrap: evaluate V(s_{t+1}), but do NOT bootstrap through failure transitions.
            input_dict = {"obs": self.obs["obs"], "rnn_states": self.rnn_states}
            next_vals = self._eval_critic(input_dict)
            next_vals *= (1.0 - terminated)

            self.experience_buffer.update_data_rnn("next_values", indices, play_mask, next_vals)

            # Update per-env episode accumulators (used by logging and MotionStats).
            self.current_rewards += rewards
            self.current_lengths += 1

            all_done = self.dones.nonzero(as_tuple=False)
            done_envs = all_done[:: self.num_agents][:, 0] if len(all_done) > 0 else None

            # Update per-motion episode statistics before current_rewards/current_lengths get reset.
            if done_envs is not None and done_envs.numel() > 0:
                self._update_motion_stats_on_done(done_envs, infos, infos["terminate"].bool())

            # Handle RNN done logic and observer hooks.
            self.process_rnn_dones(all_done, indices, seq_indices)
            self.algo_observer.process_infos(infos, all_done[:: self.num_agents])

            # Update episode reward/length meters used by base logging.
            not_dones = 1.0 - self.dones.float()
            self.game_rewards.update(self.current_rewards[all_done[:: self.num_agents]])
            self.game_lengths.update(self.current_lengths[all_done[:: self.num_agents]])

            # Reset accumulators for finished envs (mask style).
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_envs if done_envs is not None else []

        # After rollout, compute AMP rewards and PPO advantages/returns.
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]

        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        # Flatten [T, N, ...] -> [T*N, ...] for training.
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)

        # RNN training additionally needs stored rnn_states and masks.
        batch_dict["rnn_states"] = mb_rnn_states
        batch_dict["rnn_masks"] = mb_rnn_masks

        # Played frames is used by some schedulers/loggers.
        batch_dict["played_frames"] = n * self.num_actors * self.num_agents

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        batch_dict["mb_rewards"] = a2c_common.swap_and_flatten01(mb_rewards)
        return batch_dict

    # -------------------------
    # train_epoch override: snapshot + stats + auto-eval
    # -------------------------
    def train_epoch(self):
        """
        One training epoch:
          1) (Optional) snapshot obs RMS for stable gradient preprocessing
          2) roll out horizon steps (RNN/non-RNN)
          3) sample demo and replay AMP obs
          4) run PPO + discriminator updates via train_actor_critic
          5) record motion stats summary and periodic eval
        """
        self._begin_epoch_obs_rms_snapshot()
        self._try_init_motion_stats()

        play_time_start = time.time()

        with torch.no_grad():
            # During rollout we generally do NOT want to use the snapshot:
            # rollout uses eval mode (no RMS update). Using snapshot here is unnecessary.
            self._use_obs_rms_snapshot = False

            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()

        # Refresh demo buffer and prepare discriminator batches.
        self._update_amp_demos()
        num_obs_samples = batch_dict["amp_obs"].shape[0]
        batch_dict["amp_obs_demo"] = self._amp_obs_demo_buffer.sample(num_obs_samples)["amp_obs"]
        batch_dict["amp_obs_replay"] = (
            batch_dict["amp_obs"]
            if (self._amp_replay_buffer.get_total_count() == 0)
            else self._amp_replay_buffer.sample(num_obs_samples)["amp_obs"]
        )

        # Switch to training mode for gradient updates.
        self.set_train()

        # During gradient computation, optionally preprocess obs with frozen RMS snapshot.
        self._use_obs_rms_snapshot = bool(self._mm_cfg.freeze_obs_rms_per_epoch)

        self.curr_frames = batch_dict.pop("played_frames")

        # Convert raw rollout batch into dataset buffers used by rl-games PPO loops.
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        # Run PPO mini-epochs/minibatches.
        train_info = None
        for _ in range(0, self.mini_epochs_num):
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                # Aggregate outputs from many minibatches for logging.
                if train_info is None:
                    train_info = {k: [v] for k, v in curr_train_info.items()}
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

        update_time_end = time.time()

        # Store latest agent AMP obs into replay buffer.
        self._store_replay_amp_obs(batch_dict["amp_obs"])

        # Record timing.
        train_info["play_time"] = play_time_end - play_time_start
        train_info["update_time"] = update_time_end - update_time_start
        train_info["total_time"] = update_time_end - play_time_start

        # Optional: mean reward of the rollout batch (useful scalar).
        train_info["mb_rewards"] = batch_dict.get("mb_rewards", None)

        # (A2) Report motion coverage and worst motions by termination rate.
        if self._motion_stats is not None:
            ms = self._motion_stats.summary(topk=self._mm_cfg.motion_stats_topk)
            train_info["motion/coverage"] = ms["coverage"]
            train_info["motion/termination_rate"] = ms["termination_rate"]
            train_info["motion/top_fail_ids"] = ms["top_fail_ids"]
            train_info["motion/top_fail_rates"] = ms["top_fail_rates"]

        # (A3) Periodic auto-eval (deterministic episodes).
        if self._mm_cfg.eval_enabled and self._mm_cfg.eval_freq > 0:
            if (self.epoch_num % self._mm_cfg.eval_freq) == 0:
                eval_info = self.eval()
                train_info.update(eval_info)

        # Delegate to base logging hooks (tensorboard, stdout, etc.).
        self._record_train_batch_info(batch_dict, train_info)

        # Clear snapshot.
        self._end_epoch_obs_rms_snapshot()
        return train_info
