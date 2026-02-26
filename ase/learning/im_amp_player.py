"""IM (Imitation) AMP player (minimal).

This is a PHC-style evaluation player adapted for an ASE/rl_games AMP stack.

Goal:
- Evaluate ONE policy (one "primitive") against MANY reference motions.
- Always reset to the initial state between motion batches (no fail recovery).

Expected env/task interface (same conventions as PHC):
- self.env.task.num_envs : int
- self.env.task.forward_motion_samples() : loads the next batch of motions (one per env)
- self.env.task._motion_lib.get_motion_num_steps() -> torch.Tensor[num_envs]
- self.env.task._motion_lib._num_unique_motions : int
- self.env.task._motion_lib._curr_motion_ids : torch.Tensor[num_envs]

Expected info keys from env_step:
- info["terminate"] : torch.BoolTensor[num_envs] (early termination signal)
- either:
  - info["mpjpe"] : torch.Tensor[num_envs]
  - OR info["body_pos"] and info["body_pos_gt"] as arrays/tensors [num_envs, J, 3]

This file intentionally omits optional PHC features (dataset dumping, z collection,
compute_metrics_lite, debug breakpoints).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from tqdm import tqdm

import learning.amp_players as amp_players


class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    """Minimal PHC-style imitation evaluation player."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.im_eval: bool = bool(config.get("im_eval", True))
        self._im_finished: bool = False

        humanoid_env = self.env.task
        self.num_envs: int = int(humanoid_env.num_envs)

        # Disable recovery/fall init if present.
        if hasattr(humanoid_env, "_recovery_episode_prob"):
            humanoid_env._recovery_episode_prob = 0
        if hasattr(humanoid_env, "_fall_init_prob"):
            humanoid_env._fall_init_prob = 0

        # Book-keeping for success-rate.
        self.terminate_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.terminate_memory: List[np.ndarray] = []

        # MPJPE aggregation (store per-step vectors, then reduce per motion).
        self._mpjpe_steps: List[torch.Tensor] = []
        self.mpjpe_per_motion: List[float] = []

        self.curr_steps: int = 0

        # Progress over the full motion set (best-effort).
        self.total_motions: Optional[int] = getattr(humanoid_env._motion_lib, "_num_unique_motions", None)
        if self.total_motions is not None and self.total_motions > 0:
            num_batches = int(math.ceil(self.total_motions / self.num_envs))
            self.pbar = tqdm(total=num_batches, disable=not self.im_eval)
        else:
            self.pbar = tqdm(disable=True)

    # -----------------------------
    # Helpers
    # -----------------------------

    def _get_motion_steps(self) -> torch.Tensor:
        humanoid_env = self.env.task
        steps = humanoid_env._motion_lib.get_motion_num_steps()
        if not torch.is_tensor(steps):
            steps = torch.as_tensor(steps, device=self.device)
        return steps

    def _get_valid_bound(self, motion_steps: torch.Tensor) -> int:
        """Best-effort handling of the final (possibly padded) batch.

        PHC sometimes pads the last batch by repeating the last motion id.
        We mimic PHC's heuristic: if curr_motion_ids contain the max id,
        treat everything after the first occurrence as padding.
        """
        humanoid_env = self.env.task
        total = getattr(humanoid_env._motion_lib, "_num_unique_motions", None)
        if total is None:
            return self.num_envs

        max_id = int(total - 1)
        curr_ids = getattr(humanoid_env._motion_lib, "_curr_motion_ids", None)
        if curr_ids is None:
            return self.num_envs

        if not torch.is_tensor(curr_ids):
            curr_ids = torch.as_tensor(curr_ids, device=self.device)

        # If max id appears, assume padding begins at its first occurrence.
        max_mask = (curr_ids == max_id)
        if max_mask.any():
            first = int(max_mask.nonzero(as_tuple=False)[0].item())
            return min(first + 1, self.num_envs)
        return self.num_envs

    def _compute_mpjpe(self, info: Dict[str, Any]) -> torch.Tensor:
        """Return MPJPE per env as a torch.Tensor[num_envs]."""
        if "mpjpe" in info:
            mpjpe = info["mpjpe"]
            if not torch.is_tensor(mpjpe):
                mpjpe = torch.as_tensor(mpjpe, device=self.device)
            return mpjpe

        if "body_pos" in info and "body_pos_gt" in info:
            pred = info["body_pos"]
            gt = info["body_pos_gt"]
            if not torch.is_tensor(pred):
                pred = torch.as_tensor(pred, device=self.device)
            if not torch.is_tensor(gt):
                gt = torch.as_tensor(gt, device=self.device)
            # pred/gt: [N, J, 3]
            diff = pred - gt
            mpjpe = torch.linalg.norm(diff, dim=-1).mean(dim=-1)  # [N]
            return mpjpe

        raise RuntimeError(
            "IMAMPPlayerContinuous requires either info['mpjpe'] or info['body_pos']+info['body_pos_gt']."
        )

    def _finalize_batch(self) -> None:
        """Reduce per-step MPJPE into per-motion MPJPE and update success stats."""
        humanoid_env = self.env.task

        motion_steps = self._get_motion_steps()  # [N]
        valid_bound = self._get_valid_bound(motion_steps)

        # Success bookkeeping.
        self.terminate_memory.append(self.terminate_state[:valid_bound].detach().cpu().numpy())

        # MPJPE per motion.
        if len(self._mpjpe_steps) == 0:
            per_step = None
        else:
            per_step = torch.stack(self._mpjpe_steps, dim=0)  # [T, N]

        if per_step is not None:
            for env_id in range(valid_bound):
                n_steps = int(motion_steps[env_id].item())
                # PHC averages [: (n_steps - 1)] to avoid counting the first frame.
                end = max(n_steps - 1, 1)
                end = min(end, per_step.shape[0])
                self.mpjpe_per_motion.append(float(per_step[:end, env_id].mean().item()))

        # Progress bar update.
        if hasattr(self, "pbar") and self.pbar is not None:
            term_count = int(self.terminate_state[:valid_bound].sum().item())
            succ_rate = self.get_success_rate()
            mpjpe_mean = self.get_mean_mpjpe()
            start_idx = getattr(humanoid_env, "start_idx", None)
            desc = (
                f"Terminated: {term_count}/{valid_bound} | "
                f"steps: {self.curr_steps} | "
                + (f"Start: {start_idx} | " if start_idx is not None else "")
                + f"Succ: {succ_rate:.3f} | MPJPE: {mpjpe_mean * 1000:.2f}mm"
            )
            self.pbar.set_description(desc)
            self.pbar.update(1)
            self.pbar.refresh()

        # Prepare next batch.
        self.terminate_state.zero_()
        self._mpjpe_steps.clear()
        self.curr_steps = 0

        # Load next motions.
        humanoid_env.forward_motion_samples()

    def _check_finished(self) -> bool:
        """Return True if we've evaluated all motions (best-effort)."""
        humanoid_env = self.env.task
        total = getattr(humanoid_env._motion_lib, "_num_unique_motions", None)
        if total is None:
            return False

        start_idx = getattr(humanoid_env, "start_idx", None)
        if start_idx is None:
            return False

        return bool(start_idx >= total)

    def get_success_rate(self) -> float:
        if len(self.terminate_memory) == 0:
            return 0.0
        term = np.concatenate(self.terminate_memory, axis=0)
        # Clip to total motions if known.
        if self.total_motions is not None:
            term = term[: self.total_motions]
        return float(1.0 - term.mean())

    def get_mean_mpjpe(self) -> float:
        if len(self.mpjpe_per_motion) == 0:
            return 0.0
        if self.total_motions is not None:
            vals = self.mpjpe_per_motion[: self.total_motions]
        else:
            vals = self.mpjpe_per_motion
        return float(np.mean(vals))

    # -----------------------------
    # Hooks used by run loop
    # -----------------------------

    def _post_step(self, info: Dict[str, Any], done: torch.Tensor) -> torch.Tensor:
        # Keep parent behaviour (viewer debug, etc.).
        super()._post_step(info)

        if not self.im_eval or self._im_finished:
            return done

        humanoid_env = self.env.task
        motion_steps = self._get_motion_steps()  # [N]

        # Early termination should count as failure only if it happens before the last reference frame.
        terminate = info.get("terminate", None)
        if terminate is None:
            # Fallback: if env doesn't provide a terminate signal, treat done as terminate.
            terminate = done.clone().view(-1)[: self.num_envs]
        if not torch.is_tensor(terminate):
            terminate = torch.as_tensor(terminate, device=self.device, dtype=torch.bool)

        early_term = terminate & (self.curr_steps < (motion_steps - 1))
        self.terminate_state |= early_term

        # Record MPJPE for this step.
        mpjpe_vec = self._compute_mpjpe(info).view(-1)[: self.num_envs]
        self._mpjpe_steps.append(mpjpe_vec.detach())

        self.curr_steps += 1

        # Decide whether to end the batch.
        active_mask = ~self.terminate_state
        if active_mask.any():
            curr_max = int(motion_steps[active_mask].max().item())
        else:
            curr_max = int(motion_steps.max().item())

        batch_done = (self.curr_steps >= curr_max) or (int(self.terminate_state.sum().item()) == self.num_envs)

        if batch_done:
            # Finalize stats for this batch.
            self._finalize_batch()

            # Mark all envs done so the player loop triggers reset.
            done[:] = 1

            # If env uses start_idx to indicate completion, stop once it passes the motion count.
            if self._check_finished():
                self._im_finished = True
                if hasattr(self, "pbar") and self.pbar is not None:
                    self.pbar.close()

                print("------------------------------------------")
                print(f"IM Eval finished | Success Rate: {self.get_success_rate():.6f} | Mean MPJPE: {self.get_mean_mpjpe() * 1000:.2f} mm")

        return done

    # -----------------------------
    # Custom run loop (needed to thread `done` through _post_step)
    # -----------------------------

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic

        n_games = n_games * n_game_life
        games_played = 0

        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        sum_rewards = 0.0
        sum_steps = 0.0

        for _ in range(n_games):
            if self._im_finished or games_played >= n_games:
                break

            obs_dict = self.env_reset()
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            done_indices = []

            with torch.no_grad():
                for _ in range(self.max_steps):
                    if self._im_finished:
                        break

                    obs_dict = self.env_reset(done_indices)

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    done = self._post_step(info, done.clone())

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)

                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())

                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        if games_played >= n_games:
                            break

                    # rl_games expects done_indices to be 1-D indices.
                    if torch.is_tensor(done_indices) and done_indices.ndim == 2 and done_indices.shape[-1] == 1:
                        done_indices = done_indices[:, 0]

        if not self._im_finished:
            # Fallback summary (useful when games_num ends early).
            print("------------------------------------------")
            print(
                f"IM Eval stopped | games_played={games_played} | "
                f"Success Rate: {self.get_success_rate():.6f} | "
                f"Mean MPJPE: {self.get_mean_mpjpe() * 1000:.2f} mm"
            )

        # Standard player stats.
        if games_played > 0:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

        return
