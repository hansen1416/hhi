from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv

from isaacgym.torch_utils import *

import os
import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn
import wandb

import learning.replay_buffer as replay_buffer
import learning.common_agent as common_agent 

from ase.learning.wandb_logger import wandb_logger

class HHIAgent(common_agent.CommonAgent):
    """
    AMP (Adversarial Motion Priors) agent built on top of an RL-Games PPO-style agent.

    High-level algorithmic structure (per epoch):
      1) Rollout policy for horizon_length steps (play_steps / play_steps_rnn)
         - store standard PPO tensors (obs, actions, values, logp, etc.)
         - store AMP-specific tensors (amp_obs) from the environment infos
         - store rand_action_mask (eps-greedy mixing between stochastic and deterministic actions)
      2) Sample discriminator "real" examples from demo buffer (amp_obs_demo)
      3) Sample discriminator "fake" examples from replay buffer (amp_obs_replay)
      4) Train actor-critic + discriminator jointly in train_actor_critic / calc_gradients
      5) Push latest fake amp_obs into replay buffer

    Core AMP idea:
      - Discriminator distinguishes real motion features (demo) vs agent motion features (rollout/replay).
      - Discriminator output is used both:
          (a) as an additional reward signal (disc_rewards)
          (b) as a discriminator training objective (disc_loss, plus regularizers)
    """
    def __init__(self, base_name, config):
        """
        Construct the agent. Base class CommonAgent handles:
          - env creation / vec_env wrapper
          - PPO networks (actor-critic, optional RNN)
          - observation/value normalizers (running_mean_std, value_mean_std)
          - experience buffer allocation

        AMPAgent additionally creates an RMS normalizer for amp_obs (if enabled).
        """
        super().__init__(base_name, config)

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        return

    def init_tensors(self):
        """
        Called after base buffers are created.
        We extend the experience buffer to store AMP-specific tensors (amp_obs, rand_action_mask, etc.).
        """
        super().init_tensors()
        self._build_amp_buffers()
        return
    
    def set_eval(self):
        """
        Switch model(s) into evaluation mode for rollout.
        Important for components like dropout / batchnorm (if any),
        and to prevent RMS modules from being updated if they use train/eval behavior.
        """
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()
        return

    def set_train(self):
        """
        Switch model(s) into training mode for gradient updates.
        """
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()
        return

    def get_stats_weights(self):
        """
        Save statistics-related state (RMS normalizers) into checkpoint.

        Base agent typically includes:
          - running_mean_std (obs normalization)
          - value_mean_std (value normalization) if enabled
        AMPAgent adds:
          - _amp_input_mean_std (amp_obs normalization)
        """
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()
        
        return state

    def set_stats_weights(self, weights):
        """
        Restore statistics-related state from checkpoint.
        """
        super().set_stats_weights(weights)
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
        
        return

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


    def play_steps(self):
        """
        Collect one rollout batch of length horizon_length (non-RNN version).

        Data flow:
          - env_reset(done_indices): reset only environments that ended last step
          - get_action_values(): compute actions + auxiliary PPO fields
          - env_step(actions): step physics and receive infos['amp_obs'] and infos['terminate']
          - store into experience_buffer for PPO training

        Termination handling:
          - infos['terminate'] indicates failure termination (e.g., fall).
          - next_values are multiplied by (1 - terminate) so we do not bootstrap through failure.
        """
        self.set_eval()
        
        # placeholder for episode info aggregation (often unused in this file)
        epinfos = []
        # env indices that need reset at the next step
        done_indices = []
        # list of tensor names to store (actions, values, logp, mu, sigma, etc.)
        update_list = self.update_list

        # horizon length is 32, is it optimal?
        for n in range(self.horizon_length):
            # Reset environments that ended previously (partial reset)
            self.obs = self.env_reset(done_indices)
            # Store current observations.
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            # Compute actions and PPO tensors.
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs, self._rand_action_probs)
            # Store per-step rollout data needed for PPO loss computation.
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            # For asymmetric actor-critic: store privileged "states" for central value net.
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            
            # Step the environment with chosen actions.
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            
            # Apply optional reward shaping (often identity, but can scale/clip).
            shaped_rewards = self.rewards_shaper(rewards)

            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            # AMP-specific: store discriminator input features from env.
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])

            # the shape of infos['amp_shape'] is [num_env, 11]
            # and it becomes [num_env * horizon_length, 11]
            # disc-shape-condition
            self.experience_buffer.update_data('amp_shape', n, infos['amp_shape'])

            # Store which envs used random vs deterministic actions this step.
            # Used to mask actor loss/entropy/bounds loss in calc_gradients().
            self.experience_buffer.update_data('rand_action_mask', n, res_dict['rand_action_mask'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            # critic learns a value function for PPO
            # # Source: learning/hhi_network_builder.py -> HHIBuilder.Network.eval_critic
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            # Bookkeeping for episode-level logging.
            self.current_rewards += rewards
            self.current_lengths += 1

            # RL-Games uses (num_agents * num_envs) layout; take every num_agents entry.
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
  
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)
            
            # Reset episode accumulators for envs that finished
            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            # Optional runtime visualization / debug (Isaac Gym viewer).
            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)
                
            done_indices = done_indices[:, 0]

        # ----- After rollout: compute AMP rewards, combined rewards, and GAE returns -----

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        # disc-shape-condition
        mb_amp_shape = self.experience_buffer.tensor_dict['amp_shape']
        # Runs discriminator on AMP obs to compute style rewards
        amp_rewards = self._calc_amp_rewards(mb_amp_obs, mb_amp_shape)
        # Combine task reward and discriminator reward via configured weights, eg. 0.5, 0.5.
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        # Generalized Advantage Estimation (GAE) or discounted returns.
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values
        # Flatten rollout tensors into a batch for PPO training.
        # here will pack amp_shape
        # disc-shape-condition
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        # Add AMP reward tensors into the batch_dict for logging and diagnostics.
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        batch_dict['task_reward'] = mb_rewards.mean().item()

        return batch_dict
    
    def get_action_values(self, obs_dict, rand_action_probs):
        """
        Forward pass through the policy to obtain actions + PPO fields.

        Eps-greedy (deterministic/stochastic mixture):
          - Sample rand_action_mask ~ Bernoulli(rand_action_probs) per env.
          - For envs where mask == 0, replace sampled action with mean action (mu).
          - This yields smoother trajectories for some envs, which can help discriminator training:
              discriminator should not trivially classify "fake" by action noise/jitter.
        """
        processed_obs = self._preproc_obs(obs_dict['obs'])

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)

            # If using a central value function with privileged states, override values.
            if self.has_central_value:
                states = obs_dict['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value

        # If value normalization is enabled, denormalize/normalize accordingly.
        # In rl-games, value_mean_std(x, True) often means "denorm for inference" or
        # "apply in eval mode"; semantics depend on their RMS implementation.
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        
        # Bernoulli per env: 1 => keep stochastic action, 0 => use deterministic mean action.
        rand_action_mask = torch.bernoulli(rand_action_probs)
        det_action_mask = rand_action_mask == 0.0
        res_dict['actions'][det_action_mask] = res_dict['mus'][det_action_mask]

        # Store mask for loss masking (calc_gradients).
        res_dict['rand_action_mask'] = rand_action_mask

        return res_dict

    def prepare_dataset(self, batch_dict):
        """
        Move rollout tensors into dataset structure expected by rl-games PPO loop.

        Base agent populates standard fields (obs, actions, values, logp, returns, etc.).
        AMPAgent additionally provides:
          - amp_obs: discriminator input from rollout (fake samples)
          - amp_obs_demo: discriminator input from motion demos (real samples)
          - amp_obs_replay: discriminator input from replay buffer (fake samples; older)
          - rand_action_mask: used to mask policy losses when using eps-greedy mixing
        """
        super().prepare_dataset(batch_dict)

        # AMP discriminator batches
        # shapes are [num_env * horizon_length, 1960]
        self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        # disc-shape-condition
        self.dataset.values_dict['amp_shape'] = batch_dict['amp_shape']
        self.dataset.values_dict['amp_shape_demo'] = batch_dict['amp_shape_demo']
        self.dataset.values_dict['amp_shape_replay'] = batch_dict['amp_shape_replay']

        # # [num_env * horizon_length, 1960]
        # print(self.dataset.values_dict['amp_obs'].shape)
        # print(self.dataset.values_dict['amp_obs_demo'].shape)
        # print(self.dataset.values_dict['amp_obs_replay'].shape)
        # # [num_env * horizon_length, 11]
        # print(self.dataset.values_dict['amp_shape'].shape)
        # print(self.dataset.values_dict['amp_shape_demo'].shape)
        # print(self.dataset.values_dict['amp_shape_replay'].shape)
        # exit()

        rand_action_mask = batch_dict['rand_action_mask']
        # Mask for eps-greedy losses
        self.dataset.values_dict['rand_action_mask'] = rand_action_mask
        return

    def train_epoch(self):
        """
        One full training epoch:
          1) rollout
          2) sample demo + replay discriminator batches
          3) run PPO/discriminator updates for mini_epochs_num passes
          4) push rollout amp_obs into replay buffer
          5) log timings and diagnostics
        """
        play_time_start = time.time()

        # Rollout without gradient tracking.
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps() 

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        # Refresh demo buffer (real motion samples) and attach a demo minibatch.
        # print(batch_dict.keys())
        # dict_keys(['actions', 'neglogpacs', 'values', 'mus', 'sigmas', 'obses', 'dones', 'next_obses', 'amp_obs', 'amp_shape', 'rand_action_mask', 'returns', 'played_frames', 'disc_rewards', 'task_reward'])

        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0]

        # when sampling from `self._amp_obs_demo_buffer` , 
        # we make sure each of the motion has same gender-beta as `amp_obs`
        # [num_env * horizon_length, 1960]
        demo_sample = self._amp_obs_demo_buffer.sample(num_obs_samples)
        batch_dict['amp_obs_demo'] = demo_sample['amp_obs']
        batch_dict['amp_shape_demo'] = demo_sample['amp_shape']

        # Replay buffer provides older agent samples; if empty, fallback to current rollout.
        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
            batch_dict['amp_shape_replay'] = batch_dict['amp_shape']
        else:
            # batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']
            replay_sample = self._amp_replay_buffer.sample(num_obs_samples)
            batch_dict['amp_obs_replay'] = replay_sample['amp_obs']
            batch_dict['amp_shape_replay'] = replay_sample['amp_shape']

        # # [num_env * horizon_length, 1960] for obs; [num_env * horizon_length, 1960] for shape
        # print(batch_dict['amp_obs'].shape)
        # print(batch_dict['amp_shape'].shape)
        # print(batch_dict['amp_obs_demo'].shape)
        # print(batch_dict['amp_shape_demo'].shape)
        # print(batch_dict['amp_obs_replay'].shape)
        # print(batch_dict['amp_shape_replay'].shape)
        # exit()

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        # Optional debug: fraction of valid frames for RNN masking.
        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        # PPO mini-epochs over the prepared dataset.
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])
                
                # rl-games learning-rate scheduling based on KL
                if self.schedule_type == 'legacy':  
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                # Aggregate train_info across minibatches for logging.
                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)
            # "standard" scheduling updates once per mini-epoch based on mean KL.
            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        # Some rl-games configs use "standard_epoch": update schedule once per epoch.
        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        # Some rl-games configs use "standard_epoch": update schedule once per epoch.
        # disc-shape-condition
        self._store_replay_amp_obs(batch_dict['amp_obs'], batch_dict['amp_shape'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time

        # Allow base class to record additional batch info (and this class adds disc_rewards in override).
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        """
        Compute gradients for actor-critic + discriminator (AMP) and apply optimizer step.

        This method is invoked by train_actor_critic() in the base rl-games agent.
        Expected fields in input_dict include:
          - PPO fields: old_values, old_logp_actions, advantages, returns, actions, obs, mu, sigma, ...
          - AMP fields: amp_obs (current rollout), amp_obs_replay, amp_obs_demo
          - rand_action_mask: masks out deterministic-action envs for policy losses
        """
        self.set_train()

        # PPO standard tensors.
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']

        obs_batch = input_dict['obs']
        # Preprocess obs for actor-critic.
        obs_batch = self._preproc_obs(obs_batch)

        # AMP discriminator batches.
        # - amp_obs: current rollout (fake)
        # - amp_obs_replay: replay buffer (fake, older)
        # - amp_obs_demo: demo buffer (real), with grad enabled for gradient penalty
        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        # disc-shape-condition
        amp_shape = input_dict['amp_shape'][0:self._amp_minibatch_size]
        amp_shape_demo = input_dict['amp_shape_demo'][0:self._amp_minibatch_size]
        amp_shape_replay = input_dict['amp_shape_replay'][0:self._amp_minibatch_size]

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)
        # Eps-greedy mask: only env-steps where rand_action_mask==1 contribute to policy losses.
        rand_action_mask = input_dict['rand_action_mask']
        rand_action_sum = torch.sum(rand_action_mask)

        lr = self.last_lr
        kl = 1.0
        # PPO clipping coefficient (potentially scaled by schedule).
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        # Package inputs for the model forward.
        # The model is expected to emit both actor-critic outputs and discriminator logits.
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
            'amp_obs' : amp_obs,
            'amp_shape': amp_shape,
            
            'amp_obs_replay' : amp_obs_replay,
            'amp_shape_replay': amp_shape_replay,
            
            'amp_obs_demo' : amp_obs_demo,
            'amp_shape_demo': amp_shape_demo,
        }

        # RNN-specific fields (if applicable).
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # Discriminator logits:
            #   disc_agent_logit: logits for current rollout samples (fake)
            #   disc_agent_replay_logit: logits for replay samples (fake)
            #   disc_demo_logit: logits for demo samples (real)
            disc_agent_logit = res_dict['disc_agent_logit']
            disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
            disc_demo_logit = res_dict['disc_demo_logit']

            # PPO actor loss (clipped surrogate).
            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info['actor_loss']
            a_clipped = a_info['actor_clipped'].float()

            # PPO critic loss (value regression, optionally clipped).
            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info['critic_loss']
            # Action bounds penalty (keeps mean actions in a reasonable range).
            b_loss = self.bound_loss(mu)
            # Reduce losses; policy-side losses are masked by rand_action_mask.
            c_loss = torch.mean(c_loss)
            a_loss = torch.sum(rand_action_mask * a_loss) / rand_action_sum
            entropy = torch.sum(rand_action_mask * entropy) / rand_action_sum
            b_loss = torch.sum(rand_action_mask * b_loss) / rand_action_sum
            a_clip_frac = torch.sum(rand_action_mask * a_clipped) / rand_action_sum
            # Discriminator loss computed over:
            #   fake logits = concat(current rollout, replay)
            #   real logits = demo
            disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
            disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
            disc_loss = disc_info['disc_loss']
            
            # Total loss:
            #   PPO objective + discriminator objective
            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                 + self._disc_coef * disc_loss
            
            # Store reduced losses back into dicts for logging.
            a_info['actor_loss'] = a_loss
            a_info['actor_clip_frac'] = a_clip_frac
            c_info['critic_loss'] = c_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None
        # Backprop with AMP scaler if mixed precision is enabled.
        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of the year
        # Gradient clipping + optimizer step.
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()    
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        # Compute policy KL (used for LR scheduling/logging).
        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask
        
        # Pack training results for logging.
        self.train_result = {
            'entropy': entropy,
            'kl': kl_dist,
            'last_lr': self.last_lr, 
            'lr_mul': lr_mul, 
            'b_loss': b_loss
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        self.train_result.update(disc_info)

        return

    def _load_config_params(self, config):
        """
        Parse AMP-specific hyperparameters from config.

        enable_eps_greedy:
            If True, rollouts mix deterministic mu-actions with stochastic sampled actions.
            Motivation: reduce jitter in some trajectories, improving discriminator training signal.
        """
        super()._load_config_params(config)
        
        # when eps greedy is enabled, rollouts will be generated using a mixture of
        # a deterministic and stochastic actions. The deterministic actions help to
        # produce smoother, less noisy, motions that can be used to train a better
        # discriminator. If the discriminator is only trained with jittery motions
        # from noisy actions, it can learn to phone in on the jitteriness to
        # differential between real and fake samples.
        self._enable_eps_greedy = bool(config['enable_eps_greedy'])

        # Reward mixing weights:
        # combined_reward = task_reward_w * task_reward + disc_reward_w * disc_reward
        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        # AMP observation space (provided by env_info, typically set by the task).
        self._amp_observation_space = self.env_info['amp_observation_space']

        # AMP minibatch controls for discriminator training.
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert(self._amp_minibatch_size <= self.minibatch_size)

        # Discriminator loss hyperparameters.
        
        # weight of disc_loss in total loss
        self._disc_coef = config['disc_coef']
        # L2 penalty on final logit layer
        self._disc_logit_reg = config['disc_logit_reg']
        # gradient penalty on demo samples
        self._disc_grad_penalty = config['disc_grad_penalty']
        # L2 penalty on discriminator weights
        self._disc_weight_decay = config['disc_weight_decay']

        # Discriminator reward scale.
        self._disc_reward_scale = config['disc_reward_scale']
        # Normalize amp_obs inputs to discriminator.
        self._normalize_amp_input = config.get('normalize_amp_input', True)

        # ---- transfer learning ----
        self._pretrained_ckpt = config.get('pretrained_ckpt')
        self._pretrained_loaded = False
        self._pretrained_raw_ckpt = None
        self._pretrained_model_state = None
        # ---- transfer learning ----
        return

    def _build_net_config(self):
        """
        Extend base network config by specifying discriminator input shape.
        The model builder uses this to construct the discriminator MLP.
        """
        config = super()._build_net_config()
        config['amp_input_shape'] = self._amp_observation_space.shape
        return config
    
    def _build_rand_action_probs(self):
        """
        Construct per-environment probability of taking a stochastic action.

        The schedule is a monotonic function of env id:
          env 0: prob = 1.0 (always stochastic)
          env N-1: prob = 0.0 (always deterministic)
        and smoothly varies in between.

        If eps-greedy is disabled, we force prob=1.0 for all envs (fully stochastic).
        """
        num_envs = self.vec_env.env.task.num_envs
        env_ids = to_torch(np.arange(num_envs), dtype=torch.float32, device=self.ppo_device)

        self._rand_action_probs = 1.0 - torch.exp(10 * (env_ids / (num_envs - 1.0) - 1.0))
        self._rand_action_probs[0] = 1.0
        self._rand_action_probs[-1] = 0.0
        
        if not self._enable_eps_greedy:
            self._rand_action_probs[:] = 1.0

        return

    def _load_pretrained_checkpoint(self):
        """
        Load selected weights from an old PHC checkpoint into the current model.

        # load transfer learning model
        # the phc_3 model has obs 934=358+576
        # 358 = PHC’s self observation for SMPL with max-coordinate observations enabled:
        # 1+∣body_names∣×(3+6+3+3)−3
        # 576 = PHC’s task / imitation observation under env_im_pnn
        # ∣track_bodies∣×num_traj_samples×24 = 24×1×24=576.
        
        Manual mapping:
        - old actor: a2c_network.pnn.actors.0.*
        - new actor: a2c_network.actor_mlp.* and a2c_network.mu.*
        - critic/disc/sigma: same names, load directly

        New conditioning layers are intentionally left randomly initialized:
        - cond_mlp, cond_linear
        - critic_cond_mlp, critic_cond_linear
        - disc_cond_mlp, disc_cond_linear
        """
        if self._pretrained_loaded:
            return

        ckpt_path = self._pretrained_ckpt
        if ckpt_path is None or ckpt_path == "":
            return

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

        print(f"[PHC] Loading pretrained checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.ppo_device)

        if "model" not in ckpt:
            raise KeyError(f"Checkpoint does not contain 'model': {ckpt_path}")

        src_state = ckpt["model"]
        dst_state = self.model.state_dict()

        # ---------------------------------------------------------
        # 1) explicit mapping: old key -> new key
        # ---------------------------------------------------------
        key_map = {
            # direct load
            "a2c_network.sigma": "a2c_network.sigma",

            "a2c_network.critic_mlp.0.weight": "a2c_network.critic_mlp.0.weight",
            "a2c_network.critic_mlp.0.bias":   "a2c_network.critic_mlp.0.bias",
            "a2c_network.critic_mlp.2.weight": "a2c_network.critic_mlp.2.weight",
            "a2c_network.critic_mlp.2.bias":   "a2c_network.critic_mlp.2.bias",
            "a2c_network.critic_mlp.4.weight": "a2c_network.critic_mlp.4.weight",
            "a2c_network.critic_mlp.4.bias":   "a2c_network.critic_mlp.4.bias",
            "a2c_network.critic_mlp.6.weight": "a2c_network.critic_mlp.6.weight",
            "a2c_network.critic_mlp.6.bias":   "a2c_network.critic_mlp.6.bias",
            "a2c_network.critic_mlp.8.weight": "a2c_network.critic_mlp.8.weight",
            "a2c_network.critic_mlp.8.bias":   "a2c_network.critic_mlp.8.bias",
            "a2c_network.critic_mlp.10.weight":"a2c_network.critic_mlp.10.weight",
            "a2c_network.critic_mlp.10.bias":  "a2c_network.critic_mlp.10.bias",

            "a2c_network.value.weight": "a2c_network.value.weight",
            "a2c_network.value.bias":   "a2c_network.value.bias",

            "a2c_network._disc_mlp.0.weight": "a2c_network._disc_mlp.0.weight",
            "a2c_network._disc_mlp.0.bias":   "a2c_network._disc_mlp.0.bias",
            "a2c_network._disc_mlp.2.weight": "a2c_network._disc_mlp.2.weight",
            "a2c_network._disc_mlp.2.bias":   "a2c_network._disc_mlp.2.bias",
            "a2c_network._disc_logits.weight":"a2c_network._disc_logits.weight",
            "a2c_network._disc_logits.bias":  "a2c_network._disc_logits.bias",

            # actor remap: primitive 0 -> current actor trunk/head
            "a2c_network.pnn.actors.0.0.weight":  "a2c_network.actor_mlp.0.weight",
            "a2c_network.pnn.actors.0.0.bias":    "a2c_network.actor_mlp.0.bias",
            "a2c_network.pnn.actors.0.2.weight":  "a2c_network.actor_mlp.2.weight",
            "a2c_network.pnn.actors.0.2.bias":    "a2c_network.actor_mlp.2.bias",
            "a2c_network.pnn.actors.0.4.weight":  "a2c_network.actor_mlp.4.weight",
            "a2c_network.pnn.actors.0.4.bias":    "a2c_network.actor_mlp.4.bias",
            "a2c_network.pnn.actors.0.6.weight":  "a2c_network.actor_mlp.6.weight",
            "a2c_network.pnn.actors.0.6.bias":    "a2c_network.actor_mlp.6.bias",
            "a2c_network.pnn.actors.0.8.weight":  "a2c_network.actor_mlp.8.weight",
            "a2c_network.pnn.actors.0.8.bias":    "a2c_network.actor_mlp.8.bias",
            "a2c_network.pnn.actors.0.10.weight": "a2c_network.actor_mlp.10.weight",
            "a2c_network.pnn.actors.0.10.bias":   "a2c_network.actor_mlp.10.bias",
            "a2c_network.pnn.actors.0.12.weight": "a2c_network.mu.weight",
            "a2c_network.pnn.actors.0.12.bias":   "a2c_network.mu.bias",
        }

        # ---------------------------------------------------------
        # 2) load according to mapping, with checks
        # ---------------------------------------------------------
        loaded_pairs = []
        skipped_pairs = []

        for src_key, dst_key in key_map.items():
            if src_key not in src_state:
                skipped_pairs.append((src_key, dst_key, "missing_src"))
                continue

            if dst_key not in dst_state:
                skipped_pairs.append((src_key, dst_key, "missing_dst"))
                continue

            if src_state[src_key].shape != dst_state[dst_key].shape:
                skipped_pairs.append((
                    src_key, dst_key,
                    f"shape_mismatch src={tuple(src_state[src_key].shape)} dst={tuple(dst_state[dst_key].shape)}"
                ))
                continue

            dst_state[dst_key] = src_state[src_key].clone()
            loaded_pairs.append((src_key, dst_key, tuple(src_state[src_key].shape)))

        self.model.load_state_dict(dst_state, strict=False)

        self._load_pretrained_stats(ckpt)

        self._pretrained_raw_ckpt = ckpt
        self._pretrained_model_state = src_state
        self._pretrained_loaded = True

        print(f"[PHC] Manually loaded param pairs: {len(loaded_pairs)}")
        for src_key, dst_key, shape in loaded_pairs:
            print(f"[PHC] loaded  {src_key}  -->  {dst_key}  {shape}")

        if len(skipped_pairs) > 0:
            print(f"[PHC] Skipped param pairs: {len(skipped_pairs)}")
            for src_key, dst_key, reason in skipped_pairs:
                print(f"[PHC] skipped {src_key}  -->  {dst_key}  ({reason})")

        return

    def _try_expand_and_load_running_mean_std(self, module, src_state, module_name="running_mean_std"):
        """
        Special-case loader for obs RunningMeanStd when current obs dim > old obs dim.

        Strategy:
        - copy old prefix stats into the new tensors
        - leave new extra dims at identity normalization:
            mean = 0
            var  = 1
        - copy scalar counters exactly
        """
        if module is None or src_state is None:
            print(f"[PHC][stats] skip {module_name}: module or src_state is None")
            return False

        dst_state = module.state_dict()

        # expected vector stats
        required_vec_keys = ["running_mean", "running_var"]
        for k in required_vec_keys:
            if k not in dst_state or k not in src_state:
                print(f"[PHC][stats] skip {module_name}: missing key '{k}'")
                return False

        dst_mean = dst_state["running_mean"]
        dst_var  = dst_state["running_var"]
        src_mean = src_state["running_mean"]
        src_var  = src_state["running_var"]

        if src_mean.ndim != 1 or dst_mean.ndim != 1:
            print(f"[PHC][stats] skip {module_name}: running_mean is not 1D")
            return False

        if src_var.ndim != 1 or dst_var.ndim != 1:
            print(f"[PHC][stats] skip {module_name}: running_var is not 1D")
            return False

        src_dim = src_mean.shape[0]
        dst_dim = dst_mean.shape[0]

        if src_dim > dst_dim:
            print(f"[PHC][stats] skip {module_name}: src_dim={src_dim} > dst_dim={dst_dim}")
            return False

        # start from current dst state, then overwrite prefix
        load_state = {}
        for k, dst_v in dst_state.items():
            if torch.is_tensor(dst_v):
                load_state[k] = dst_v.clone()
            else:
                load_state[k] = dst_v

        # identity normalization for all dims first
        load_state["running_mean"].zero_()
        load_state["running_var"].fill_(1.0)

        # copy old prefix
        load_state["running_mean"][:src_dim] = src_mean.to(device=dst_mean.device, dtype=dst_mean.dtype)
        load_state["running_var"][:src_dim]  = src_var.to(device=dst_var.device, dtype=dst_var.dtype)

        # copy any scalar/non-vector bookkeeping if shape matches
        for k, dst_v in dst_state.items():
            if k in ["running_mean", "running_var"]:
                continue
            if k not in src_state:
                continue

            src_v = src_state[k]
            if torch.is_tensor(dst_v) and torch.is_tensor(src_v):
                if tuple(dst_v.shape) == tuple(src_v.shape):
                    load_state[k] = src_v.to(device=dst_v.device, dtype=dst_v.dtype)
            else:
                load_state[k] = src_v

        module.load_state_dict(load_state, strict=False)

        print(
            f"[PHC][stats] expanded {module_name}: copied prefix {src_dim} -> {dst_dim}, "
            f"new tail {dst_dim - src_dim} dims set to mean=0 var=1"
        )
        return True

    def _try_load_stats_module(self, module, src_state, module_name):
        """
        Safely load a normalizer/state module only if the state dict is fully compatible.

        Rule:
        - all destination keys must exist in source
        - tensor shapes must match exactly
        - then load the whole module state
        - otherwise skip entirely
        """
        if module is None:
            print(f"[PHC][stats] skip {module_name}: module is None")
            return False

        if src_state is None:
            print(f"[PHC][stats] skip {module_name}: checkpoint state is None")
            return False

        dst_state = module.state_dict()

        for k, dst_v in dst_state.items():
            if k not in src_state:
                print(f"[PHC][stats] skip {module_name}: missing key '{k}'")
                return False

            src_v = src_state[k]

            if torch.is_tensor(dst_v) != torch.is_tensor(src_v):
                print(f"[PHC][stats] skip {module_name}: tensor/non-tensor mismatch at '{k}'")
                return False

            if torch.is_tensor(dst_v):
                if tuple(dst_v.shape) != tuple(src_v.shape):
                    print(
                        f"[PHC][stats] skip {module_name}: shape mismatch at '{k}', "
                        f"src={tuple(src_v.shape)} dst={tuple(dst_v.shape)}"
                    )
                    return False

        # copy destination-shaped state only
        load_state = {}
        for k, dst_v in dst_state.items():
            src_v = src_state[k]
            if torch.is_tensor(dst_v):
                load_state[k] = src_v.to(device=dst_v.device, dtype=dst_v.dtype)
            else:
                load_state[k] = src_v

        module.load_state_dict(load_state, strict=True)
        print(f"[PHC][stats] loaded {module_name}")
        return True


    def _load_pretrained_stats(self, ckpt):
        # 1) obs normalizer
        if "running_mean_std" in ckpt and hasattr(self, "running_mean_std"):
            # fallback: allow 934 -> 945 style expansion
            self._try_expand_and_load_running_mean_std(
                self.running_mean_std,
                ckpt["running_mean_std"],
                "running_mean_std",
            )
        else:
            print("[PHC][stats] skip running_mean_std: missing in ckpt or agent")

        # 2) AMP input normalizer
        if self._normalize_amp_input and hasattr(self, "_amp_input_mean_std"):
            if "amp_input_mean_std" in ckpt:
                self._try_load_stats_module(
                    self._amp_input_mean_std,
                    ckpt["amp_input_mean_std"],
                    "amp_input_mean_std",
                )
            else:
                print("[PHC][stats] skip amp_input_mean_std: missing in ckpt")
        else:
            print("[PHC][stats] skip amp_input_mean_std: not enabled in current run")

        # 3) optional value normalizer
        if "value_mean_std" in ckpt and hasattr(self, "value_mean_std"):
            self._try_load_stats_module(
                self.value_mean_std,
                ckpt["value_mean_std"],
                "value_mean_std",
            )

        # 4) intentionally skip reward normalizer
        if "reward_mean_std" in ckpt:
            print("[PHC][stats] skip reward_mean_std intentionally")

    def _init_train(self):
        """
        Called once before training begins.
        We populate the demo replay buffer so discriminator immediately sees "real" samples.
        """
        super()._init_train()

        if self._pretrained_ckpt:
            self._load_pretrained_checkpoint()

        self._init_amp_demo_buf()
        return

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        """
        Discriminator objective with regularizers.

        Inputs:
            disc_agent_logit: logits for fake samples (agent rollout + replay)
            disc_demo_logit: logits for real samples (demo)
            obs_demo: demo observations with requires_grad=True for gradient penalty

        Loss components:
          - BCE(fake, 0) and BCE(real, 1)
          - logit regularization (L2 on final logit layer weights)
          - gradient penalty on demo samples
          - weight decay on discriminator weights
        """
        # prediction loss (classification loss)
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit regularization (typically last layer weights)
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty  ||∇_x D(x)||^2 on real (demo) samples
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
                                             create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay over discriminator weights (optional)
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay
        # simple accuracy indicators for monitoring
        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info

    def _disc_loss_neg(self, disc_logits):
        """BCE loss that encourages discriminator to output 0 for fake samples."""
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss
    
    def _disc_loss_pos(self, disc_logits):
        """BCE loss that encourages discriminator to output 1 for real samples."""
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        """
        Cheap proxy accuracies:
          - fake correct if logit < 0 (sigmoid < 0.5)
          - real correct if logit > 0 (sigmoid > 0.5)
        """
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self):
        """
        Fetch real AMP observations from the environment's demo provider.
        Typically this samples from a motion dataset (e.g., AMASS clips).
        """
        # amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo()
        return amp_obs_demo

    def _build_amp_buffers(self):
        """
        Extend experience_buffer with AMP-specific tensors and allocate replay buffers.

        Experience buffer additions:
          - amp_obs: (T, N, amp_obs_dim) discriminator input from rollout
          - rand_action_mask: (T, N) mask for eps-greedy policy loss masking

        Replay buffers:
          - _amp_obs_demo_buffer: holds real demo samples
          - _amp_replay_buffer: holds older fake samples from the agent
        """
        batch_shape = self.experience_buffer.obs_base_shape
        
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape,
                                                                    device=self.ppo_device)
        self.experience_buffer.tensor_dict['rand_action_mask'] = torch.zeros(batch_shape, dtype=torch.float32, device=self.ppo_device)
        # disc-shape-condition
        self.experience_buffer.tensor_dict['amp_shape'] = torch.zeros(batch_shape + (11,), device=self.ppo_device)

        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)
        
        self._build_rand_action_probs()
        # disc-shape-condition
        self.tensor_list += ['amp_obs', 'amp_shape', 'rand_action_mask']
        return

    def _init_amp_demo_buf(self):
        """
        Pre-fill the demo buffer with real samples.
        This avoids a cold-start where discriminator sees no real data.
        
        # disc-shape-condition
        """

        # we have the beta_env information already, maybe just use them
        # [num_env, 1960]
        # print(self.vec_env.env.task._betas_env.shape)
        amp_obs_demo, amp_shape = self._fetch_amp_obs_demo()

        # both [num_env * horizon_length, 1960]
        amp_obs_demo_flat = self.expand_env_tensor_to_horizon_flat(amp_obs_demo)
        amp_shape_flat = self.expand_env_tensor_to_horizon_flat(amp_shape)

        self._amp_obs_demo_buffer.store({'amp_obs': amp_obs_demo_flat, 'amp_shape': amp_shape_flat})

        # self._amp_obs_demo_buffer.store({'amp_obs': amp_obs_demo})

        # buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        # num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        # for i in range(num_batches):
        #     curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
        #     self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return
    
    def _update_amp_demos(self):
        """
        Add a fresh batch of real demo samples each epoch.
        Maintains diversity and keeps demo distribution current if dataset sampling is dynamic.

        # disc-shape-condition
        """
        # [amp_batch_size, 1960]
        # new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        new_amp_obs_demo, new_amp_shape = self._fetch_amp_obs_demo()

        # both [num_env * horizon_length, 1960]
        amp_obs_demo_flat = self.expand_env_tensor_to_horizon_flat(new_amp_obs_demo)
        new_amp_shape_flat = self.expand_env_tensor_to_horizon_flat(new_amp_shape)

        # self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        self._amp_obs_demo_buffer.store({'amp_obs': amp_obs_demo_flat, 'amp_shape': new_amp_shape_flat})
        return

    def _preproc_amp_obs(self, amp_obs):
        """
        Preprocess discriminator input.

        - First, sanitize non-finite values (debug safety).
        - Then optionally apply RunningMeanStd normalization.
        """

        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        """
        Combine environment task reward and discriminator reward.

        This is where you control the trade-off:
          - task_reward encourages tracking / control objectives from env
          - disc_reward encourages matching the demo motion distribution
        """
        disc_r = amp_rewards['disc_rewards']
        
        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs, amp_shape):
        """
        Forward discriminator on preprocessed amp_obs.

        amp_obs: [horizon_length, num_envs, get_num_amp_obs]

        root_h_obs,
        root_rot_obs,
        local_root_vel,
        local_root_ang_vel,
        dof_obs,
        dof_vel,
        flat_local_key_pos

        root height
        root rotation observation
        local root linear velocity
        local root angular velocity
        DOF pose observation
        DOF velocity
        local key-body positions
        """

        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs, amp_shape)
    
    def _calc_advs(self, batch_dict):
        """
        Compute advantages with masking for eps-greedy.

        Here, advantages are computed as returns - values and then optionally normalized
        *only over frames where rand_action_mask == 1*.
        This matches masking in calc_gradients(), so actor updates align with the data used.
        """
        returns = batch_dict['returns']
        values = batch_dict['values']
        rand_action_mask = batch_dict['rand_action_mask']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)
        if self.normalize_advantage:
            advantages = torch_ext.normalization_with_masks(advantages, rand_action_mask)

        return advantages

    def _calc_amp_rewards(self, amp_obs, amp_shape):
        """
        Compute all AMP-related rewards to be added into the RL reward.

        Currently only includes discriminator reward, but the dict structure allows extensions.
        """
        disc_r = self._calc_disc_rewards(amp_obs, amp_shape)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs, amp_shape):
        """
        Convert discriminator logits into a shaped reward.

        discriminator learns a real-vs-fake motion classifier for AMP

        Standard AMP shaping:
            prob = sigmoid(logit)
            r = -log(max(1 - prob, eps))
        so higher "realness" yields higher reward.
        """
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs, amp_shape)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale

        return disc_r

    def _store_replay_amp_obs(self, amp_obs, amp_shape):
        """
        # disc-shape-condition

        Store agent-generated AMP observations into replay buffer.

        Replay buffer purpose:
            discriminator negatives should include both current rollout samples and older samples
            to stabilize discriminator training and reduce oscillation.

        This method also implements:
          - filtering non-finite rows
          - downsampling when buffer is over capacity using keep_prob
          - random subsampling if batch is still too large
        """

        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]
            # disc-shape-condition
            amp_shape = amp_shape[keep_mask]

        if (amp_obs.shape[0] > buf_size):
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]
            amp_shape = amp_shape[rand_idx]

        self._amp_replay_buffer.store({'amp_obs': amp_obs, 'amp_shape': amp_shape,})
        return

    
    def _record_train_batch_info(self, batch_dict, train_info):
        """
        Extend base batch-info recording: attach disc_rewards for logging convenience.
        """
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        train_info['task_reward'] = batch_dict['task_reward']

        return
    
    def _log_train_info(self, train_info, frame):
        # Call super if needed (though base is stub)
        super()._log_train_info(train_info, frame)

        # Compute and log FPS (previously in train())
        if 'curr_frames' in train_info and 'scaled_time' in train_info:
            total_fps = train_info['curr_frames'] / train_info['scaled_time']
            step_fps = train_info['curr_frames'] / train_info['scaled_play_time']
            # Log to TensorBoard (if still using) or WandB
            # self.writer.add_scalar('performance/total_fps', total_fps, frame)  # Uncomment if keeping TB
            # self.writer.add_scalar('performance/step_fps', step_fps, frame)
            wandb_logger.log({'performance/total_fps': total_fps, 'performance/step_fps': step_fps}, step=frame)

        # Log epochs
        if 'epoch_num' in train_info:
            # self.writer.add_scalar('info/epochs', train_info['epoch_num'], frame)
            wandb_logger.log({'info/epochs': train_info['epoch_num']}, step=frame)

        # Log rewards and lengths if available
        if 'mean_rewards' in train_info and 'mean_lengths' in train_info:
            mean_rewards = train_info['mean_rewards']
            mean_lengths = train_info['mean_lengths']

            for i in range(self.value_size):
                wandb_logger.log({
                    f'rewards{i}/frame': mean_rewards[i],
                    f'rewards{i}/iter': mean_rewards[i],
                    f'rewards{i}/time': mean_rewards[i]
                }, step=frame)

            # self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
            # self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
            wandb_logger.log({
                'episode_lengths/frame': mean_lengths,
                'episode_lengths/iter': mean_lengths
            }, step=frame)

        # Add your detailed reward breakdowns here (from previous suggestion)
        # Assuming you added 'mean_reward_pos', etc., to train_info in play_steps/train_epoch
        if 'mean_reward_pos' in train_info:
            # self.writer.add_scalar('rewards/pos', train_info['mean_reward_pos'], frame)
            wandb_logger.log({
                'rewards/mean_reward_pos': train_info['mean_reward_pos'],
                'rewards/mean_reward_pos': train_info['mean_reward_pos'],
                'rewards/mean_reward_rot': train_info['mean_reward_rot'],
                'rewards/mean_reward_vel': train_info['mean_reward_vel'],
                'rewards/mean_reward_ang_vel': train_info['mean_reward_ang_vel'],
                'rewards/mean_reward_power': train_info['mean_reward_power'],
                'rewards/mean_amp_reward': train_info['disc_rewards'].mean().item(),
                'rewards/mean_task_reward': train_info['task_reward'],
            }, step=frame)

        # # Custom console print (moved here)
        # if 'mean_rewards' in train_info and 'mean_lengths' in train_info:
        #     log_parts = [f'fps step: {step_fps:.1f} fps total: {total_fps:.1f}']
        #     for i in range(self.value_size):
        #         log_parts.append(
        #             f"rewards{i}/frame={mean_rewards[i]:.4f}, rewards{i}/iter={mean_rewards[i]:.4f}, rewards{i}/time={mean_rewards[i]:.4f}"
        #         )
        #     log_parts.append(f"episode_lengths/frame={mean_lengths:.4f}")
        #     log_parts.append(f"episode_lengths/iter={mean_lengths:.4f}")
        #     log_str = " | ".join(log_parts)
        #     print(log_str)

        # Add any AMP/PHC-specific logging (e.g., disc_loss, amp_rewards) from train_info
        # Example: if 'disc_loss' in train_info: wandb.log({'losses/disc_loss': train_info['disc_loss']}, step=frame)


    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            # disc-shape-condition
            amp_shape = info['amp_shape']
            amp_shape = amp_shape[0:1]
            disc_pred = self._eval_disc(amp_obs, amp_shape)
            amp_rewards = self._calc_amp_rewards(amp_obs, amp_shape)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward)
        return

    def expand_env_tensor_to_horizon_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_env, n]
            horizon_length: rollout horizon length

        Returns:
            [num_env * horizon_length, n]
        """

        # [1, num_env, n] -> [horizon_length, num_env, n]
        x_seq = x.unsqueeze(0).expand(self.horizon_length, -1, -1)

        # same flattening logic used by rl-games
        return a2c_common.swap_and_flatten01(x_seq)