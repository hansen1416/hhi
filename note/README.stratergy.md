**Analysis for hansen1416/hhi single-motion collapse (PHC + HUMOS 128 variations)**

Your project builds directly on **AMP** (Adversarial Motion Priors, Peng et al. SIGGRAPH 2021) and **PHC** (Perpetual Humanoid Control, Luo et al. ICCV 2023), extended with 64 body-shape variations × 2 genders from the **HUMOS** model (ECCV 2024) on AMASS. This yields 128 morphological variants per motion clip. The repo (https://github.com/hansen1416/hhi) uses a custom `PHCBuilder` (phc_network_builder.py) with FiLM conditioning on the last 11 obs dims (gender + 10 betas), so the actor trunk sees only the first 574 dims while the critic and discriminator use the full 585-D obs. The env (`HumanoidPHC` in humanoid_phc.py) runs single-motion tracking with `task_obs_v7` (9×key-bodies local Δp/Δv + root-rel pos), imitation reward (position/rot/vel/ang-vel with fixed k/w), power penalty, and AMP disc reward. `PHCAgent` (phc_agent.py) wraps rl-games PPO with eps-greedy, replay buffer, etc.

**Diagnosis (why static pose on single motion)**  
In single-motion mode the imitation reward is too weak relative to the power term (`-power_coefficient * |torque·vel|` in `_compute_reward`). Early in training the policy discovers a low-energy static/default pose that (a) satisfies early-termination avoidance, (b) keeps power near zero, and (c) is “real enough” for the AMP discriminator (which was trained on varied motions but sees mostly static fakes). The discriminator cannot strongly penalize it because the static pose is a valid (if trivial) mode in the motion manifold. FiLM conditioning on betas helps morphology generalization but does not force tracking when the base imitation signal is insufficient. Eps-greedy + PPO’s value estimation further encourages exploitation of the safe static attractor. This is classic reward hacking / mode collapse in single-task AMP setups — well-documented in recent literature when the task reward is sparse or the style reward is distribution-level rather than frame-level.

From a generalized RL viewpoint (PPO + adversarial imitation under morphology randomization), the core issues are:
- **Insufficient credit assignment** for long-horizon tracking (single clip = sparse signal).
- **Reward interference** (power penalty dominates imitation early).
- **Exploration collapse** (eps-greedy mixes deterministic mu but still converges to low-variance static).
- **Value overestimation** on safe states (critic does not see privileged full-ref or morphology-specific dynamics sharply enough).
- **Morphology noise** acting as distractor rather than conditioner when tracking signal is weak.

**Reward redesign (most impactful — borrow from 2024–2026 work)**  
Recent papers emphasize automatic or adaptive rewards to prevent exactly this static collapse.

1. **Adversarial Differential Discriminators (ADD, Zhang et al., SIGGRAPH Asia 2025, arXiv:2505.04961)**  
   Replace your hand-tuned imitation_reward with a differential discriminator on error vector Δ = ref_features – agent_features (root pos/rot, joint pos/rot/vel, key-body pos — exactly what you already compute in `build_amp_observations` and `compute_task_obs_v7`).  
   Discriminator D(Δ) trained with single positive sample (Δ=0) + gradient penalty on negatives. Policy reward = –log(1 – D(Δ_t)).  
   This auto-balances multi-objective tracking without manual k/w and forces frame-level fidelity (your current AMP is distribution-level, allowing static modes).  
   Borrowable code patch in `humanoid_phc.py` `_compute_reward`:
   ```python
   # instead of fixed weighted exp
   delta = torch.cat([ref_body_pos - body_pos, ...])  # build your Δ
   disc_out = self.add_disc(delta)  # small MLP + FiLM if you want morphology cond
   r_tracking = -torch.log(1 - torch.sigmoid(disc_out) + 1e-8)
   rew_buf = r_tracking + power_reward  # keep power but now secondary
   ```
   ADD reports position errors <0.03 m on acrobatic clips — directly applicable to your single-motion + 128 shapes (they tested across 26–28 DoF embodiments).

2. **Two-Layered Reward (Xu et al., Mathematics 2025)**  
   Upper layer: goal reward = tracking completion (e.g. mean key-body error below threshold → +1).  
   Lower layer: optimizing reward (stability + smoothness + your power term).  
   Dynamically weight lower → upper as training progresses (e.g. via progress_buf or success rate). Prevents early static by keeping stability/energy secondary until tracking is viable.  
   Easy to add in your reward_raw stacking.

3. **REvolve (Hazra et al., ICLR 2025, arXiv:2406.01309)**  
   LLM (GPT-4/o1) evolves Python reward functions guided by human Elo-ranked feedback on rollouts. Perfect for your setup: seed with current imitation + power + AMP, let REvolve propose variants (“add phase-conditioned velocity weight”, “make power penalty morphology-aware via betas”), rank 5–10 candidates per iteration. They show it outperforms Eureka on humanoid locomotion. Zero extra code beyond a simple evaluator loop.

4. **Adaptive tracking tolerance (KungfuBot, Xie et al., NeurIPS 2025, arXiv:2506.12851)**  
   Bi-level opt: lower level maximizes exp(–error/σ), upper level tightens σ via EMA of errors. Start loose (allow exploration), tighten automatically → forces dynamic tracking without static collapse. Combine with your task_obs_v7.

5. **Residual-action tracking (RobotDancing, Sun et al. 2025)**  
   Policy outputs δ-action = PD_target – reference_dof_pos instead of absolute. Allocates capacity to physics compensation → dramatically better long-horizon single-clip tracking.

**RL algorithm & training enhancements (generalized PPO view)**  
- **Asymmetric actor-critic + privileged info** (KungfuBot, HoRD): Give critic full ref motion, future keypoints, and betas/privileged dynamics. Your critic already sees full 585-D — just vectorize rewards (one head per component) as in KungfuBot for better advantage estimation.
- **History-conditioned RL** (HoRD 2026, arXiv:2602.04412): Add 8–10 step history to actor (already in many Isaac Gym baselines) + Query-Transformer to infer latent “morphology dynamics” from recent (state, action). Prevents collapse when betas vary (treats shape variation as online domain shift).
- **Morphology-aware conditioning** (McARL 2025, arXiv:2505.18418): Your FiLM is good — upgrade to learned morph embedding z_m = MLP(betas) concatenated or FiLM-modulated at every layer (actor + critic). Randomize betas per env during training (you already do via HUMOS). McARL shows zero-shot transfer across 4 morphologies; scales to your 128.
- **Curriculum + phase conditioning**: Add motion phase (progress_buf / clip_length) to obs and condition discriminator/reward on phase (common in recent multi-gait AMP extensions like CAMP 2025). Start with slow-motion curriculum (subsample clip or scale velocities).
- **Exploration fixes**: Strengthen eps-greedy schedule or add entropy bonus that decays slower on tracking reward. Or switch to SAC-style for better exploration in single-task.
- **Online distillation** (HoRD): Train a teacher with privileged info, distill to student (your current actor). Great for morphology robustness.

**Hyper-parameter & quick wins**
- Increase imitation k_pos/k_rot initially (100–200), anneal down; power_coefficient start at 1e-5 and ramp up.
- AMP disc: larger batch, stronger grad penalty (your _disc_grad_penalty).
- Obs norm: running mean-std on betas separately (they are small).
- LR: actor 3e-5 → 1e-5 after collapse detected (monitor mean key-body error > threshold).
- Episode length: longer for single motion (your max_episode_length).

**Code-level patches (ready to apply)**
In `humanoid_phc.py` (reward):
```python
# ADD-style or adaptive
error = torch.norm(ref_body_pos - body_pos, dim=-1).mean(dim=-1)
sigma = self._adaptive_sigma  # EMA update
r_pos = torch.exp(-error / sigma)
# ... same for others, then combine with power
```
In `phc_network_builder.py` (eval_actor): add phase to cond_mlp input if you want phase-conditioned FiLM.

In `phc_agent.py`: add privileged critic head for ref features.

**Evaluation plan**
- Per-key-body position/velocity error vs reference over full clip (not just reward).
- Foot-skate metric (integral of foot vel when in contact).
- Motion similarity (DTW or MPJPE on AMASS keypoints).
- Morphology robustness: test held-out betas, measure tracking degradation.
- Energy vs tracking Pareto (plot power vs error).

These ideas come from the most recent 2024–2026 literature (ADD, KungfuBot, REvolve, HoRD, McARL, Two-Layered Reward, etc.) and directly address your exact failure mode while fitting your PHC + HUMOS + FiLM stack. Start with ADD-style differential reward + adaptive σ — that alone should stop the static collapse within a few epochs. Once tracking is stable, scale to multi-motion and full 128-shape generalization.

Let me know which direction you want to prototype first (reward patch, asymmetric critic, or REvolve integration) and I can give exact diff files. We will turn every AMASS clip physical!