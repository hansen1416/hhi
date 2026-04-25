# Preventing Static-Pose Collapse in Morphology-Conditioned AMP/PHC Humanoid RL

## Problem setting and code-grounded diagnosis

Your current stack (as implemented in the repo) is a hybrid of PHC-style per-frame reference tracking plus AMP-style adversarial priors, with explicit morphology conditioning in the policy. Concretely, the actor-critic is built via a custom HHIBuilder that **FiLM-conditions the actor** on the last 11 observation dimensions (“gender + 10 betas”), while feeding only the first 574 dimensions (state + task features) into the actor trunk. The critic still consumes the full observation. citeturn10view0

The environment `HumanoidHHI` constructs a **585-D observation** by concatenating a self-state block (computed from rigid-body positions/orientations/velocities and root height), a task block (`task_obs_v7`) built from **local root-frame ∆position/∆velocity and root-relative target position** over key bodies, and finally the 11-D shape condition appended at the end. citeturn9view0turn23view0turn19view0turn10view0

The reward in `HumanoidHHI` is computed as:

- an **imitation reward** that is a *weighted sum of exponentials* over global body position error, body rotation error, linear velocity error, and angular velocity error, using the tunables `k_pos=50, k_rot=30, k_vel=0.1, k_ang_vel=0.1` and weights `(0.5, 0.3, 0.1, 0.1)`; and  
- a **power penalty** of the form `-power_coefficient * sum(|torque * joint_vel|)` (skipping the first few frames), with `power_coefficient` coming from the env config. citeturn23view0turn9view0turn8view1

Separately, the PPO+AMP training loop in `HHIAgent` shapes a **discriminator reward** as  
\[
r_{\text{disc}} = -\log(\max(1-\sigma(\text{logit}), \varepsilon)) \cdot \text{disc\_reward\_scale},
\]
and then combines it with the environment task reward via  
\[
r_{\text{combined}} = w_{\text{task}}\,r_{\text{task}} + w_{\text{disc}}\,r_{\text{disc}}.
\]
This is exactly the AMP-style positive-only “realness” shaping (reward is near 0 when prob≈0, and grows as prob→1). citeturn24view0turn11view1turn8view0turn31view0

Your failure mode is therefore well aligned with the code path: **collapse to a static/default standing pose** (minimal joint velocities → near-zero power penalty) plus **early termination exploitation** (short episodes reduce exposure to future tracking errors), while the discriminator provides **no negative gradient pressure** against “non-motion-like” stasis beyond the absence of an extra positive bonus. citeturn23view0turn24view0turn8view1

## Why static collapse is a rational optimum under the current objective

### The power term can dominate the return scale

The imitation reward is bounded: each exponential term lies in \([0,1]\), and the weighted sum therefore lies in \([0,1]\). citeturn23view0  
In contrast, the power penalty is **unbounded below** in practice because it scales with the sum of \(|\tau \cdot \dot{q}|\) across DoFs, multiplied by `power_coefficient`. A standing/low-velocity policy makes power ≈ 0, so the penalty ≈ 0, even if it tracks poorly. citeturn9view0turn8view1turn23view0

This establishes a simple **scale mismatch**: unless tracking reward is strong enough *early in training* to compensate for the reduction in energy usage, the easiest way to improve return is to reduce joint motion and/or terminate early.

### Early termination plus value bootstrapping makes “quit early” attractive

Your termination logic flags failure when (i) non-foot contacts exceed a threshold and (ii) some body heights drop below a per-body termination height; this is then used as a “terminate” flag passed to the agent. citeturn23view0turn8view1  
In `HHIAgent`, the critic bootstrap is explicitly masked by `next_vals *= (1 - terminated)` to avoid bootstrapping through failure transitions. citeturn8view2

This masking is reasonable for stability, but it also means that if the agent can **trigger termination at will** after accruing only small penalties (or avoiding future penalties), the RL objective can prefer this behavior—especially when per-step rewards are small and dominated by a negative component.

More broadly, PPO optimizes expected discounted return with GAE-style advantage estimation; with poorly balanced reward scales and short-horizon rollouts, the optimizer will reliably find these “cheap” local optima. citeturn29search0turn29search1turn8view0

### The AMP reward is positive-only and does not penalize stasis

The discriminator shaping in `HHIAgent` is monotone nonnegative: if `prob` is small (fake), \(r_{\text{disc}}\approx 0\). It does not produce negative reward for “very fake” behaviors; it only withholds a bonus. citeturn24view0  
So a static policy can sit at a regime where:
- imitation reward is low-to-moderate (depending on how far reference motion is),  
- power penalty is near 0, and  
- discriminator reward is near 0,  
yielding a **stable training equilibrium** that PPO will not easily escape.

This is consistent with AMP’s original intent: AMP is designed to supply *style rewards* that complement (or partly replace) hand-crafted imitation objectives, not necessarily to act as a standalone anti-collapse penalty in the presence of strong energy regularization. citeturn31view0turn24view0

## Reward redesign to remove the exploit and increase tracking pressure

The highest-leverage fix is to change the **relative incentives** so that (a) standing still is no longer competitive, and (b) “ending the episode early” is strictly worse than continuing.

### Make the power term a constrained or scheduled objective, not a dominant penalty

**Recommended change:** start training with **no (or tiny) power regularization**, then ramp it up only after tracking is already stable. This mirrors a common recipe in high-dynamic tracking pipelines: first solve feasibility and tracking, then optimize efficiency. It is conceptually aligned with the “perfect first, then scale up” curriculum philosophy used in more recent imitation/control works. citeturn15search3turn15search2

Concretely in your code, this is easiest as a curriculum on `power_coefficient` (or on a separate coefficient applied after combining rewards). Right now, power enters directly in `HumanoidHHI._compute_reward` and is therefore inseparable from “task reward” at combination time. citeturn9view0turn23view0turn24view0  
If you keep it in-env, implement:

- **Warm-start:** `power_coefficient = 0` for the first N epochs.
- **Ramp:** linearly or sigmoidally increase to the final value over the next M epochs.
- **Clamp:** optionally cap per-step power penalty to \([-c, 0]\) so it cannot dominate the per-step return.

This is directly motivated by your bounded imitation reward and unbounded energy term. citeturn23view0turn8view1

### Add an explicit failure/termination penalty and/or survival bonus

Because termination currently merely truncates the return (and blocks bootstrap), you need an explicit term that punishes “giving up.” You can implement either:

- **terminal penalty:** if `terminated==1`, add a one-time negative reward \(-\lambda_{\text{fail}}\); or  
- **survival reward:** add a small constant \(+\lambda_{\text{alive}}\) each step the character remains non-terminated.

PHC’s broader “perpetual control” framing is partly about removing the dependency on resets and making recovery an explicit part of the control problem; that design choice also prevents the trivial solution of repeatedly failing/resetting when tracking is hard. citeturn30view0turn30view1  
If you do not want to fully switch to a recovery setting yet, a termination penalty is the minimal surgical fix that targets the same exploit.

### Replace fixed imitation sharpness with adaptive tracking tolerance

Your imitation reward uses fixed exponential kernels with relatively large `k_pos` and `k_rot`, which can yield very small gradients once the policy deviates appreciably from the reference (especially for rotations). citeturn23view0turn9view0  
Recent work targeting highly dynamic skills explicitly uses **adaptive tolerance / curriculum** mechanisms to prevent early collapse and progressively tighten tracking. KungfuBot formulates an adaptive tracking mechanism that “dynamically adjusts the tracking accuracy tolerance based on current tracking error,” functioning as an automatic curriculum. citeturn15search2turn15search18

A practical implementation in your setting:

- Define a current tracking error scalar \(e_t\) (e.g., the same MSE quantities already computed inside `compute_imitation_reward`). citeturn23view0  
- Make the kernel sharpness \(k\) a function of error or training progress:
  - **progress-based:** \(k(\text{epoch})\) ramps from small to large; or
  - **error-based:** \(k(e)=k_{\min}+ (k_{\max}-k_{\min})\cdot \sigma((e-\tau)/s)\).

This avoids “all reward goes to ~0” regimes and makes the policy’s early learning signal dense, while still enabling tight final tracking.

### Introduce residual action tracking to reduce drift and stabilize long horizons

RobotDancing (2025) reports robust multi-minute humanoid tracking by predicting **residual joint targets** rather than absolute commands, explicitly to compensate model mismatch and prevent error accumulation. citeturn15search0turn15search16  
In your case, with PD control enabled (`pdControl: True`), it is natural to define the policy output as a residual around the reference pose (or around the reference next-step change):

\[
q^{\text{target}}_t = q^{\text{ref}}_t + \Delta q^{\pi}_t, \quad \Delta q^{\pi}_t \in [-\delta, \delta].
\]

This changes the optimization geometry: “standing still” corresponds to a nonzero residual as the reference moves, so it becomes easier to punish, and the policy capacity is used for **physics compensation** rather than re-synthesizing the entire kinematic trajectory. This is precisely the mechanism RobotDancing emphasizes. citeturn15search16turn8view1

## Algorithmic stabilizers from recent imitation-control research

Several 2023–2025 systems that succeed at long-horizon tracking share a key trend: **they reduce reliance on sparse on-policy RL gradients early in training**, using staged learning, distillation, or progressive capacity.

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["AMP adversarial motion priors diagram","Perpetual Humanoid Control PMCP diagram","RobotDancing residual action humanoid motion tracking","KungfuBot adaptive motion tracking framework"],"num_per_query":1}

### Use a teacher–student pipeline to bootstrap tracking before full AMP training

InterMimic (CVPR 2025) explicitly argues for a curriculum of **“perfect first, then scale up”** by training teacher policies to mimic/refine imperfect MoCap, then distilling into a unified student, with RL fine-tuning afterwards. citeturn15search3turn15search7turn15search15  
Even though your immediate task is “single motion tracking,” the same idea applies cleanly:

- Train an **overfit teacher** on the single motion (and possibly per-morphology subsets) with a very tracking-heavy reward (no energy penalty; termination penalty enabled). PHC’s own project page explicitly shows that overfitting to one clip is possible as a diagnostic/control baseline. citeturn30view1  
- Distill the teacher into the morphology-conditioned student (FiLM policy) using supervised losses on actions or target joint positions, then re-enable PPO fine-tuning with energy regularization and AMP.

This addresses your core failure mode: the policy “never discovers” the moving solution because it collapses immediately. A teacher bypasses exploration problems.

### Progressive capacity and “hard-case” sampling as automatic curricula

PHC proposes PMCP, which “dynamically allocates new network capacity to learn harder and harder motion sequences,” explicitly to scale to large motion databases and add tasks without catastrophic forgetting. citeturn30view0turn30view1  
Your MotionLibHUMOS already contains scaffolding for **hard-case based sampling** (“Auto PMCP”) by tracking failures and biasing sampling towards failed sequences. citeturn34view2

For single-motion training, you can reinterpret “hard cases” as:

- the most difficult *time segments* (fast turns, foot contacts, large accelerations), and/or  
- the most difficult *morphologies* (extreme betas that create challenging inertia/limb-length combinations).

Sampling bias towards these segments after the policy becomes competent on easy segments is a direct analog of PMCP-style progression and will reduce the chance that training settles into a low-motion attractor.

### Distill motion priors into compact latents, then control through them

ASE (2022) and subsequent universal motion representation work explicitly learn reusable motion priors/embeddings for physics-based characters. citeturn12search0turn31view0  
More recently, “Universal Humanoid Motion Representations for Physics-Based Control” (PULSE, 2023/ICLR 2024) follows a two-stage pattern: train a large-scale imitator, then distill skills into a latent space used for downstream control, with a learned prior conditioned on proprioception. citeturn12search2turn13search28

If your end goal is “convert arbitrary kinematic AMASS motion into physical, morphology-aware motion,” this suggests a robust alternative to direct single-clip RL:

- Learn a *general imitator* (PHC-style) across many motions and morphologies, then  
- condition a decoder/policy on a **latent that represents the target motion slice**, rather than raw per-frame deltas alone.

Even if you keep your existing per-frame task_obs, adding a latent representation (learned via distillation) can stabilize long-horizon behavior by providing a smoother control manifold than direct tracking on noisy kinematic targets.

## Strengthening AMP against stasis and improving discriminator usefulness

### Condition the discriminator on morphology and/or reference phase

Your actor is explicitly morphology-conditioned via FiLM. citeturn10view0  
But your discriminator operates on **AMP observations that do not explicitly include shape condition**, and the AMP reward is therefore effectively matching a *marginal* motion distribution across all shapes and phases. citeturn23view0turn24view0turn9view0

Recent conditional adversarial skill learning (e.g., C·ASE, 2023) emphasizes conditioning adversarial objectives on control/skill variables to avoid mode averaging and to make the reward informative for the intended behavior. citeturn12search7turn12search3  
For your setting, the most direct mapping is:

- Condition \(D\) on morphology \(m\) (gender+betas, or a richer physical parameter vector), and  
- optionally condition on **phase / motion-time** (for single motion, phase is unambiguous and prevents “statistically plausible but wrong time” states).

This turns the discriminator into a **conditional critic**: “is this state realistic given this body and this phase,” which is far more effective at rejecting static standing when the reference says the body should be moving.

### Re-center discriminator reward so “fake” becomes actively bad

As implemented, AMP reward is \(r=-\log(1-D)\), which is always \(\ge 0\). citeturn24view0turn31view0  
If you want AMP to directly oppose static collapse, you need the policy to experience **negative advantage** for stasis relative to motion-like behavior. Two robust options:

- **mean-centered AMP reward:** \(r'_{\text{disc}} = r_{\text{disc}} - \mathbb{E}[r_{\text{disc}}]\) (moving baseline);  
- **logit reward:** use the discriminator logit (or \(\log D - \log(1-D)\)) as the reward term so that clearly fake behaviors become negative.

These modifications preserve the ordering of “more real is better,” but prevent the “do nothing and just receive ~0” equilibrium.

### Ensure discriminator negatives cover “static but stable” counterexamples

Your discriminator replay buffer stores agent-generated AMP observations and down-samples them with a keep probability once full. citeturn24view1turn8view0  
When the policy collapses early, the replay buffer can become dominated by “static standing” negatives; paradoxically, this can make it easier for the discriminator to learn to separate “standing” from “motion,” but (because the reward is nonnegative) that still may not generate sufficient gradient to move the policy.

A stronger tactic is to deliberately add **hard negatives**:

- Take real demo states and **time-shuffle** or **velocity-zero** them (pose correct, dynamics wrong) and force the discriminator to reject them.  
- Include “standing frames” only if the reference motion truly contains them; otherwise filter low-speed segments from the demo set so that the discriminator’s notion of “real” is motion-rich.

This is consistent with the goal of using adversarial learning to encode style/motion statistics rather than “any stable pose.”

## Morphology-aware specifics and a prioritized ablation roadmap

### Make morphology conditioning physically meaningful, not only kinematic

FiLM on SMPL betas is a reasonable first step, but several 2024–2025 morphology-conditioned control papers for legged robots emphasize that conditioning on **control-relevant physical parameters** (masses, lengths, torque limits, stiffness/damping) is more reliable than treating morphology as mere noise. McARL (2025) conditions both actor and critic on a morphology/control vector and reports improved transfer and reduced retuning across morphologies. citeturn33view0

Your pipeline already has access (or can easily derive) many of these quantities from the simulated bodies; consider augmenting the 11-D SMPL condition with:

- link lengths (or bone vectors),  
- per-link masses/inertias,  
- joint torque limits / PD gains,  
- foot geometry/contact parameters.

Then, condition **actor, critic, and discriminator** on this richer vector. This directly targets the “morphology-aware physical plausibility” goal.

### Normalize tracking errors across morphologies

Your imitation reward uses global-body MSEs averaged over bodies and coordinates. citeturn23view0  
Across very different body shapes (limb lengths, mass distribution), the same absolute error can correspond to very different *relative* deviations. A practical fix is to compute tracking error in a **morphology-normalized coordinate system**, e.g.:

- divide positional errors by a per-body length scale (bone length, pelvis height, etc.),  
- weight body terms by mass (or inverse mass) depending on what “looks wrong” perceptually,  
- weight end-effectors more heavily (feet/hands) to prevent “looks stable but wrong motion.”

PHC explicitly demonstrates multi-shape motion imitation and highlights shape variation behavior in the project materials, indicating that shape robustness is an intended capability—but it does not imply naïve unnormalized losses are optimal. citeturn30view1

### Recommended ablation order to eliminate collapse quickly

Given the failure mode you described, the fastest path to a moving solution is to isolate which incentive is causing collapse. The following sequence is deliberately minimal and high-signal:

1. **Remove power penalty entirely**, keep only imitation reward, disable early termination penalty exploitation (add terminal penalty or survival bonus), and verify that the policy can track the single motion at all. This directly tests whether collapse is driven by energy dominance. citeturn23view0turn8view1  
2. **Reintroduce power penalty with curriculum** (warm-start then ramp). citeturn23view0turn15search2  
3. **Switch to residual action tracking around the reference pose**, following RobotDancing’s residual principle, and evaluate long-horizon drift reduction. citeturn15search0turn15search16  
4. **Add adaptive tolerance on imitation sharpness**, inspired by KungfuBot’s adaptive tracking tolerance mechanism, and only then turn on AMP. citeturn15search2turn23view0  
5. **Condition the discriminator** on morphology (+ optionally phase) and re-center disc reward so stasis becomes disadvantageous. citeturn12search7turn24view0turn10view0

This roadmap aligns with the dominant patterns in 2023–2025 success cases: staged difficulty, residual corrections for long horizons, and conditional/teacher-driven supervision for imperfect reference signals. citeturn30view0turn15search3turn15search0turn15search2