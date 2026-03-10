**Proposed Plan (Concise Summary)**

1. **Physics-based Motion Synthesis via RL**

   * Use *non-physical* (kinematic) motion as the **target**.
   * Train a policy in Isaac Gym where the **observation space** includes:

     * target kinematic motion,
     * humanoid rigid-body states (pose, joint velocities, contacts),
     * **body shape parameters**.
   * Condition the policy on body shape using a modulation mechanism (e.g., FiLM-like conditioning).
   * Outcome: a **physically plausible motion dataset** aligned with the original kinematic motions.

2. **Text-to-Motion Model Training**

   * Pre-train a **transformer-based text-to-motion model** on the physics-based motion dataset (supervised learning).
   * Fine-tune the model with **gradient-based learning under physics constraints** (Isaac Gym–conditioned rollout or RL-style fine-tuning) to enforce stability, balance, and feasibility.

3. **Objective**

   * Ensure text-generated motions are **physically valid by construction**, avoiding trial-and-error at inference time through intensive physics-grounded training.

4. **Key Validation Focus**

   * Stability (no falling, realistic contacts),
   * Text–motion semantic alignment,
   * Generalization across body shapes.


------


From a theoretical standpoint, **the discriminator should also be shape-conditioned**, and **the critic should at least be shape-aware**.

Your current setup is effectively:

* **actor**: explicitly conditioned on the last 11 dims `[gender, betas]` via FiLM 
* **critic**: not explicitly conditioned, but it still consumes the **full observation**, and your environment appends those 11 morphology dims to the observation
* **discriminator**: still unconditional, because it only receives `amp_obs`, and `eval_disc()` is applied directly to that AMP input for rollout, replay, and demo branches

So theoretically, the clean formulation is:

* actor learns (\pi(a \mid s, m))
* critic learns (V^\pi(s, m))
* discriminator learns (D(x \mid m))

where (m) is morphology, here `[gender, betas]`.

### Why the discriminator should be conditioned

This is the most important missing piece.

If you mix many body shapes but train an **unconditional** discriminator, it is forced to model the marginal motion distribution (p(x)), not the conditional one (p(x \mid m)). That is a mismatch, because with different shapes:

* limb lengths differ,
* inertia and contacts differ,
* the same intended motion can look dynamically different.

So an unconditional discriminator can punish a motion not because it is “unrealistic,” but because it is unrealistic **for the average body in the mixture**. That contaminates the AMP reward.

In other words, once morphology changes the physically valid realization of motion, the right adversarial target is no longer

[
D(\text{motion})
]

but

[
D(\text{motion} \mid \text{shape})
]

That is why, theoretically, **conditioning the discriminator is more important than adding stronger conditioning to the critic**.

### Why the critic should also be shape-aware

Also yes, but with one nuance.

If shape changes transition dynamics and future reward, then the correct value function is (V(s,m)), not just (V(s)). Otherwise the critic sees two states that look similar in kinematic/task terms but actually have different future return because they belong to different bodies.

So, in theory, the critic should know morphology too.

But in your case, the critic already receives the full observation, and those 11 morphology dims are appended into `obs_buf`. So it is **already shape-aware through concatenation**, even though it is not FiLM-conditioned like the actor

That means:

* **critic with concatenated shape input**: theoretically sound
* **critic with explicit FiLM conditioning**: optional refinement, not the first priority
