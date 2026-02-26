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
