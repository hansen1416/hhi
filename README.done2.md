# This part is about RL

1. Batch size (# motion_ids per batch)

    - AMP (Peng et al., 2021) shows that even modest motion libraries (10–56 clips, ~200–400 s total) are sufficient for a strong discriminator when the motions share stylistic coherence (e.g., locomotion only). Larger unstructured sets (hundreds of clips) improve generalization but increase variance in early training.
    - PHC (Luo et al., 2023) scales to the full cleaned AMASS (11 313 sequences) but never loads everything at once. It uses progressive training: start small, add harder sequences incrementally. This avoids memory blow-up and lets the discriminator stay focused.
    - In your case, each motion_id already expands to 128 HUMOS variations → effective library size per batch is already multiplied. Too small (<50 motion_ids) risks an under-diverse AMP discriminator (poor style prior). Too large (>500) hits Isaac Gym / RL-Games memory limits and slows per-batch convergence.

2. Transfer-learning strategy
    - PHC explicitly uses checkpoint resuming + progressive primitive addition (PMCP) to add capacity for new motions without catastrophic forgetting. Their ablation shows that fine-tuning from a previous primitive checkpoint is far more sample-efficient than training from scratch when the new data is similar (physics + humanoid dynamics).
    - ASE and later works (e.g., PULSE, Universal Humanoid Motion Representations) confirm that low-level physics skills (balance, contacts, torque patterns) transfer extremely well across motion styles. Warm-starting the actor and value network while optionally resetting/retraining only the discriminator is standard.

3. Actor network architecture
    - PHC (and the PHC+ / PULSE lineage) uses two-layer MLPs [1024, 512] for the primitive policy, value function, and discriminator. This is deliberately compact (final PHC+ model ~28 MB) yet sufficient for 11 k+ AMASS clips when combined with progressive capacity allocation.

Batch size: 128 motion_ids per batch, or 64 if not enough memory



A brand-new 2025 paper ("Benchmarking Humanoid Imitation Learning with Motion Difficulty") formalizes this idea with Motion Difficulty Score (MDS)


Curriculum Splitting Algorithm (Easy-to-Hard Batches)
Step 0: Pre-compute Difficulty Score (one-time, cheap)
For each unique motion_id:

Load one representative .pkl (e.g., neutral gender + mean-β key, or average across 2–3 variations — 128 variations are too many to average fully).
Extract kinematic features directly from the stored motion (root_pos/rot, dof_pos, velocities if present, or compute them):
max_root_hvel = max horizontal root velocity (m/s) — captures fast locomotion.
flight_ratio = fraction of frames where both feet are > 0.1 m off ground (or max root vertical vel) — captures jumps, aerial phases, dynamic balance.
max_dof_vel = max joint angular velocity across all DoFs and frames (rad/s) — captures explosive or high-frequency motion.
kinetic_var = standard deviation of approximate kinetic energy (or COM height variance) — captures irregular dynamics / energy changes.

Normalize each feature to [0, 1] across the whole dataset.
Compute scalar Difficulty Score:$$\text{difficulty\_score} = 0.4 \cdot \text{norm(max\_root\_hvel)} + 0.3 \cdot \text{norm(flight\_ratio)} + 0.2 \cdot \text{norm(max\_dof\_vel)} + 0.1 \cdot \text{norm(kinetic\_var)}$$(Weights are empirically motivated by PHC hard cases — jumps/spinkicks score high — and can be tuned later.)

This score correlates strongly with imitation difficulty (as validated in the MDS paper) and is trivial to compute in a single pass over your Google Drive files.