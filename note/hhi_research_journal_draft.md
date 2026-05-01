## 1. Project motivation and scope

The HHI project is built around the problem of converting non-physical human motion data into physically simulated humanoid motion. The long-term aim is to construct a dataset in which motions originally represented as kinematic human motion, we use AMASS as source, reproduced by a physically controlled humanoid in simulation.

The project is based mainly on three sources of prior work:

1. **AMP / ASE**: adversarial motion priors and reusable adversarial skill embeddings for physically simulated character control.
2. **PHC**: We adopt a similar target-motion imitation learning strategy.
3. **HUMOS**: used to generate motion variants (2*64=128) across body shapes and genders from AMASS dataset.

The key extension is **heteromorphic imitation**: the same motion content is associated with multiple humanoid morphologies, parameterized by gender and SMPL betas.

Let the morphology condition be

\[
m = [g, \beta_1, \ldots, \beta_{10}] \in \mathbb{R}^{11},
\]

where `g` is encoded as:

- `male` → `+1`
- `female` → `-1`
- `neutral` → `0` not used, since HUMOS only support male/female.

Each target motion can therefore be treated as a pair:

\[
(\text{motion content}, \text{morphology}).
\]

The intended training objective is not merely to imitate a motion, but to learn a controller that can realize motion across a broad distribution of body shapes.

---

## 2. Data and morphology generation

### 2.1 Initial design: multiple humanoid templates

The first major implementation step was to allow the Isaac Gym environment to load multiple SMPL humanoid XML templates rather than a single fixed humanoid. The base humanoid code was modified so that the environment can construct actors from different XML files of the form:

```text
mjcf/smpl/{gender}_{beta_key}_smpl.xml
```

For each loaded humanoid asset, the code checks that all templates remain compatible at the simulation-structure level:

- same number of rigid bodies;
- same number of degrees of freedom;
- same number of joints;
- same actuator effort limits.

This is important because all humanoids share one policy/action space. If one morphology had a different DOF count or actuator layout, a single actor network could not control all bodies consistently.

The current SMPL humanoid uses 24 rigid bodies and 23 actuated joints. Each joint is represented as a 3-DoF spherical joint, giving:

\[
23 \times 3 = 69
\]

action dimensions.

The current body list is:

```python
['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe',
 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe',
 'Torso', 'Spine', 'Chest', 'Neck', 'Head',
 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
```

The DOF bodies exclude the root pelvis, so the DOF names are the remaining 23 bodies.

### 2.2 Shape code in the observation

The base humanoid observation was extended with the morphology code:

```python
[gender, beta_1, ..., beta_10]
```

This adds 11 dimensions to the observation.

The self-observation part is computed as:

\[
1 + N_{body}(3 + 6 + 3 + 3) - 3
\]

where:

- `1` is root height;
- `3` is body position;
- `6` is body rotation in tangent-normalized quaternion form;
- `3` is body linear velocity;
- `3` is body angular velocity;
- `-3` removes the root position from the relative body-position part.

For 24 bodies:

\[
1 + 24 \times 15 - 3 = 358.
\]

After appending morphology:

\[
358 + 11 = 369.
\]

The current HHI task then appends the target-motion task observation, described later.

### 2.3 Generating 64 body shapes and two genders

The data-generation plan uses 64 body-shape variants and two genders. Therefore, each motion can be expanded into:

\[
64 \times 2 = 128
\]

morphology variants.

The 64 beta settings are evenly distributed across a range approximately from `-3.0` to `3.0`. The practical intention is not to model every possible human body, but to cover a sufficiently broad and balanced morphology space so that the policy cannot overfit to one default body shape.

The notes record an important sampling assumption:

Because each motion key has the same set of morphology variants, the dataset can be treated as a factorized space:

\[
(m, a),
\]

where `m` indexes motion content and `a = (g, \beta)` indexes morphology. Training over all variants per motion gives an unbiased Monte Carlo estimate of the morphology-averaged objective:

\[
\min_\pi \mathbb{E}_{m \sim p(m)} \mathbb{E}_{a \sim \mathrm{Unif}(\mathcal{A})}
\left[L(\pi; m, a)\right].
\]

This removes motion–morphology confounding. In other words, the policy should not learn that a particular motion only belongs to a particular body type.

### 2.4 Stability fixes for generated humanoids

There were several attempts to stabilize generated humanoid XMLs.

Earlier attempts included:

- using `real_weight = False` for some unstable humanoids;
- tuning body/capsule separation in `smpl_sim/smpllib/skeleton_local.py`;
- increasing separation for central torso components:

```python
if bone.name in ["Torso", "Chest", "Spine"]:
    separation = 0.6
else:
    separation = 0.2
```

The main successful fix was to explicitly set the MuJoCo compiler coordinate and angle convention after parsing the XML tree in `smpl_sim/smpllib/smpl_local_robot.py`:

```python
compiler = self.tree.getroot().find("compiler")
compiler.attrib["coordinate"] = "local"
compiler.attrib["angle"] = "radian"
```

The likely reason is that if the angle convention is missing or defaults incorrectly, joint ranges and axis values can be interpreted in degrees rather than radians, producing invalid joint limits or unstable initial poses.

At the time of the notes, the project had reached 64 stable body shapes.

---

## 3. HUMOS output generation and processing

### 3.1 HUMOS inference

The HUMOS generation command recorded in the notes is:

```bash
python humos/infer.py --cfg humos/configs/cfg_template_test.yml
```

The inference reads from:

```text
humos/annotations/humanml3d/annotations_processed.json
```

The processed dataset includes all splits:

- train;
- validation;
- test.

The total number of motion sequences is recorded as:

```text
22,459 motion sequences
```

The HUMOS output is stored on Google Drive:

```text
gdrive:humos_output
```

The recorded size is:

```text
Total objects: 22.459k
Total size: 778.054 GiB
```

Each motion has 128 variants:

```text
male/female × 64 body shapes
```

### 3.2 Splitting and copying HUMOS output

The notes record a file-list based copying workflow.

First, list all HUMOS files:

```bash
/home/hlz/repos/PHC/cmd/list_all_humos.sh
```

This writes filenames to:

```text
/home/hlz/repos/PHC/cmd/all_humos_files.txt
```

Then split the file list into four parts:

```bash
/home/hlz/repos/PHC/cmd/split_humos_files.sh
```

Example command for copying one part from Google Drive:

```bash
PART=4 && rclone copy gdrive:humos_output /home/hlz/datasets/humos_output_part${PART} \
  --files-from=/home/hlz/repos/PHC/cmd/all_humos_part${PART}.txt \
  --progress \
  --transfers=32 \
  --checkers=64 \
  --drive-chunk-size=256M \
  --fast-list
```

### 3.3 Converting HUMOS output to PHC-style data

The notes record conversion through the PHC fork:

```bash
cd /home/hlz/repos/PHC
python scripts/humos2phc_data_gpu.py 2
python scripts/humos2phc_data_gpu.py 3
```

The resulting data is stored in part-specific PHC-compatible output folders.

### 3.4 Empty-space validity filtering

A later filtering stage aims to remove motions that cannot be performed by an isolated humanoid in empty space. This is important because many text-motion datasets contain actions that require furniture, props, terrain, walls, another person, animals, or environmental supports.

Recorded statistics:

```text
Input motions: 22459
Hard invalid candidates: 1068
Soft review candidates: 3543
Total flagged candidates: 4611
```

Hard invalid categories:

```text
furniture_or_seat_support: 217
terrain_or_structure: 205
other_person_or_animal_contact: 188
table_shelf_counter_surface: 182
wall_door_window_structure: 111
obstacle_collision_or_gap: 88
external_support_contact: 53
forceful_object_interaction: 34
environment_medium: 13
```

Soft review categories:

```text
prop_or_tool_semantic: 3537
sports_or_instrument_semantic: 339
```

Interpretation:

- **hard_invalid**: the motion requires external support, terrain, furniture, wall/door/window/table/shelf/counter/surface, obstacle collision, another person/animal, water/shower medium, or forceful object interaction;
- **soft_review**: the motion semantically uses props or tools, but may still be acceptable if the humanoid is allowed to pantomime the object.

The unresolved issue is how to treat sitting motions. Sitting can be physically meaningful, but if there is no chair or surface in the environment, many sitting motions become invalid for empty-space humanoid imitation.

---

## 4. Motion library: `MotionLibHUMOS`

The project replaces or extends the original SMPL motion loader with a HUMOS-aware motion library.

The current motion library supports two input modes:

1. a single motion file;
2. a directory of `.pkl` motion files.

For directory mode, the file names encode motion identity, gender, and beta key. The loaded data includes fields such as:

```python
'beta'
'beta_key'
'gender'
'pose_aa'
'pose_quat_global'
'trans_orig'
'fps'
```

### 4.1 Skeleton-tree construction per morphology

For each loaded `(gender, beta_key)` pair, `MotionLibHUMOS` constructs a corresponding skeleton tree from the matching XML:

```text
ase/data/assets/mjcf/smpl/{gender}_{beta_key}_smpl.xml
```

This ensures that forward kinematics uses the correct body template.

### 4.2 Height fixing

The loader applies shape-aware height correction using the SMPL mesh implied by the morphology code:

```python
MotionLibHUMOS.fix_trans_height(pose_aa, trans, curr_gender_beta, fix_height_mode)
```

Conceptually, this corrects the vertical translation:

\[
\tilde{t}_z = t_z - \Delta h(m),
\]

where the height offset depends on the body shape and pose. This is needed because different beta shapes have different mesh extents and root offsets.

The current method checks a small number of initial frames, computes the minimum vertex height after accounting for root offset, applies a tolerance, and shifts the translation accordingly.

### 4.3 Motion-state outputs

For a given `motion_id` and `motion_time`, the motion library returns:

```python
root_pos
root_rot
dof_pos
root_vel
root_ang_vel
dof_vel
motion_aa
rg_pos
key_pos
rb_rot
body_vel
body_ang_vel
motion_bodies
```

These values are used by:

- environment reset;
- task observation construction;
- imitation reward computation;
- AMP demo observation construction;
- target-marker visualization.

### 4.4 Conditional motion sampling

The motion library builds a mapping:

```python
beta_key_motion_id_mapping[(gender, beta_key)] -> set(motion_ids)
```

This allows shape-matched motion sampling:

```python
sample_motions(n, gender_beta_keys)
```

If `gender_beta_keys` is provided, the loader samples a valid motion for each requested body condition. This is important for heteromorphic imitation: an environment with a specific body shape should reset to a motion generated for the same shape.

If `gender_beta_keys` is `None`, the loader samples uniformly across all motions.

### 4.5 Motion-shape lookup

The loader stores:

```python
_motion_id_shape: [num_motions, 11]
```

This allows later code to recover the morphology code for a sampled motion:

```python
get_motion_shape(motion_ids)
```

This is needed by the shape-conditioned discriminator so that real demo windows and fake rollout windows are paired with the appropriate morphology condition.

---

## 5. Humanoid environment and simulation design

### 5.1 Base humanoid task

The base humanoid task inherits from `BaseTask` and constructs the Isaac Gym simulation. It handles:

- loading humanoid assets;
- creating environments;
- creating actors;
- acquiring root, DOF, rigid-body, contact-force, force-sensor, and DOF-force tensors;
- maintaining per-environment morphology codes;
- setting PD targets or torque forces;
- building observation and action sizes.

The code uses ankle force sensors by default:

```python
force_sensor_joints = ["L_Ankle", "R_Ankle"]
```

The current actuation path is PD control when `pdControl=True`:

```python
pd_target = pd_action_offset + pd_action_scale * action
```

If PD control is disabled, actions are interpreted as force commands scaled by actuator effort limits and `powerScale`.

### 5.2 Current main observation layout: 585-D

The current `main` implementation uses the following observation layout:

```text
obs = [self_obs, task_obs, morphology]
```

Current dimensions:

```text
self_obs = 358
task_obs = 9 × |keyBodies|
morphology = 11
```

With 24 key bodies:

\[
9 \times 24 = 216.
\]

Therefore:

\[
358 + 216 + 11 = 585.
\]

The task observation version currently used is `v7`:

```python
self._task_obs_v = 7
self._num_task_obs = 9 * len(key_bodies)
```

This stores, for each key body:

1. local position difference between reference and current body;
2. local velocity difference between reference and current body;
3. local reference position relative to the root.

For each key body:

\[
\Delta p_{local} \in \mathbb{R}^3,
\quad
\Delta v_{local} \in \mathbb{R}^3,
\quad
p^*_{rel, local} \in \mathbb{R}^3.
\]

Total per key body:

\[
3 + 3 + 3 = 9.
\]

### 5.3 Historical transfer observation layout: 945-D

A later PHC-transfer-oriented note records a different observation design:

```text
self_obs = 358
task_obs = 576
morphology = 11
total = 945
```

The 576-D task observation was designed as:

\[
24 \times |keyBodies| \times numTrajSamples.
\]

With 24 key bodies and one trajectory sample:

\[
24 \times 24 \times 1 = 576.
\]

This change was introduced for PHC checkpoint compatibility, because the source PHC observation layout was recorded as:

\[
934 = 358 + 576.
\]

After appending morphology:

\[
934 + 11 = 945.
\]

This should be treated as a transfer-learning stage or branch-specific design, not the same as the current `main` code path, which still shows the 585-D layout.

### 5.4 Reset strategy

The environment uses hard reset into a frame from the target motion. There is no failure recovery stage in the earlier notes.

For each reset environment:

1. identify the environment’s morphology via `env_id_beta_keys_map`;
2. sample a motion with the same `(gender, beta_key)`;
3. sample a valid start time;
4. query `MotionLibHUMOS.get_motion_state`;
5. set root pose, root velocity, root angular velocity, DOF position, and DOF velocity directly;
6. upload root and DOF states back into Isaac Gym tensors;
7. recompute observations and AMP history.

Training mode uses random start times with truncation so AMP history windows remain valid. Test/play mode can force motion playback from `t = 0` so the complete motion can be visualized.

### 5.5 Target marker and follow camera

The project added feature-style plugins:

- `TargetMarkerFeature`
- `FollowCameraFeature`

The target marker is enabled during non-headless play/test visualization. The marker uses reference key-body positions from the target motion, allowing visual comparison between the simulated humanoid and target motion.

This was part of the earlier design notes around moving `_compute_observations` and `_compute_task_obs_v7` into the HHI task so that reference key positions can be passed into marker hooks.

---

## 6. AMP observation design

### 6.1 Current main AMP observation layout: 2920-D

The current `main` code has:

```python
self._has_dof_subset = False
self._has_shape_obs_disc = False
self._has_limb_weight_obs_disc = False
```

AMP observation per step is:

```text
root height                 1
root rotation               6
root linear velocity        3
root angular velocity       3
dof pose observation        138
dof velocity                69
key-body local positions    3 × |keyBodies|
```

The first four terms sum to 13:

\[
1 + 6 + 3 + 3 = 13.
\]

For 23 joints:

\[
dof\_obs = 23 \times 6 = 138,
\]

and

\[
dof\_vel = 23 \times 3 = 69.
\]

With 24 key bodies:

\[
3 \times 24 = 72.
\]

Therefore:

\[
13 + 138 + 69 + 72 = 292.
\]

With 10 AMP history steps:

\[
292 \times 10 = 2920.
\]

This is the current AMP observation layout for the current HHI baseline path.

### 6.2 Historical PHC-transfer AMP layout: 1960-D

The notes also record a transfer-oriented discriminator observation redesign:

```text
AMP obs per step = 196
AMP history steps = 10
total AMP obs = 1960
```

The motivation was compatibility with a PHC-3 discriminator checkpoint whose AMP input size was 1960.

The idea was to remove hands and toes from the discriminator DOF subset:

```python
remove_names = {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}
```

This reduced the DOF-related part of the discriminator observation. The transfer-stage note says this was recorded in commits related to changing discriminator observation from 2920 to 1960.

Again, this should be kept as a separate historical stage, because the current `main` code has `_has_dof_subset = False` and therefore still uses the full 2920-D AMP observation.

---

## 7. Reward design

### 7.1 Imitation reward

The PHC-style imitation reward was migrated into the ASE/HHI task.

The reward contains four main terms:

1. body position reward;
2. body rotation reward;
3. body linear velocity reward;
4. body angular velocity reward.

The current reward specification is:

```python
reward_specs = {
    "k_pos": 50,
    "k_rot": 30,
    "k_vel": 0.2,
    "k_ang_vel": 0.2,
    "w_pos": 0.45,
    "w_rot": 0.25,
    "w_vel": 0.15,
    "w_ang_vel": 0.15,
}
```

The reward form is approximately:

\[
r = w_p e^{-k_p E_p}
  + w_r e^{-k_r E_r}
  + w_v e^{-k_v E_v}
  + w_\omega e^{-k_\omega E_\omega}.
\]

The notes record that the reward was verified to be maximal when the humanoid is reset exactly to a frame in the target motion.

### 7.2 Power penalty

A power penalty is added:

\[
r_{power} = -\lambda_{power} \sum_j |\tau_j \dot{q}_j|.
\]

The current default coefficient is:

```python
power_coefficient = 0.00005
```

The first few frames are excluded:

```python
power_reward[self.progress_buf <= 3] = 0
```

This avoids penalizing artifacts immediately after reset.

### 7.3 Smoothness reward for jitter mitigation

The current base humanoid code contains jitter-mitigation switches:

```python
pd_gain_tuning = False
action_filtering = False
smoothness_reward = True
```

If `smoothness_reward=True`, the task adds an action-difference penalty:

\[
r_{smooth} = -\lambda_{smooth} \|a_t - a_{t-1}\|^2.
\]

The default coefficient is:

```python
smoothActionCoef = 0.005
```

The first frame is excluded:

```python
smooth_reward[self.progress_buf <= 1] = 0
```

The notes also record a disabled or experimental jitter-fix commit. The recommended order was:

1. tune PD gains first;
2. add light action filtering;
3. add a small smoothness reward.

Proposed PD tuning:

```text
K_d × 1.5–2.0
K_p × 0.7–0.9
```

Proposed action filter:

\[
a_t^f = \alpha a_{t-1}^f + (1 - \alpha)a_t,
\]

with:

```text
alpha = 0.8–0.9
```

The current `main` code has action filtering implemented but disabled by default.

---

## 8. RL algorithm: HHI agent

The HHI agent is an AMP-style agent built on top of an RL-Games PPO-style common agent.

The high-level training loop per epoch is:

1. collect rollout trajectories for `horizon_length` steps;
2. store standard PPO tensors:
   - observations;
   - actions;
   - values;
   - log probabilities;
   - action distribution means and sigmas;
   - rewards;
   - dones;
   - next observations;
3. store AMP-specific tensors:
   - `amp_obs`;
   - `amp_shape`;
   - `rand_action_mask`;
4. compute discriminator rewards from AMP observations;
5. combine task reward and discriminator reward;
6. compute PPO advantages and returns;
7. sample real demo AMP windows;
8. sample replay fake AMP windows;
9. train actor, critic, and discriminator jointly;
10. store latest fake AMP observations into the replay buffer.

### 8.1 Reward combination

The combined reward is:

\[
r = w_{task} r_{task} + w_{disc} r_{disc}.
\]

The weights are loaded from config:

```python
task_reward_w
disc_reward_w
```

### 8.2 Discriminator reward

The discriminator returns a logit. The AMP reward is computed from the probability that the motion window is real/demo-like:

\[
p = \sigma(D(x,m)).
\]

The discriminator reward is:

\[
r_{disc} = -\log(\max(1 - p, \epsilon)) \cdot scale.
\]

This rewards the policy when its motion windows become difficult to distinguish from real demo motion windows.

### 8.3 Discriminator loss

The discriminator receives three branches:

1. current rollout fake samples;
2. replay-buffer fake samples;
3. demo real samples.

The current and replay fake branches are concatenated for the fake side of the loss. The demo branch is the real side.

The discriminator loss consists of:

- BCE loss for fake samples with label 0;
- BCE loss for demo samples with label 1;
- logit-layer regularization;
- gradient penalty on demo samples;
- optional discriminator weight decay.

Conceptually:

\[
\mathcal{L}_D =
\frac{1}{2}\mathrm{BCE}(D(x_{fake},m_{fake}),0)
+
\frac{1}{2}\mathrm{BCE}(D(x_{real},m_{real}),1)
+
\lambda_{logit}\|w\|_2^2
+
\lambda_{gp}\|\nabla_x D(x_{real},m_{real})\|_2^2
+
\lambda_{wd}\|\theta_D\|_2^2.
\]

### 8.4 Epsilon-greedy action mixing

The agent includes an epsilon-greedy-style action selection mechanism.

For each environment, it samples a Bernoulli mask. If the mask is 0, the sampled stochastic action is replaced by the deterministic mean action `mu`.

Purpose:

- stochastic actions improve exploration;
- deterministic actions produce smoother motion;
- a mixture can prevent the discriminator from simply identifying fake samples by high-frequency stochastic jitter.

The mask is also stored and used to mask actor-related losses, so deterministic-action frames do not incorrectly drive the stochastic policy update.

### 8.5 Termination-aware bootstrapping

The agent masks next-state values through failure termination:

```python
next_values *= (1.0 - terminated)
```

This avoids bootstrapping through falls or failure states.

---

## 9. Shape-conditioned actor, critic, and discriminator

### 9.1 Actor FiLM conditioning

The actor uses the observation split:

```text
obs[:, :574]  -> state/task input
obs[:, 574:]  -> morphology condition [gender, betas]
```

The actor trunk consumes only the state/task part. The morphology code is passed through a conditioner network:

```text
11 -> 64 -> 64 -> FiLM parameters
```

For each actor hidden layer, the conditioner outputs:

```text
gamma_i, beta_i
```

The hidden activation is modulated as:

\[
h_i = \tilde{h}_i \odot \gamma_i(m) + \beta_i(m).
\]

This is preferable to simple concatenation because it allows morphology to modulate internal features without forcing the state representation and morphology representation to mix only at the input layer.

### 9.2 Critic FiLM conditioning

The current network builder reconstructs the critic trunk with the same split:

```text
state/task input: 574 dims
morphology condition: 11 dims
```

The critic also receives FiLM parameters from a morphology conditioner and estimates:

\[
V(s,m).
\]

This makes the value function morphology-aware. This is theoretically appropriate because body shape affects dynamics, contacts, action effectiveness, and therefore future return.

### 9.3 Discriminator FiLM conditioning

The discriminator is also shape-conditioned. It receives:

```text
amp_obs: motion window features
amp_shape: [gender, betas]
```

The discriminator motion trunk consumes only `amp_obs`. The morphology condition is passed through a separate FiLM conditioner and modulates the discriminator hidden layers.

Thus the discriminator is:

\[
D(x,m),
\]

not merely:

\[
D(x).
\]

This is important because the physically valid realization of a motion depends on morphology. Limb length, inertia, contact timing, and feasible joint behavior differ across shapes. An unconditional discriminator would model a marginal motion distribution and could penalize motions that are valid for one body but not typical for the average body.

### 9.4 Shape-matched real and fake discriminator branches

The model wrapper evaluates all three discriminator branches with morphology:

```python
eval_disc(amp_obs, amp_shape)
eval_disc(amp_obs_replay, amp_shape_replay)
eval_disc(amp_obs_demo, amp_shape_demo)
```

The intended interpretation is:

\[
x_{real} \sim p_{data}(x \mid m),
\quad
x_{fake} \sim p_\pi(x \mid m).
\]

The discriminator objective is distributional, not pairwise. Therefore, the real sample does not need to be the exact same motion clip as the fake sample. However, it should come from the same morphology-conditioned slice of the data distribution.

---

## 10. PHC transfer learning stage

A separate set of notes records work on loading a pretrained PHC-3 checkpoint.

The key idea is partial parameter transport:

\[
\theta^{new}_k \leftarrow \theta^{old}_k
\]

only when:

- the source and target keys are mapped correctly;
- tensor shapes match.

Otherwise, the new parameter is skipped and trained from scratch.

### 10.1 Motivation

Training from scratch on physics-based humanoid control is slow and unstable. A PHC checkpoint already contains useful low-level control knowledge:

- balance;
- contact timing;
- torque patterns;
- stable locomotor behavior;
- generic humanoid dynamics.

Therefore, transfer learning should improve sample efficiency compared with training from random initialization.

### 10.2 Observation compatibility issue

The PHC checkpoint was associated with an observation layout:

```text
934 = 358 + 576
```

This explains the transfer-stage observation redesign:

```text
934 + 11 = 945
```

where the extra 11 dimensions are morphology.

The discriminator compatibility issue was similar:

```text
PHC AMP input = 1960
HHI full AMP input = 2920
```

The transfer-stage solution was to redesign the discriminator AMP observation to match the PHC-compatible 1960-D layout.

### 10.3 Checkpoint loading strategy

The recorded strategy was:

1. load checkpoint from `hhi_models/phc_3_Humanoid.pth`;
2. map PHC keys to current HHI model keys;
3. copy only matching tensors;
4. load with `strict=False`;
5. leave new FiLM conditioners and morphology-specific modules randomly initialized;
6. selectively load normalization statistics where shapes match;
7. intentionally skip reward normalization statistics.

The intended call site was inside agent initialization after the model and normalization modules exist but before training begins.

---

## 11. Curriculum, batch size, and motion difficulty

The RL notes discuss how many unique motion IDs should be used per training batch.

Important point: because each motion ID expands into 128 morphology variants, the effective dataset size per batch is already multiplied by morphology.

Practical rule recorded:

```text
128 motion_ids per batch
64 motion_ids if memory is insufficient
```

Too small a batch risks an under-diverse AMP discriminator. Too large a batch increases memory pressure and may slow convergence because the discriminator sees too broad a distribution too early.

### 11.1 Motion Difficulty Score curriculum

The notes propose an easy-to-hard curriculum based on a Motion Difficulty Score.

For each unique motion, compute features such as:

- maximum horizontal root velocity;
- flight ratio, for jump/aerial detection;
- maximum DOF velocity;
- kinetic-energy variance or COM-height variance.

Normalize each feature to `[0, 1]` and compute:

\[
\text{difficulty} =
0.4 \cdot \text{norm(max root horizontal velocity)}
+ 0.3 \cdot \text{norm(flight ratio)}
+ 0.2 \cdot \text{norm(max DOF velocity)}
+ 0.1 \cdot \text{norm(kinetic variance)}.
\]

The recorded script is:

```text
scripts/compute_difficulty_score.py
```

The plan is to sort motions by difficulty and split them into batches of 128 motions.

This curriculum is consistent with PHC-style progressive training: start from easier motions, then gradually add harder motions.

---

## 12. Current implementation notes and unresolved issues

### 12.1 Current dimension mismatch between journal notes and `main`

The notes contain two design stages:

1. current HHI baseline:

```text
obs = 585
AMP obs = 2920
```

2. PHC transfer stage:

```text
obs = 945
AMP obs = 1960
```

These should be preserved as two stages. They should not be merged into one description unless the corresponding branch/commit is specified.

### 12.2 Shape normalization detail

The current HHI task computes:

```python
betas_norm = betas.clone()
betas_norm[:, 1:] /= 3.0
```

but then appends:

```python
obs = torch.cat([obs, betas], dim=-1)
```

rather than `betas_norm`.

This may be intentional if raw betas are desired, but the comment says normalization was intended. This should be checked before large-scale training.

### 12.3 AMP shape demo buffer size

The current `fetch_amp_obs_demo` path builds demo AMP observations using `self.num_envs`, then the agent samples from the demo replay buffer using the number of rollout samples. This is probably acceptable if the replay buffer accumulates demo samples over time, but the exact shape-matching distribution should be checked carefully.

The ideal discriminator training distribution is:

\[
(x_{fake}, m) \sim p_\pi(x \mid m),
\quad
(x_{real}, m) \sim p_{data}(x \mid m).
\]

The real branch should match the fake branch morphology distribution, not merely sample globally.

### 12.4 Sitting motions

The filtering notes still leave sitting positions unresolved. The main question is whether sitting should be removed entirely for empty-space humanoid imitation, or retained only when the simulator contains a chair/seat geometry.

A practical rule may be:

- remove clear chair-dependent sitting from empty-space training;
- retain crouching, squatting, kneeling, or ground-sitting only if physically feasible without props;
- create a separate interaction dataset later for chair/table/object-dependent motions.

### 12.5 Jitter and arm shaking

Jitter remains an important training issue. The current order of interventions should be:

1. quantify jitter using jerk, action difference, torque variance, and high-frequency spectral energy;
2. use PD gain tuning if oscillation is physically induced;
3. use action filtering if high-frequency action noise dominates;
4. use smoothness reward carefully so imitation accuracy is not over-smoothed;
5. increase motion diversity and use curriculum training to reduce single-clip overfitting.

---

## 13. Current concise status

The project has implemented the core infrastructure needed for shape-conditioned physics-based humanoid imitation:

- multiple SMPL humanoid bodies in the same Isaac Gym task;
- 11-D morphology code in observation and discriminator conditioning;
- HUMOS-based motion loading with shape-aware skeletons;
- shape-matched reset and motion sampling;
- PHC-style hard reset into target motion frames;
- target-motion task observations;
- imitation reward plus power and smoothness penalties;
- AMP discriminator reward integrated into PPO training;
- FiLM-conditioned actor, critic, and discriminator;
- replay/demo buffers carrying both AMP observations and shape codes;
- planned curriculum over motion difficulty;
- partial PHC checkpoint transfer strategy.

The main remaining tasks are:

1. decide whether the active training branch should use the current 585/2920 layout or the PHC-transfer 945/1960 layout;
2. verify shape normalization in the observation;
3. validate shape-matched demo sampling statistically;
4. finalize empty-space motion filtering, especially sitting motions;
5. run controlled smoke tests before long training;
6. log training curves, discriminator accuracy, task reward, AMP reward, fall rate, and per-motion success rate;
7. compare from-scratch training against PHC-transfer training.

---

## 14. Suggested next journal entries

For future records, each training run should include:

```text
Date:
Commit hash:
Config files:
Checkpoint initialized from:
Observation dim:
AMP observation dim:
Number of envs:
Number of motion IDs:
Number of morphology variants:
Reward weights:
Power coefficient:
Smoothness settings:
PD settings:
Action filtering settings:
Training duration:
Mean return:
Task reward:
Discriminator reward:
Discriminator real/fake accuracy:
Fall rate:
Per-motion success rate:
Qualitative result:
Failure modes:
Next action:
```

This format should make later thesis/paper writing much easier because the reasoning, implementation state, and empirical outcome are recorded together.

