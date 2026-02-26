1. Introduce betas into the observations
    - allow env load multiple humanoid models, refer to code in `# multi humanoid template change ===============` in ase/env/tasks/humanoid_amp.py and ase/env/tasks/humanoid.py
    - load beta into observation , refer to code in `# load beta into observation ===============` in ase/env/tasks/humanoid_amp.py and ase/env/tasks/humanoid.py
    - code in `---- 1211 actions` is the attempt to fix unstable humanoid models.
2. Generate multiple betas
    - why 64 body shapes, refer to notes in ./betas
    - use `"real_weight": False, ` to fix some of the unstable humanoid.
    - tuning 
    ```
        if bone.name in ["Torso", "Chest", "Spine"]:
            seperation = 0.6
        else:
            seperation = 0.2
    ```
    in `smpl_sim/smpllib/skeleton_local.py` to fix penetration, attempt to fix them all. but there is still something wrong, try to adjust `seperation` and `capsule size`
    at this stage, we have 64 stable ones.

3. deffierence motion load stratergy.

4. new reward function for ASE, take the target motion into consideration.

    - use global `motion_times`, in reset_actor and `_compute_observations`. extract `_compute_task_obs_v7` to top level, calculate obs, key_pos, and pass them to reset_actor and _compute_observations. so we don't have to calculate motion state in `reset_actor`.

5. let target motion load multiple body shapes.

- red marker logic

- move _compute_observations to humanoid_phc, so we can move `_compute_task_obs_v7` to humanoid_phc, 
then in `_compute_task_obs_v7` we can pass `key_pos` to red marker hooks

- merge humanoid.py and humanoid_phc.py, HumanoidPHC will inherit BaseTask directly.

- can we merge the marker logic in reset env and reset actor, we probably only need key_pos. To do this, we need to first figure out how PHC build its observation, it should use the target motion as part of the observation.


- what is the purpose of `def fetch_amp_obs_demo(self, num_samples)` in humanoid_phc.py. Build the descriminator reward


- add target motion to observation space.

- Migrated PHC `compute_imitation_reward` in `humanoid_im.py` to ASE, with zero_out_far = False; _full_body_reward = True.
`compute_imitation_reward` has 4 terms: # body position reward; # body rotation reward; # body linear velocity reward; # body angular velocity reward.
Verified the reward is maximum when reset the humanoid.

- it's always Hard Resets to a frame in the target motion, no fail recover

- For training, they state they use AMASS with 480 identities: 274 male and 206 female, and they “concatenate … betas and gender” into the input features. There is no mention of a third “neutral” gender being used as a training category in their data description. So we also use only male and female for training.

- data sampling stratergy:
    Because each motion key has the *same* set of (N=192) (or 128) morphology variants and the betas are uniformly distributed, we can view the data as samples from a factorized space ((m,a)), where (m) indexes motion content and (a=(g,\beta)) indexes morphology. Our objective is morphology-robust control, i.e.,
    [
    \min_\pi\ \mathbb{E}*{m\sim p(m)}\ \mathbb{E}*{a\sim \mathrm{Unif}(\mathcal A)}\big[L(\pi; m,a)\big].
    ]
    Using all variants per motion (or sampling uniformly within each motion) yields an unbiased Monte Carlo estimate of this risk, since (|\mathcal A_m|=|\mathcal A|) for all (m) and thus (p(a\mid m)=\mathrm{Unif}(\mathcal A)). This removes motion–morphology confounding and enforces invariance to body shape: the policy must realize the same motion across a broad, uniformly covered morphology set rather than exploiting correlations between particular motions and particular bodies. Consequently, training optimizes the intended average-case performance over morphologies and empirically should improve generalization to held-out shapes relative to unbalanced or morphology-narrow sampling.

- Use humos generate results, use them as target motion for training.

- We finally fixed the unstable humanoid issue by adding these 2 lines:
    compiler = self.tree.getroot().find("compiler")
    compiler.attrib["coordinate"]     = "local"
    compiler.attrib["angle"]          = "radian"
    in `load_from_skeleton` in `smpl_sim/smpllib/smpl_local_robot.py` after `self.tree = parse(...)`.
    The likely reason is: "Similarly, if angle is missing or not "radian", any range/axis values might be silently parsed in degrees → completely wrong joint limits and initial poses."