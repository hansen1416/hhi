## This part is mostly about data processing, smoke run

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

- 1. **Shape-conditioned discriminator plumbing**

   * **Math:** let morphology be (m=[g,\beta_1,\dots,\beta_{10}] \in \mathbb{R}^{11}), AMP history be (x \in \mathbb{R}^{1960}). Your discriminator is no longer (D_\phi(x)), but (D_\phi(x,m)), with FiLM-style modulation at hidden layers:
     [
     \tilde h_\ell=\sigma(W_\ell h_{\ell-1}+b_\ell),\qquad
     h_\ell=\tilde h_\ell \odot \gamma_\ell(m)+\beta_\ell(m),\qquad
     D_\phi(x,m)=w^\top h_L+b.
     ]
     The implemented discriminator objective is the AMP BCE objective with regularization:
     [
     \mathcal L_D=\tfrac12\mathrm{BCE}(D(x_{\text{fake}},m_{\text{fake}}),0)+\tfrac12\mathrm{BCE}(D(x_{\text{real}},m_{\text{real}}),1)
     +\lambda_{\text{logit}}|w|*2^2+\lambda*{\text{gp}}|\nabla_x D(x_{\text{real}},m_{\text{real}})|*2^2+\lambda*{\text{wd}}|\theta_D|_2^2.
     ]
   * **Code record:** `humanoid_phc.py` now exports `extras["amp_shape"] = self._betas_env`; `phc_agent.py` stores `amp_shape` through rollout, dataset, replay, and minibatch assembly; `phc_models.py` evaluates all three discriminator branches as `eval_disc(amp_obs, amp_shape)`, `eval_disc(amp_obs_replay, amp_shape_replay)`, and `eval_disc(amp_obs_demo, amp_shape_demo)`; `phc_network_builder.py` builds a discriminator morphology conditioner and defines `eval_disc(self, amp_obs, amp_shape)`. ([GitHub][1])
   * **Commit history:** the discriminator-conditioning work is recorded in `e1ab5be` and `a922b08` (“shape-condition the discriminator, and critic”), then finalized in `da6cb21` (“disc-shape-condition”). ([GitHub][2])

2. **Critic FiLM conditioning**

   * **Math:** your value function is now naturally
     [
     V_\psi(s,m),
     ]
     not just (V_\psi(s)). The critic follows the same split-and-condition pattern:
     [
     s=[s_{\text{state}},m],\qquad
     h_\ell^{V}=\sigma(W_\ell^{V}h_{\ell-1}^{V}+b_\ell^{V})\odot \gamma_\ell^{V}(m)+\beta_\ell^{V}(m),\qquad
     V_\psi(s,m)=w_V^\top h_L^{V}+b_V.
     ]
     In PPO/GAE this means the TD residual is morphology-aware:
     [
     \delta_t=r_t+\gamma(1-d_t)V_\psi(s_{t+1},m)-V_\psi(s_t,m),\qquad
     A_t=\sum_{k\ge 0}(\gamma\lambda)^k\delta_{t+k}.
     ]
   * **Code record:** the builder reconstructs the critic trunk with `critic_in_dim=934`, then in `eval_critic` splits `obs[:, :934]` and `obs[:, 934:]`, runs a critic conditioner on the 11-D morphology code, and uses those FiLM parameters in the critic path before the value head. ([GitHub][3])
   * **Commit history:** the critic-conditioning stage is part of the same March 10 sequence, `e1ab5be` and `a922b08` (“shape-condition the discriminator, and critic”). ([GitHub][2])

3. **Policy observation redesign for PHC transfer, plus AMP/discriminator observation redesign**

   * **Math:** the policy observation was changed to
     [
     o_t=[o_t^{\text{self}},,o_t^{\text{task}},,m] \in \mathbb{R}^{945},
     ]
     where
     [
     \dim(o_t^{\text{self}})=358,\qquad
     \dim(o_t^{\text{task}})=24\cdot |\text{keyBodies}|\cdot \text{numTrajSamples}=24\cdot 24\cdot 1=576,\qquad
     \dim(m)=11.
     ]
     So (358+576+11=945). For AMP, you reduced each step to
     [
     x_t=[h_{\text{root}},r_{\text{root}},v_{\text{root}},\omega_{\text{root}},q_{\text{subset}},\dot q_{\text{subset}},p_{\text{key}}]\in \mathbb{R}^{196},
     ]
     and with 10 AMP steps:
     [
     X_t=[x_t,x_{t-1},\dots,x_{t-9}]\in \mathbb{R}^{1960}.
     ]
   * **Code record:** `humanoid_phc.py` now sets `self._num_task_obs = 24 * num_key_bodies * self._num_traj_samples` and then `self._num_obs += self._num_task_obs`, giving 945 total obs. The same file also enables DOF subsetting for AMP, computes `num_amp_joints = len(self.dof_subset) // 3`, and rebuilds `_num_amp_obs_per_step` from the reduced subset, producing the 196-per-step discriminator input used later as 1960 over 10 steps. ([GitHub][1])
   * **Commit history:** `8b413be` records the policy-side change (“expand task obs from 574 to 934, task obs 7 to task obs 6”); `cb3525f` and `0f6b4d9` record the discriminator-side change (“disc obs change from 2920 to 1960”); `d16416c` and `09c3074` are later cosmetic follow-ups. ([GitHub][2])

4. **Shape-matched motion sampling for env reset and AMP demo sampling**

   * **Math:** instead of sampling motions from the unconditional marginal (p(\text{motion})), the code now implements conditional sampling
     [
     \text{motion_id} \sim p(\text{motion}\mid m),
     ]
     where (m=(\text{gender},\beta\text{-key})). For the discriminator, this means the real branch is sampled from the same conditional slice as the fake branch:
     [
     x_{\text{real}} \sim p_{\text{data}}(x\mid m),\qquad
     x_{\text{fake}} \sim p_{\pi}(x\mid m).
     ]
   * **Code record:** `humanoid.py` first parses the motion file(s) to determine which `(gender, beta_key)` pairs are actually loaded, builds humanoid assets only for those pairs, and stores `env_id_beta_keys_map`. Then `motion_lib_humos.py` constructs `beta_key_motion_id_mapping` and exposes `sample_motions(self, n, gender_beta_keys=None)` for exact shape-matched sampling. Finally, `humanoid_phc.py` uses `env_id_beta_keys_map` when fetching AMP demos, so real discriminator windows are drawn with the same body-shape condition as the env batch. ([GitHub][4])
   * **Commit history:** this path was built in stages: `1313b6c` (“correct motion sampling logic”), `1eb53da` (“prepare control betas by motion files”), `be2d59c` (“read gender betas by motion files”), and later integrated with discriminator conditioning in `da6cb21` (“disc-shape-condition”). ([GitHub][5])

5. **Manual PHC-3 transfer loader**

   * **Math:** this is a partial parameter transport from a pretrained checkpoint (\theta^{\text{old}}) into the new model (\theta^{\text{new}}):
     [
     \theta^{\text{new}}*{k*{\text{dst}}}\leftarrow \theta^{\text{old}}*{k*{\text{src}}}
     \quad\text{if keys are mapped and tensor shapes match,}
     ]
     otherwise the parameter is skipped and learned from scratch. The source checkpoint corresponds to the PHC observation layout
     [
     934=358+576,
     ]
     which is exactly why your later observation redesign was needed for transfer compatibility.
   * **Code record:** `phc_agent.py` defaults to `phc_models/phc_3_Humanoid.pth`, explicitly documents the `934 = 358 + 576` source geometry, remaps old PHC keys such as `a2c_network.pnn.actors.0.0.weight` into the current actor trunk, and loads with `strict=False` after filtering shape mismatches. ([GitHub][6])
   * **Commit history:** the transfer loader itself is the April 11 commit `766b117` (“load existing weights!”). ([GitHub][7])

6. **HUMOS motion-library height fixing and per-motion shape bookkeeping**

   * **Math:** the motion library now performs shape-aware grounding, i.e. a height-corrected translation
     [
     \tilde{\mathbf t}=\text{fix_trans_height}(\text{pose_aa},\mathbf t,m),
     ]
     where the vertical root translation is adjusted using the body mesh implied by (m=[g,\beta]). At the same time, each motion stores its own morphology code (m_i), so later queries can recover
     [
     m_i = \text{get_motion_shape}(i).
     ]
   * **Code record:** `motion_lib_humos.py` calls `MotionLibHUMOS.fix_trans_height(...)`, attaches `curr_motion.gender_beta = curr_gender_beta`, builds `beta_key_motion_id_mapping`, appends `motion_shape_list`, and exposes `get_motion_shape(self, motion_ids)` as a direct lookup of `_motion_id_shape[motion_ids]`. ([GitHub][8])
   * **Commit history:** the motion-library side evolved through `1313b6c` (“correct motion sampling logic”), `1eb53da` (“prepare control betas by motion files”), `be2d59c` (“read gender betas by motion files”), then entered the shape-conditioned discriminator stage via `a922b08` and `da6cb21`. ([GitHub][9])

7. **Training/config refinements after the baseline**

   * **Math:** the actor activation is now SiLU,
     [
     \mathrm{SiLU}(x)=x,\sigma(x),
     ]
     and the reward uses a smaller power penalty coefficient:
     [
     r_t = r_t^{\text{imit}} + r_t^{\text{AMP}} - \lambda_{\text{power}}\sum_j |\tau_{t,j}\dot q_{t,j}|,
     \qquad \lambda_{\text{power}}=5\times 10^{-5}\ \text{(current default)}.
     ]
   * **Code record:** `phc_humanoid.yaml` now sets the actor MLP activation to `silu`, and `humanoid_phc.py` sets `self.power_coefficient = cfg["env"].get("power_coefficient", 0.00005)`. ([GitHub][10])
   * **Commit history:** the activation change is recorded in `c4abd80` (“add silu”), and the training YAML was further updated in `bad1d73` on April 5. ([GitHub][11])

[1]: https://github.com/hansen1416/hhi/blob/main/ase/env/tasks/humanoid_phc.py "hhi/ase/env/tasks/humanoid_phc.py at main · hansen1416/hhi · GitHub"
[2]: https://github.com/hansen1416/hhi/commits/main/ase/env/tasks/humanoid_phc.py "History for ase/env/tasks/humanoid_phc.py - hansen1416/hhi · GitHub"
[3]: https://github.com/hansen1416/hhi/blob/main/ase/learning/phc_network_builder.py "hhi/ase/learning/phc_network_builder.py at main · hansen1416/hhi · GitHub"
[4]: https://github.com/hansen1416/hhi/blob/main/ase/env/tasks/humanoid.py "hhi/ase/env/tasks/humanoid.py at main · hansen1416/hhi · GitHub"
[5]: https://github.com/hansen1416/hhi/commits/main/ase/env/tasks/humanoid.py "History for ase/env/tasks/humanoid.py - hansen1416/hhi · GitHub"
[6]: https://github.com/hansen1416/hhi/blob/main/ase/learning/phc_agent.py "hhi/ase/learning/phc_agent.py at main · hansen1416/hhi · GitHub"
[7]: https://github.com/hansen1416/hhi/commits/main/ase/learning/phc_agent.py "History for ase/learning/phc_agent.py - hansen1416/hhi · GitHub"
[8]: https://github.com/hansen1416/hhi/blob/main/ase/utils/motion_lib_humos.py "hhi/ase/utils/motion_lib_humos.py at main · hansen1416/hhi · GitHub"
[9]: https://github.com/hansen1416/hhi/commits/main/ase/utils/motion_lib_humos.py "History for ase/utils/motion_lib_humos.py - hansen1416/hhi · GitHub"
[10]: https://github.com/hansen1416/hhi/blob/main/ase/data/cfg/train/rlg/phc_humanoid.yaml "hhi/ase/data/cfg/train/rlg/phc_humanoid.yaml at main · hansen1416/hhi · GitHub"
[11]: https://github.com/hansen1416/hhi/commits/main/ "Commits · hansen1416/hhi · GitHub"

8. Gneerate humos results. 
  run `python humos/infer.py --cfg humos/configs/cfg_template_test.yml`;
  It will read from "humos/annotations/humanml3d/annotations_processed.json";
  all of "train", "val", "test", 22459 motions sequences;

  Save the results to `gdrive:humos_output`

  $ rclone size gdrive:humos_output
  Total objects: 22.459k (22459)
  Total size: 778.054 GiB (835428739234 Byte)
  
  Each motion will have 128 variations, male/female * 64 body shapes;
  The 64 body shapes are evenly spreaded between -3.0 and 3.0 (/home/hlz/repos/SMPLSim/run.py);

9. Process humos results

  `/home/hlz/repos/PHC/cmd/list_all_humos.sh` will save all filenames in `gdrive:humos_output` to local file `/home/hlz/repos/PHC/cmd/all_humos_files.txt`

  `/home/hlz/repos/PHC/cmd/split_humos_files.sh` split them to 4 parts

  # To copy them from gdrive to local:

  PART=4 && rclone copy gdrive:humos_output /home/hlz/datasets/humos_output_part${PART} --files-from=/home/hlz/repos/PHC/cmd/all_humos_part${PART}.txt --progress --transfers=32 --checkers=64 --drive-chunk-size=256M --fast-list

  # Command to to process them and save to potable drive.

  `cd /home/hlz/repos/PHC`   # or wherever your PHC fork lives
  # processes humos_output_part2 → humos_phc_results_part2
  `python scripts/humos2phc_data_gpu.py 2`     
  # processes part 3, etc.
  `python scripts/humos2phc_data_gpu.py 3`

10. figure a fast way to upload humos_phc_results to gdrive
  
