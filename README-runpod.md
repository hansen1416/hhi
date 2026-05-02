Following instruction in `PHC/docs/docker_instruction.MD`

---

cd /home/gymuser/ && \
git clone https://github.com/hansen1416/hhi.git && \
cd /home/gymuser/hhi && \
mkdir -p output artifacts hhi_models phc_models humos_results && \
chmod -R 777 output artifacts

<!-- phc_3_Humanoid.pth -->
gdown 1q-IcBL-MUuvtMAjEKi1YREYT6PAh1umQ

<!-- smpl_model.zip -->
gdown 1SGcAjy9YciAFkuAzPtCtKMbAO_A3tfa8

<!-- simple_walk_motions  -->
gdown 1pWVFwgkJD5Yc73nar99Emnlb7m4s5z3W

<!-- valid_sorted_start_256_size_32_motions.zip -->
gdown 1Xp4IonnXhYrC5kEM0tTov58ovwz4A209

<!-- valid_sorted_start_288_size_64_motions.zip -->
gdown 1g-QCgYtS0IuCESMgGm6BjUuDkn_1uh9Y

<!-- hhi_film_model_0426_256_32.pth -->
gdown 1l1dy6wz5wVdyc8TLMvAn-sRRXrX7ja6N

<!-- hhi_film_phc_transfer_0427_256_32.pth -->
gdown 1I5BRwMF1rSj4FbZxelpDPWtIHQ0LKnez

<!-- film_256_32_288_64.pth -->
gdown 1uOUXkQp__lA-DTfeZVC6CLbDv7qxt7n2

<!-- transfer_256_32_288_64.pth -->
gdown 13o8D207zPnrhRwwriVuxwZAQ3w68MNnX

unzip valid_sorted_start_256_size_32_motions.zip -d ./humos_results/

unzip smpl_model.zip && \
mv smpl_model ase/data/ && \
rm smpl_model.zip

mv phc_3_Humanoid.pth ./phc_models

rm valid_sorted_start_256_size_32_motions.zip

-----------------------------

wandb login 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

-----------------------------

python ase/run.py --task HumanoidHHI --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/gymuser/hhi/humos_results/ --headless

python ase/run.py --task HumanoidHHI --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file ./humos_results/ --checkpoint hhi_film_model_0426_256_32.pth --headless

python ase/run.py --task HumanoidHHITraj --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/gymuser/hhi/humos_results/ --wandb_artifact hhi_traj --headless

python ase/run.py --task HumanoidHHIRootOffset --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/gymuser/hhi/humos_results/ --wandb_artifact hhi_root_offset --headless

-----------------------------

tmux new -s hhi

python ase/run.py --task HumanoidTransfer --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/transfer_humanoid.yaml --motion_file ./humos_results/ --headless

python ase/run.py --task HumanoidTransfer --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/transfer_humanoid.yaml --motion_file /home/gymuser/hhi/humos_results/ --checkpoint hhi_film_phc_transfer_0427_256_32.pth --headless

------------

# Then detach from tmux by pressing:

`Ctrl+b, then d`

# Later, reconnect with:

`tmux attach -t hhi`

tmux kill-session -t hhi