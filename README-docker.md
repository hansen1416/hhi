Following instruction in `PHC/docs/docker_instruction.MD`

---

docker pull hansen1416/phc:latest

cd /home && \
git clone https://github.com/hansen1416/hhi.git && \
cd /home/hhi && \
mkdir -p output artifacts hhi_models && \
chmod -R 777 output artifacts && \
apt update && apt install -y zip unzip

gdown 1q-IcBL-MUuvtMAjEKi1YREYT6PAh1umQ && \
gdown 1SGcAjy9YciAFkuAzPtCtKMbAO_A3tfa8 && \
gdown 1Xp4IonnXhYrC5kEM0tTov58ovwz4A209

unzip valid_sorted_start_256_size_32_motions.zip -r ./humos_results/
unzip smpl_model.zip
mv smpl_model ase/data/

gdown 15mM1JLHtWXTmjrFF78bEDAQOh46ktsV-

-----------------------------

> -i (--interactive): keeps STDIN open and attaches it to your terminal.

> -t (--tty): allocates a pseudo-TTY so programs think they are in a real terminal.

> Together -it means: run the container and immediately attach my terminal to its STDIN/TTY → Docker then attaches you directly into /bin/bash.

> -d (--detach): run the container in the background and do not attach your terminal; Docker just prints the container ID and returns you to your shell.

docker run -d \
--mount type=bind,source=$HOME/repos/hhi,target=/home/gymuser/hhi \
--network=host \
--gpus=all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
hansen1416/phc \
tail -f /dev/null

docker run -d \
--name hhi \
--mount type=bind,source=/home/hhi,target=/home/gymuser/hhi \
--network=host \
--gpus=all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
hansen1416/phc \
tail -f /dev/null

docker exec -it hhi /bin/bash


python -m pip install -U "pydantic>=1.10.8,<2" && \
python -m pip install -U "wandb==0.22.3"

cd /home/gymuser/hhi/

---------------

wandb login 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

python ase/run.py --task HumanoidHHI --cfg_env ase/data/cfg/humanoid_hhi.yaml --cfg_train ase/data/cfg/train/rlg/hhi_humanoid.yaml --motion_file /home/gymuser/hhi/humos_results/ --headless

