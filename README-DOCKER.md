Following instruction in `PHC/docs/docker_instruction.MD`

---

docker pull hansen1416/phc:latest

cd /home && \
git clone https://github.com/hansen1416/hhi.git && \
cd /home/hhi && \
mkdir -p output artifacts && \
chmod -R 777 output artifacts && \
apt update && apt install -y zip unzip

scp -i ~/.ssh/id_ed25519 /home/hlz/datasets/humos_results.zip root@202.181.159.138:/home/hhi


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


python -m pip install -U "pydantic>=1.10.8,<2"

python -m pip install -U "wandb==0.22.3"

cd /home/gymuser/hhi/

---------------

wandb login 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

python ase/run.py --task HumanoidPHC --cfg_env ase/data/cfg/humanoid_phc.yaml --cfg_train ase/data/cfg/train/rlg/phc_humanoid.yaml --motion_file /home/gymuser/hhi/humos_results/ --headless

