Following instruction in `PHC/docs/docker_instruction.MD`

---

cd /home/

git clone https://github.com/hansen1416/ASE.git

cd /home/ASE/ && mkdir output && chmod -R 777 output/


scp -i ~/.ssh/id_ed25519 /home/hlz/datasets/humos_results.zip root@202.181.159.138:/home/ASE

apt install zip unzip


-----------------------------

> -i (--interactive): keeps STDIN open and attaches it to your terminal.

> -t (--tty): allocates a pseudo-TTY so programs think they are in a real terminal.

> Together -it means: run the container and immediately attach my terminal to its STDIN/TTY → Docker then attaches you directly into /bin/bash.

> -d (--detach): run the container in the background and do not attach your terminal; Docker just prints the container ID and returns you to your shell.

docker run -d \
--mount type=bind,source=$HOME/repos/ASE,target=/home/gymuser/ASE \
--network=host \
--gpus=all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
hansen1416/phc \
tail -f /dev/null

docker run -d \
--mount type=bind,source=/home/ASE,target=/home/gymuser/ASE \
--network=host \
--gpus=all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
hansen1416/phc \
tail -f /dev/null

docker exec -it <CONTAINER_ID> /bin/bash


---------------


python ase/run.py --task HumanoidPHC --cfg_env ase/data/cfg/humanoid_phc.yaml --cfg_train ase/data/cfg/train/rlg/phc_humanoid.yaml --motion_file /home/gymuser/ASE/humos_results/ --headless