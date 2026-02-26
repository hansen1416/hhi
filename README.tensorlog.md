# 1) Create a new conda env with Python 3.8.10
conda create -n tb_env python=3.8.10 -y

# 2) Activate the environment
conda activate tb_env

# 3) Install tensorboard (and pin protobuf to avoid your previous error)
pip install "tensorboard" "protobuf<5"
