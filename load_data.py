import numpy as np

file_path = "ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy"

data = np.load(file_path, allow_pickle=True).item()

# odict_keys(['rotation', 'root_translation', 'global_velocity', 'global_angular_velocity', 'skeleton_tree', 'is_local', 'fps', '__name__'])
# print(data.keys())


file_path1 = "ase/data/motions/amass/A2 - Sway_poses.npz"

data1 = np.load(file_path1, allow_pickle=True)
print(data1.files)