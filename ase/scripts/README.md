- `bad_capsule.py` checks for problematic capsules in the humanoid model, but it currently seems irrelevant.

- `load_motion_npy.py` loads an AMASS motion stored in .npy format.

- `load_motion_pkl.py` loads an AMASS motion stored in .pkl format.

- `mujoco_test.py` checks for geometry overlaps in the humanoid model using MuJoCo.

- `vis_motion_multiple.py` loads multiple humanoids into an Isaac Gym scene without applying any control; if a humanoid is unstable, it may start flying.

- `vis_motion_single.py` loads a single humanoid into an Isaac Gym scene without applying any control; if it is unstable, it may start flying.