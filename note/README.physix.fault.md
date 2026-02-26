## Asset loading: inertia and self-collision

in `env/tasks/humanoid.py`, in `_create_envs` (right before calling `gym.load_asset` and after creating `gymapi.AssetOptions()`). These attributes suppose control the physics, but it doesn't solve the humanoid explosion problem.

  - `asset_options.vhacd_enabled = True`
  - `asset_options.override_com = True`
  - `asset_options.override_inertia = True`

---

## Morphology-aware spawn height and safe resets

in `env/tasks/humanoid.py` and `ase/env/tasks/humanoid_phc.py`:

  - `_build_env` (initial root pose; currently uses a single `char_h` for all envs).
  - `_reset_envs` / `_reset_actors`.

We should calculate the init position height offset by reading the lowest rigid body position at frome 0.

---

I remeber we finally solved this problem to some extent by changing parameteres in SMPL generation script.
By extending the size between rigid bodies.

---

