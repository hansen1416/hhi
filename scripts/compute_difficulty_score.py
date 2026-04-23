import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any

def load_representative_pkl(
    motion_id: str,
    data_root: str | Path = "/path/to/gdrive/humos_phc_results",  # ← change to your local mount / Google Drive path
    gender: str = "male",          # or "female" — pick whichever you have for every motion_id
    beta_key: str = "mean"         # or any beta key that exists for this motion_id
) -> Dict[str, Any]:
    """
    Loads one representative .pkl for a motion_id.
    You can change gender/beta_key or make this pick the first file automatically.
    """
    pkl_path = Path(data_root) / motion_id / f"{gender}_{beta_key}.pkl"
    if not pkl_path.exists():
        # fallback: pick the first .pkl in the folder
        files = list(Path(data_root / motion_id).glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"No .pkl found for motion_id {motion_id}")
        pkl_path = files[0]
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def compute_difficulty_score(pkl_data: Dict[str, Any]) -> float:
    """
    Computes a scalar Difficulty Score for one HUMOS-generated motion.
    
    This is a lightweight kinematic proxy that correlates strongly with imitation difficulty
    (inspired by the Motion Difficulty Score from the 2025 "Benchmarking Humanoid Imitation Learning"
    paper + PHC's hard-sequence mining). Higher score = harder motion for physics-based tracking.
    
    The function works with the .pkl structure you generated from HUMOS on AMASS.
    It only needs root translation/poses (or root_pos/dof_pos) — no physics engine required.
    
    Adapt the key names below (TODOs) to match your exact .pkl dict.
    Typical keys in HUMOS/AMASS-style pickles:
        - 'trans' or 'root_pos'      (N_frames x 3)
        - 'poses' or 'dof_pos'       (N_frames x 72 or N_frames x num_dofs)
        - 'root_rot' or 'orient'     (optional)
    """
    
    # === 1. Extract motion arrays (adapt these keys to your .pkl) ===
    # TODO: change the key names if your pickle uses different keys
    if "trans" in pkl_data:                     # common in AMASS/HUMOS
        root_pos = np.asarray(pkl_data["trans"])          # (T, 3)
    elif "root_pos" in pkl_data:
        root_pos = np.asarray(pkl_data["root_pos"])
    else:
        raise KeyError("Could not find root position key ('trans' or 'root_pos') in .pkl")

    # velocities (if precomputed, great; otherwise compute from positions)
    if "root_vel" in pkl_data:
        root_vel = np.asarray(pkl_data["root_vel"])       # (T, 3)
    else:
        # finite difference
        root_vel = np.diff(root_pos, axis=0) * 30.0       # assume 30 Hz (adjust if your motion_dt differs)
        root_vel = np.concatenate([np.zeros((1, 3)), root_vel], axis=0)

    # joint poses / DoF (for angular velocity)
    if "poses" in pkl_data:                         # SMPL 72-dim
        dof_pos = np.asarray(pkl_data["poses"])[:, 3:]   # ignore root rotation (first 3)
    elif "dof_pos" in pkl_data:
        dof_pos = np.asarray(pkl_data["dof_pos"])
    else:
        dof_pos = None

    # === 2. Compute the four kinematic features ===
    # (a) max horizontal root velocity
    hvel = np.linalg.norm(root_vel[:, :2], axis=1)          # ignore vertical
    max_root_hvel = float(hvel.max())

    # (b) flight ratio (fraction of frames where both feet are airborne)
    # Simple proxy: root height > 0.15 m AND vertical velocity positive
    if "root_pos" in pkl_data or "trans" in pkl_data:
        root_height = root_pos[:, 2]
        flight_ratio = float(((root_height > 0.15) & (root_vel[:, 2] > 0.0)).mean())
    else:
        flight_ratio = 0.0

    # (c) max joint angular velocity
    if dof_pos is not None:
        dof_vel = np.diff(dof_pos, axis=0) * 30.0
        dof_vel = np.concatenate([np.zeros((1, dof_vel.shape[1])), dof_vel], axis=0)
        max_dof_vel = float(np.abs(dof_vel).max())
    else:
        max_dof_vel = 0.0

    # (d) kinetic energy variance (proxy for irregular dynamics)
    # approximate COM velocity magnitude squared
    com_vel_mag = np.linalg.norm(root_vel, axis=1)
    kinetic_var = float(np.var(com_vel_mag))

    # === 3. Normalize across the whole dataset later (here we return raw values) ===
    # For now we just compute raw features. Normalization happens once after scoring all motions.
    score = (
        0.4 * max_root_hvel +
        0.3 * flight_ratio +
        0.2 * max_dof_vel +
        0.1 * kinetic_var
    )

    return float(score)


if __name__ == "__main__":
    import json

    with open("motion_file_list.json") as f:
        motion_file_list = json.load(f)   # {motion_id: rep_pkl_path}

    for motion_id, pkl_path in motion_file_list.items():
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        score = compute_difficulty_score(pkl_data)