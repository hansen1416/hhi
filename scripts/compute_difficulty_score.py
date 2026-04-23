#!/usr/bin/env python3
"""
scripts/compute_difficulty_score.py

Computes kinematic difficulty scores for the full ~22K motion_ids.
Saves everything to data/difficulty_scores.pkl for later analysis.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
from tqdm import tqdm   # pip install tqdm if not present


def compute_difficulty_score(pkl_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns BOTH raw features + final score.
    Weights are empirically chosen (exactly as in the MDS paper arXiv:2512.07248).
    """
    # === 1. Extract motion arrays (adapt keys if your .pkl changed) ===
    if "trans_orig" in pkl_data:
        root_pos = np.asarray(pkl_data["trans_orig"])          # (T, 3)
    elif "root_pos" in pkl_data:
        root_pos = np.asarray(pkl_data["root_pos"])
    else:
        raise KeyError("Could not find root position key")

    if "root_vel" in pkl_data:
        root_vel = np.asarray(pkl_data["root_vel"])
    else:
        root_vel = np.diff(root_pos, axis=0) * 30.0       # 30 Hz assumption
        root_vel = np.concatenate([np.zeros((1, 3)), root_vel], axis=0)

    if "pose_aa" in pkl_data:
        dof_pos = np.asarray(pkl_data["pose_aa"])[:, 3:]
    elif "dof_pos" in pkl_data:
        dof_pos = np.asarray(pkl_data["dof_pos"])
    else:
        dof_pos = None

    # === 2. Raw kinematic features ===
    hvel = np.linalg.norm(root_vel[:, :2], axis=1)
    max_root_hvel = float(hvel.max())

    root_height = root_pos[:, 2]
    flight_ratio = float(((root_height > 0.15) & (root_vel[:, 2] > 0.0)).mean())

    if dof_pos is not None:
        dof_vel = np.diff(dof_pos, axis=0) * 30.0
        dof_vel = np.concatenate([np.zeros((1, dof_vel.shape[1])), dof_vel], axis=0)
        max_dof_vel = float(np.abs(dof_vel).max())
    else:
        max_dof_vel = 0.0

    com_vel_mag = np.linalg.norm(root_vel, axis=1)
    kinetic_var = float(np.var(com_vel_mag))

    # === 3. Final score (empirical weights, same principle as MDS paper) ===
    score = (
        0.4 * max_root_hvel +      # strongest predictor of dynamic balance difficulty
        0.3 * flight_ratio +       # aerial phases = high torque demand
        0.2 * max_dof_vel +        # explosive joint motion
        0.1 * kinetic_var          # irregular energy changes
    )

    return {
        "max_root_hvel": max_root_hvel,
        "flight_ratio": flight_ratio,
        "max_dof_vel": max_dof_vel,
        "kinetic_var": kinetic_var,
        "raw_score": score,
    }


if __name__ == "__main__":
    # 1. Load file list (created by build_motion_file_list.py)
    with open("scripts/motion_file_list.json") as f:
        motion_file_list: Dict[str, str] = json.load(f)

    print(f"Computing difficulty for {len(motion_file_list)} motion_ids...")

    scores_dict = {}
    for motion_id, pkl_path in tqdm(motion_file_list.items()):

        try:

            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)

            geneder_beta_key = os.path.splitext(os.path.basename(pkl_path))[0]

            motion_key = f"{motion_id}_{geneder_beta_key}"

            motion_data = pkl_data[motion_key]

            features = compute_difficulty_score(motion_data)

            scores_dict[motion_id] = features
        except Exception as e:
            print(f"⚠️  Skipping {motion_id}: {e}")
            scores_dict[motion_id] = {"error": str(e)}

    # 2. Normalize scores across the whole dataset (0–1)
    all_raw = np.array([v["raw_score"] for v in scores_dict.values() if "raw_score" in v])
    min_s, max_s = all_raw.min(), all_raw.max()
    for mid, data in scores_dict.items():
        if "raw_score" in data:
            data["normalized_score"] = float((data["raw_score"] - min_s) / (max_s - min_s + 1e-8))

    # 3. Save rich pickle for analysis
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "difficulty_scores.pkl", "wb") as f:
        pickle.dump({
            "motion_scores": scores_dict,
            "global_min_raw": float(min_s),
            "global_max_raw": float(max_s),
            "num_motions": len(scores_dict)
        }, f)

    # 4. Also save sorted curriculum batches (128 per batch, easy → hard)
    sorted_ids = sorted(scores_dict.keys(),
                        key=lambda m: scores_dict[m].get("normalized_score", 0))
    batch_size = 128
    batches = [sorted_ids[i:i+batch_size] for i in range(0, len(sorted_ids), batch_size)]

    with open(output_dir / "curriculum_batches.json", "w") as f:
        json.dump({
            "batches": batches,
            "difficulty_scores": {m: scores_dict[m]["normalized_score"]
                                  for m in sorted_ids if "normalized_score" in scores_dict[m]}
        }, f, indent=2)

    print(f"\n✅ Done! Saved {len(scores_dict)} scores → data/difficulty_scores.pkl")
    print(f"   Curriculum batches (128 motions each) → data/curriculum_batches.json")
    print(f"   Example score range: {min_s:.2f} → {max_s:.2f}")