#!/usr/bin/env python3
"""
scripts/build_motion_file_list.py

Creates a lightweight file list for curriculum difficulty scoring.
- Root: /media/hlz/R/humos_phc_results
- Output: data/motion_file_list.json  (motion_id → representative .pkl path)
- Only one .pkl per motion_id (128 variations → we pick one)
- Fast, deterministic, and progress-printed for ~22K motion_ids.
"""

from pathlib import Path
import json
from typing import Dict

def build_motion_file_list(
    data_root: str = "/media/hlz/R/humos_phc_results",
    output_path: str = "motion_file_list.json",
    preferred_gender: str = "male",      # change if you prefer "female"
    preferred_beta: str = "mean"         # change if you have a different "mean" key
) -> None:
    data_root = Path(data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"🔍 Scanning motion folders in {data_root} ...")
    motion_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])  # sorted for reproducibility

    file_list: Dict[str, str] = {}
    total = len(motion_dirs)

    for idx, motion_dir in enumerate(motion_dirs, start=1):
        motion_id = motion_dir.name
        print(f"[{idx:5d}/{total}] Processing {motion_id} ...", end="\r")

        # 1. Try preferred representative first
        preferred_file = motion_dir / f"{preferred_gender}_{preferred_beta}.pkl"
        if preferred_file.exists():
            file_list[motion_id] = str(preferred_file)
            continue

        # 2. Fallback: pick the first .pkl (alphabetically sorted)
        pkl_files = sorted(motion_dir.glob("*.pkl"))
        if pkl_files:
            file_list[motion_id] = str(pkl_files[0])
        else:
            print(f"WARNING: No .pkl found for {motion_id}")
            continue

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(file_list, f, indent=2)

    print(f"Done! Created motion file list with {len(file_list)} entries → {output_path}")
    print(f"   Example entry: {list(file_list.items())[0] if file_list else 'None'}")

if __name__ == "__main__":
    build_motion_file_list()