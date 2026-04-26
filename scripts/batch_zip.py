"""
python scripts/batch_zip.py \
--json data-processing/valid_sorted_by_difficulty.json \
--start 0 \
--size 32 \
--files-per-motion 1
"""

import json
import shutil
import zipfile
from pathlib import Path
import argparse
import tempfile
import os

MOTION_ROOT = Path("/media/hlz/R/humos_phc_results")


def create_zip_from_sorted_json(
    json_path: str,
    output_zip: str = None,
    motion_root: Path = MOTION_ROOT,
    start: int = 0,
    size: int = 32,
    files_per_motion: int = 128,
):
    print(f"📦 Loading sorted motion IDs from: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Expected JSON format: {motion_id: difficulty_score}")

    # Sort explicitly by difficulty score, low → high
    sorted_motion_ids = [
        motion_id for motion_id, score in sorted(data.items(), key=lambda x: x[1])
    ]

    batch_items = sorted_motion_ids[start:start + size]

    if not batch_items:
        print("❌ No motion IDs selected!")
        return None

    if output_zip is None:
        output_zip = os.path.join(
            "/home/hlz/datasets",
            f"valid_sorted_start_{start}_size_{size}_motions.zip"
        )

    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        copied_count = 0

        print(f"📋 Selected {len(batch_items)} motions from valid sorted list",
              f"{files_per_motion} files per motion")

        for motion_id in batch_items:
            motion_dir = Path(os.path.join(motion_root, motion_id))

            if not motion_dir.is_dir():
                print(f"⚠️  Warning: Motion directory not found → {motion_dir}")
                continue

            files_to_copy = sorted(motion_dir.glob("*.pkl"))[:files_per_motion]

            for src in files_to_copy:
                if not src.is_file() or src.suffix != ".pkl":
                    continue

                dest = os.path.join(tmpdir, src.name)
                shutil.copy2(src, dest)
                copied_count += 1

            print(f"Copied {motion_id}, progress {copied_count}")

        print(f"✅ Copied {copied_count} motion files")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tmpdir.iterdir():
                if file_path.is_file():
                    zipf.write(file_path, arcname=file_path.name)

        zip_size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"🎉 Successfully created zip → {output_path}")
        print(f"   Motion IDs selected: {len(batch_items)}")
        print(f"   Files included: {copied_count}")
        print(f"   Size: {zip_size_mb:.1f} MB")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zip HUMOS .pkl files from valid_sorted_by_difficulty.json, low difficulty first."
    )

    parser.add_argument(
        "--json",
        required=True,
        help="Path to valid_sorted_by_difficulty.json",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output zip filename",
    )

    parser.add_argument(
        "--motion-root",
        type=Path,
        default=MOTION_ROOT,
        help=f"Motion root directory, default: {MOTION_ROOT}",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in sorted valid motion list",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=32,
        help="Number of motion IDs to include",
    )

    parser.add_argument(
        "--files-per-motion",
        type=int,
        default=128,
        help="Number of .pkl files to read from each motion_id folder. Use 1 for visualization.",
    )

    args = parser.parse_args()

    create_zip_from_sorted_json(
        json_path=args.json,
        output_zip=args.output,
        motion_root=args.motion_root,
        start=args.start,
        size=args.size,
        files_per_motion=args.files_per_motion
    )