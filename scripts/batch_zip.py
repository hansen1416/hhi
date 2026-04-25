"""
python scripts/batch_zip.py \
  --json scripts/batches.json \
  --batch 0 \
  --output batch_0_motions.zip
"""

import json
import shutil
import zipfile
from pathlib import Path
import argparse
import tempfile
import os

# =============================================================================
# Configuration for the hhi project[](https://github.com/hansen1416/hhi)
# Builds on ASE (Adversarial Motion Priors) + PHC (Perpetual Humanoid Control)
# + HUMOS (Human Motion Model Conditioned on Body Shape, ECCV 2024)
# Goal: convert all non-physical AMASS motions → physics-based motions
# =============================================================================

MOTION_ROOT = Path("/media/hlz/R/humos_phc_results")   # ← constant as requested


def create_batch_zip(json_path: str, batch_key: str, output_zip: str = None, motion_root: Path = MOTION_ROOT):
    """
    Copy ALL motion files for one specific batch into a flat zip (no sub-folders).
    
    The batch JSON (built earlier) is assumed to contain either:
      - motion_ids (strings like "000001", "000002", ...)  ← most common now
      - or full paths to the .pkl files
    
    For each motion_id we automatically collect all 129 body-shape variations:
        {motion_root}/{motion_id}/{motion_id}_{gender}_{beta_string}.pkl
    (64 body shapes × 2 genders + neutral ≈ 129 files per original AMASS clip).
    
    All .pkl files are copied flat into the zip using only their original filename.
    Filename collisions are impossible because every filename already contains
    the unique motion_id prefix.
    """
    print(f"📦 Loading batch definition from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Resolve batch (supports dict with string keys or list of lists)
    if isinstance(data, dict):
        if batch_key in data:
            batch_items = data[batch_key]
        else:
            try:
                batch_idx = int(batch_key)
                batch_items = list(data.values())[batch_idx]
            except Exception:
                raise KeyError(f"Batch '{batch_key}' not found. Available keys: {list(data.keys())[:10]}...")
    elif isinstance(data, list):
        batch_idx = int(batch_key)
        batch_items = data[batch_idx]
    else:
        raise ValueError("JSON must be dict (batch_key → list) or list of lists")
    
    if not batch_items:
        print("❌ Batch is empty!")
        return None
    
    size = 128

    batch_items = batch_items[:size]
    
    if output_zip is None:
        output_zip = os.path.join("/home/hlz/datasets", f"batch_{batch_key}_{size}_motions.zip")
    
    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        copied_count = 0
        
        print(f"📋 Processing batch '{batch_key}' – expanding to all .pkl variations...")
        for motion_id in batch_items:

            motion_dir = Path(os.path.join(motion_root, motion_id))
            
            if not motion_dir.is_dir():
                print(f"⚠️  Warning: Motion directory not found → {motion_dir}")
                continue
            files_to_copy = list(motion_dir.glob("*.pkl"))
            
            for src in files_to_copy:
                if not src.is_file() or src.suffix != ".pkl":
                    continue
                dest = os.path.join(tmpdir, src.name)
                shutil.copy2(src, dest)
                copied_count += 1

            print(f"Copied {motion_dir} to {tmpdir}, progress {copied_count}")
        
        print(f"✅ Copied {copied_count} motion files (flat, no sub-folders)")
        
        # Create perfectly flat zip
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tmpdir.iterdir():
                if file_path.is_file():
                    zipf.write(file_path, arcname=file_path.name)
        
        zip_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"🎉 Successfully created flat zip → {output_path}")
        print(f"   Files included: {copied_count}")
        print(f"   Size: {zip_size_mb:.1f} MB")
        print(f"   Ready for next-stage physical motion training in the hhi pipeline!")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract one batch of HUMOS-generated .pkl motion files "
                    "(129 body-shape variations per motion_id) into a clean flat zip. "
                    "Part of the non-physical → physical conversion pipeline for hhi."
    )
    parser.add_argument('--json', required=True,
                        help='Path to the batch JSON we built earlier')
    parser.add_argument('--batch', required=True,
                        help='Batch identifier (string key or integer index)')
    parser.add_argument('--output', default=None,
                        help='Output zip filename (default: batch_<batch>_motions.zip)')
    parser.add_argument('--motion-root', type=Path, default=MOTION_ROOT,
                        help=f'Motion root (default: {MOTION_ROOT})')
    
    args = parser.parse_args()
    create_batch_zip(args.json, args.batch, args.output, args.motion_root)