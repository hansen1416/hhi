# upload_checkpoint_to_wandb.py
import wandb
import torch
import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, help="Path to your existing .pth checkpoint")
parser.add_argument("--name", type=str, default=None, help="Optional wandb run name")
args = parser.parse_args()

if args.name is None:
    args.name = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

print(f"Uploading {args.ckpt} as phc_model:latest ...")

wandb.init(
    project="hhi",
    name=args.name,
    config={"resume_from": args.ckpt, "uploaded": True}
)

artifact = wandb.Artifact("hhi_film_model", type="model", description="HHI FILM checkpoint")
artifact.add_file(args.ckpt, name="hhi_film.pth")
wandb.log_artifact(artifact, aliases=["latest"])

wandb.finish()
print("✅ Uploaded successfully! Now you can download with:")
print("   wandb artifact download hhi-phc-humos/phc_model:latest")