# ase/learning/wandb_logger.py
"""
Wandb logger utility for the HHI project (PHC + HUMOS 128 variations).
Hardcoded settings — no config needed.
"""

import wandb
import os
import getpass
from datetime import datetime


class WandbLogger:
    def __init__(self):

        USER = getpass.getuser()

        if USER == "hlz":
            self.enabled = False
        else:
            self.enabled = True

        self.project = "hhi"                    # your project name on wandb
        self.entity = None                      # set to your username/team if needed
        self.log_every = 1                      # log every N epochs
        self.run = None

        self.artifact_name = "hhi_film_model"

    def init(self, config_dict=None):
        """Call once at the start of training. run_name is auto-generated if None."""
        if not self.enabled:
            print("wandb_logger: disabled")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"hii_film_{timestamp}"

        if config_dict and isinstance(config_dict, dict):
            self.artifact_name = config_dict.get('wandb_artifact_name', self.artifact_name)

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config_dict,          # still logs the whole training config for reproducibility
            reinit=True,
            settings=wandb.Settings(start_method="fork")   # safe with Isaac Gym
        )

        # automatically log all python files in the repo (very useful for HHI)
        wandb.run.log_code(root=os.path.dirname(os.path.dirname(__file__)),
                           include_fn=lambda p: p.endswith(('.py', '.yaml', '.yml')))

        print(f"wandb_logger: initialized → {self.project}/{run_name}")

    def log(self, metrics: dict, step: int = None):
        """Log any dict of metrics. Called from _log_train_info."""
        if not self.enabled or self.run is None:
            return
        wandb.log(metrics, step=step)

    def finish(self):
        """Call at the very end of training (optional but clean)."""
        if self.run is not None:
            wandb.finish()
            print("wandb_logger: finished")

    def log_checkpoint_to_wandb(self, checkpoint_path: str, epoch: int = None):
        """Upload the latest .pth as a new version of hhi_film_model with 'latest' alias.
        
        This matches exactly the artifact name you use for loading.
        Every time you save, WandB will create v2, v3, ... and move the 'latest' alias to it.
        """
        if wandb.run is None:
            print("⚠️  WandB not initialized, skipping artifact upload")
            return

        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return

        artifact = wandb.Artifact(
            name=self.artifact_name,
            type="model",
            description="",
            metadata={
                "epoch": epoch,
                # "obs_dim": 585,
                # "actor_in_dim": 574,
                "cond_dim": 11,          # gender + 10 betas
                "body_shapes": 128
            }
        )

        # Add the file with its original name so loading code stays simple
        artifact.add_file(checkpoint_path, name=os.path.basename(checkpoint_path))

        # This automatically creates a new version and updates the :latest alias
        wandb.log_artifact(artifact, aliases=["latest"])

        print(f"✅ Uploaded {os.path.basename(checkpoint_path)} → "
            f"https://wandb.ai/yugoamaryl/hhi/artifacts/model/hhi_film_model (latest)")

# global singleton — import and use anywhere
wandb_logger = WandbLogger()