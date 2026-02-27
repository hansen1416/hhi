# ase/learning/wandb_logger.py
"""
Wandb logger utility for the HHI project (PHC + HUMOS 128 variations).
Hardcoded settings — no config needed.
"""

import wandb
import os
from datetime import datetime


class WandbLogger:
    def __init__(self):
        self.enabled = True                     # ← change to False to disable globally
        self.project = "hhi"                    # your project name on wandb
        self.entity = None                      # set to your username/team if needed
        self.log_every = 1                      # log every N epochs
        self.run = None

    def init(self, config_dict=None):
        """Call once at the start of training. run_name is auto-generated if None."""
        if not self.enabled:
            print("wandb_logger: disabled")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"hii_film_{timestamp}"

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


# global singleton — import and use anywhere
wandb_logger = WandbLogger()