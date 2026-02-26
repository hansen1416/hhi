# env/features/base.py
from __future__ import annotations
from abc import ABC

from env.features.task_types import HumanoidTask

class Feature(ABC):
    """Simple no-op hook base class."""
    enabled: bool = True

    def on_create_envs(self, task:HumanoidTask, num_envs: int) -> None:
        pass

    def on_humanoid_actor_created(self, task:HumanoidTask, env_id: int, env_ptr) -> None:
        pass

    def on_post_init_tensors(self, task:HumanoidTask) -> None:
        """Called after task._root_states / actor ids are ready."""
        pass

    def on_reset_envs(self, task:HumanoidTask, env_ids, key_pos) -> None:
        pass

    def on_post_physics_step(self, task:HumanoidTask, key_pos) -> None:
        pass

    def on_render(self, task:HumanoidTask, sync_frame_time: bool=False) -> None:
        pass