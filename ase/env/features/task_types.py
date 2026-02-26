# env/features/task_types.py
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable
import torch


@runtime_checkable
class HumanoidTask(Protocol):
    # core config/runtime
    cfg: Mapping[str, Any]
    gym: Any
    sim: Any
    viewer: Any | None
    device: Any
    num_envs: int
    dt: float

    # tensors used by features
    _root_states: torch.Tensor
    _humanoid_actor_ids: torch.Tensor
    _humanoid_root_states: torch.Tensor

    # optional (only present in PHC / imitation tasks)
    _motion_lib: Any  # MotionLibSMPL-like
    _reset_ref_motion_ids: torch.Tensor
    _reset_ref_motion_times: torch.Tensor
