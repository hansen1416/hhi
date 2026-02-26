# env/features/viewer.py
from __future__ import annotations

import numpy as np
from isaacgym import gymapi

from env.features.base import Feature
from env.features.task_types import HumanoidTask


class FollowCameraFeature(Feature):
    """
    Follow-camera logic extracted from Humanoid._init_camera/_update_camera.

    - Initializes camera based on the root position of env 0.
    - On each render, keeps camera's relative XY offset to the character while preserving Z.
    """
    def __init__(
        self,
        enabled: bool = True,
        init_offset_xy: tuple[float, float] = (-1.0, -6.0),
        init_cam_z: float = 2.0,
        target_z: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self._init_offset_xy = init_offset_xy
        self._init_cam_z = init_cam_z
        self._target_z = target_z
        self._prev_char_pos: np.ndarray | None = None

    def on_post_init_tensors(self, task: HumanoidTask) -> None:
        if not self.enabled or task.viewer is None:
            return
        self._init_camera(task)

    def on_render(self, task: HumanoidTask) -> None:
        if not self.enabled or task.viewer is None:
            return

        self._update_camera(task)

    def _init_camera(self, task: HumanoidTask) -> None:
        task.gym.refresh_actor_root_state_tensor(task.sim)
        char_pos = task._humanoid_root_states[0, 0:3].cpu().numpy()
        self._prev_char_pos = char_pos.copy()

        cam_pos = gymapi.Vec3(
            char_pos[0] + self._init_offset_xy[0],
            char_pos[1] + self._init_offset_xy[1],
            self._init_cam_z,
        )
        cam_target = gymapi.Vec3(char_pos[0], char_pos[1], self._target_z)
        task.gym.viewer_camera_look_at(task.viewer, None, cam_pos, cam_target)

    def _update_camera(self, task: HumanoidTask) -> None:
        task.gym.refresh_actor_root_state_tensor(task.sim)
        char_pos = task._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = task.gym.get_viewer_camera_transform(task.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z], dtype=np.float32)
        cam_delta = cam_pos - self._prev_char_pos

        new_cam_target = gymapi.Vec3(char_pos[0], char_pos[1], self._target_z)
        new_cam_pos = gymapi.Vec3(char_pos[0] + cam_delta[0], char_pos[1] + cam_delta[1], cam_pos[2])

        task.gym.viewer_camera_look_at(task.viewer, None, new_cam_pos, new_cam_target)
        self._prev_char_pos[:] = char_pos



