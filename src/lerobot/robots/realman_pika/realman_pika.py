#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.rotation import Rotation

from ..robot import Robot
from .config_realman_pika import RealmanPikaConfig
from .controllers import PikaController, RealmanInterpolationController
from .transforms import (
    apply_realman_tcp_relative_pose,
    pika_gripper_pose_to_realman_tcp_pose,
    pika_relative_pose_to_realman_tcp_relative_pose,
    realman_tcp_pose_to_pika_gripper_pose,
    realman_tcp_relative_pose_between,
    realman_tcp_relative_pose_to_pika_relative_pose,
)

logger = logging.getLogger(__name__)

EEF_X = "eef_x.pos"
EEF_Y = "eef_y.pos"
EEF_Z = "eef_z.pos"
EEF_RX = "eef_rx.pos"
EEF_RY = "eef_ry.pos"
EEF_RZ = "eef_rz.pos"
GRIPPER = "gripper.pos"
STATE_ACTION_KEYS = (EEF_X, EEF_Y, EEF_Z, EEF_RX, EEF_RY, EEF_RZ, GRIPPER)


def _lift_pika_gripper_above_table(
    pika_gripper_pose: np.ndarray,
    gripper_width: float,
    table_height: float,
    finger_thickness: float,
) -> tuple[np.ndarray, float]:
    """Lift a Pika gripper pose until all four finger corners clear the table."""
    pose = np.asarray(pika_gripper_pose, dtype=np.float64).copy()
    keypoints = np.array(
        [
            [dx * gripper_width / 2.0, dy * finger_thickness / 2.0, 0.0]
            for dx in (-1.0, 1.0)
            for dy in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    rotation = Rotation.from_rotvec(pose[3:6]).as_matrix()
    transformed_keypoints = (rotation @ keypoints.T).T + pose[:3]
    lift = max(float(table_height - np.min(transformed_keypoints[:, 2])), 0.0)
    pose[2] += lift
    return pose, lift


def _as_latest_float(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return float(arr)
    return float(arr.reshape(-1)[-1])


def _as_pose(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (6,):
        return arr
    if arr.ndim >= 2 and arr.shape[-1] == 6:
        return arr.reshape(-1, 6)[-1]
    raise ValueError(f"Expected pose with trailing dimension 6, got {arr.shape}.")


class RealmanPika(Robot):
    config_class = RealmanPikaConfig
    name = "realman_pika"

    def __init__(self, config: RealmanPikaConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs({} if config.disable_cameras_on_connect else config.cameras)
        self.arm: RealmanInterpolationController | None = None
        self.gripper: PikaController | None = None
        self._last_state_vector: np.ndarray | None = None
        self._reference_realman_tcp_pose: np.ndarray | None = None
        self._action_reference_realman_tcp_pose: np.ndarray | None = None

    @cached_property
    def _state_features(self) -> dict[str, type]:
        return dict.fromkeys(STATE_ACTION_KEYS, float)

    @cached_property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cam.height, cam.width, 3) for name, cam in self.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_features, **self._camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_features

    @property
    def is_connected(self) -> bool:
        arm_ready = self.arm is not None and self.arm.is_ready
        gripper_ready = self.gripper is not None and self.gripper.is_ready
        cameras_ready = all(cam.is_connected for cam in self.cameras.values())
        return arm_ready and gripper_ready and cameras_ready

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.arm = RealmanInterpolationController(
            robot_ip=self.config.robot_ip,
            robot_port=self.config.robot_port,
            level=self.config.robot_level,
            mode=self.config.robot_mode,
            frequency=self.config.robot_command_frequency,
            max_pos_speed=self.config.max_pos_speed,
            max_rot_speed=self.config.max_rot_speed,
            launch_timeout=self.config.robot_launch_timeout,
            joint_dim=self.config.robot_joint_dim,
            command_mode=self.config.robot_command_mode,
            interpolation_mode=self.config.robot_interpolation_mode,
            canfd_follow=self.config.robot_canfd_follow,
            canfd_trajectory_mode=self.config.robot_canfd_trajectory_mode,
            canfd_radio=self.config.robot_canfd_radio,
            state_read_retries=self.config.robot_state_read_retries,
            state_read_retry_delay=self.config.robot_state_read_retry_delay_sec,
            max_consecutive_command_failures=self.config.robot_max_consecutive_command_failures,
            max_consecutive_state_read_failures=self.config.robot_max_consecutive_state_read_failures,
            use_udp_state=self.config.robot_use_udp_state,
            udp_port=self.config.robot_udp_port,
            udp_cycle=self.config.robot_udp_cycle,
            udp_target_ip=self.config.robot_udp_target_ip,
            udp_state_timeout=self.config.robot_udp_state_timeout_sec,
            realman_sdk_root=self.config.realman_sdk_root,
            realman_rm_api_dir=self.config.realman_rm_api_dir,
        )
        self.gripper = PikaController(
            serial_port=self.config.gripper_serial_port,
            frequency=self.config.gripper_frequency,
            move_max_speed=self.config.gripper_move_max_speed_mm_s,
            launch_timeout=self.config.gripper_launch_timeout,
            use_meters=True,
            min_width=self.config.gripper_min_width_m,
            max_width=self.config.gripper_max_width_m,
            init_velocity=self.config.gripper_init_velocity,
            pika_sdk_root=self.config.pika_sdk_root,
        )

        try:
            for cam in self.cameras.values():
                cam.connect()
            self.arm.start(wait=True)
            self.gripper.start(wait=True)
            self.configure()
            self._reference_realman_tcp_pose = self._read_realman_tcp_pose()
            self._last_state_vector = self._read_state_vector()
        except Exception:
            self._disconnect_best_effort()
            raise

        logger.info("%s connected.", self)

    def get_connection_error_details(self) -> str | None:
        failures = []
        for name, controller in (("RealMan arm", self.arm), ("Pika gripper", self.gripper)):
            if controller is None or controller.is_alive():
                continue
            process_error = controller.get_process_error()
            if process_error:
                failures.append(f"{name} controller exited:\n{process_error}")
            else:
                failures.append(f"{name} controller exited without a reported traceback.")
        return "\n".join(failures) or None

    def configure(self) -> None:
        return None

    def calibrate(self) -> None:
        return None

    def _read_realman_tcp_pose(self) -> np.ndarray:
        if self.arm is None:
            raise RuntimeError("RealmanPika is not connected.")
        arm_state = self.arm.get_state()
        return _as_pose(arm_state["ActualTCPPose"])

    def _read_gripper_width(self) -> float:
        if self.gripper is None:
            raise RuntimeError("RealmanPika is not connected.")
        gripper_state = self.gripper.get_state()
        return _as_latest_float(gripper_state["gripper_position"])

    def _read_state_vector(self) -> np.ndarray:
        realman_tcp_pose = self._read_realman_tcp_pose()
        gripper_width = self._read_gripper_width()
        state = np.concatenate([realman_tcp_pose, np.array([gripper_width], dtype=np.float64)])
        self._last_state_vector = state
        return state

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._read_state_vector()
        observation: RobotObservation = {
            key: float(value) for key, value in zip(STATE_ACTION_KEYS, state, strict=True)
        }

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            observation[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug("%s read %s: %.1fms", self, cam_key, dt_ms)

        return observation

    def _action_to_vector(self, action: RobotAction) -> np.ndarray:
        missing = [key for key in STATE_ACTION_KEYS if key not in action]
        if missing:
            raise ValueError(f"Missing RealmanPika action keys: {missing}.")
        return np.array([float(action[key]) for key in STATE_ACTION_KEYS], dtype=np.float64)

    @check_if_not_connected
    def set_action_reference_to_current_pose(self) -> None:
        """Freeze the current TCP pose as the origin for subsequent relative actions."""
        self._action_reference_realman_tcp_pose = self._read_realman_tcp_pose().copy()

    @check_if_not_connected
    def set_action_reference_from_realman_tcp_pose(self, state: np.ndarray) -> None:
        """Use an absolute RealMan TCP pose as the origin for subsequent relative actions."""
        if not isinstance(state, np.ndarray):
            raise TypeError(f"Expected state to be a numpy.ndarray, got {type(state).__name__}.")
        if state.shape != (6,):
            raise ValueError(f"Expected state shape (6,), got {state.shape}.")
        if not np.issubdtype(state.dtype, np.number):
            raise TypeError(f"Expected state to have a numeric dtype, got {state.dtype}.")
        if not np.isfinite(state).all():
            raise ValueError("Expected state to contain only finite values.")
        self._action_reference_realman_tcp_pose = state.astype(np.float64, copy=True)

    @check_if_not_connected
    def set_action_reference_from_state(self, state: Any) -> None:
        """Freeze the TCP origin represented by an ``observation.state`` snapshot."""
        # Raise error when the robot is not connected successfully
        if self._reference_realman_tcp_pose is None:
            raise RuntimeError("RealmanPika reference TCP pose is not initialized.")

        state_vector = np.asarray(state, dtype=np.float64).reshape(-1)
        if state_vector.size != len(STATE_ACTION_KEYS):
            raise ValueError(
                f"Expected observation.state with {len(STATE_ACTION_KEYS)} values, "
                f"got shape {np.asarray(state).shape}."
            )
        realman_relative_pose = pika_relative_pose_to_realman_tcp_relative_pose(state_vector[:6])
        self._action_reference_realman_tcp_pose = apply_realman_tcp_relative_pose(
            self._reference_realman_tcp_pose, realman_relative_pose
        )

    def clear_action_reference(self) -> None:
        """Clear the TCP origin required by subsequent relative actions."""
        self._action_reference_realman_tcp_pose = None

    def _clip_relative_action(self, action: np.ndarray) -> np.ndarray:
        clipped = action.copy()
        clipped[:3] = np.clip(clipped[:3], -self.config.max_relative_pos, self.config.max_relative_pos)
        clipped[3:6] = np.clip(clipped[3:6], -self.config.max_relative_rot, self.config.max_relative_rot)
        clipped[6] = np.clip(clipped[6], self.config.gripper_min_width_m, self.config.gripper_max_width_m)
        return clipped

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.arm is None or self.gripper is None:
            raise RuntimeError("RealmanPika is not connected.")

        action_reference_pose = self._action_reference_realman_tcp_pose
        if action_reference_pose is None:
            raise RuntimeError(
                "Action reference is not set. Set a RealMan TCP reference before sending actions."
            )
        target = self._clip_relative_action(self._action_to_vector(action))
        realman_relative_pose = pika_relative_pose_to_realman_tcp_relative_pose(target[:6])
        realman_target_pose = apply_realman_tcp_relative_pose(action_reference_pose, realman_relative_pose)

        if self.config.table_collision_enabled:
            pika_target_pose = realman_tcp_pose_to_pika_gripper_pose(realman_target_pose)
            pika_target_pose, lift = _lift_pika_gripper_above_table(
                pika_target_pose,
                gripper_width=float(target[6]),
                table_height=self.config.table_height_m,
                finger_thickness=self.config.gripper_finger_thickness_m,
            )
            if lift > 0:
                realman_target_pose = pika_gripper_pose_to_realman_tcp_pose(pika_target_pose)
                corrected_realman_relative = realman_tcp_relative_pose_between(
                    action_reference_pose, realman_target_pose
                )
                target[:6] = realman_tcp_relative_pose_to_pika_relative_pose(corrected_realman_relative)
                logger.warning(
                    "Table collision guard lifted Pika gripper target by %.4fm (table_height=%.4fm)",
                    lift,
                    self.config.table_height_m,
                )

        now = time.time()
        self.arm.schedule_waypoint(
            realman_target_pose, target_time=now + float(self.config.robot_action_latency)
        )
        self.gripper.schedule_waypoint(target[6], target_time=now + float(self.config.gripper_action_latency))

        self._last_state_vector = target
        return {key: float(value) for key, value in zip(STATE_ACTION_KEYS, target, strict=True)}

    def _disconnect_best_effort(self) -> None:
        for controller in (self.gripper, self.arm):
            if controller is not None:
                with contextlib.suppress(Exception):
                    controller.stop(wait=True)
        for cam in self.cameras.values():
            with contextlib.suppress(Exception):
                if cam.is_connected:
                    cam.disconnect()
        self.arm = None
        self.gripper = None
        self._reference_realman_tcp_pose = None
        self._action_reference_realman_tcp_pose = None

    @check_if_not_connected
    def disconnect(self) -> None:
        self._disconnect_best_effort()
        logger.info("%s disconnected.", self)
