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

from ..robot import ActionExecutionStatus, Robot
from .config_realman_pika import RealmanPikaConfig
from .controllers import PikaController, RealmanInterpolationController
from .transforms import (
    apply_realman_tcp_relative_pose,
    pika_relative_pose_to_realman_tcp_relative_pose,
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


def _pose_distance(start_pose: np.ndarray, end_pose: np.ndarray) -> tuple[float, float]:
    position_distance = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
    start_rotation = Rotation.from_rotvec(start_pose[3:6])
    end_rotation = Rotation.from_rotvec(end_pose[3:6])
    rotation_distance = float(np.linalg.norm((end_rotation * start_rotation.inv()).as_rotvec()))
    return position_distance, rotation_distance


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
        self._active_action_target: np.ndarray | None = None
        self._active_action_started_at: float | None = None
        self._active_action_timeout_s: float | None = None
        self._active_action_settle_count = 0

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
            state_read_retries=self.config.robot_state_read_retries,
            state_read_retry_delay=self.config.robot_state_read_retry_delay_sec,
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
        if self._reference_realman_tcp_pose is None:
            raise RuntimeError("RealmanPika reference TCP pose is not initialized.")
        realman_tcp_pose = self._read_realman_tcp_pose()
        realman_relative_pose = realman_tcp_relative_pose_between(
            self._reference_realman_tcp_pose, realman_tcp_pose
        )
        pika_relative_pose = realman_tcp_relative_pose_to_pika_relative_pose(realman_relative_pose)
        gripper_width = self._read_gripper_width()
        state = np.concatenate([pika_relative_pose, np.array([gripper_width], dtype=np.float64)])
        self._last_state_vector = state
        return state

    def _state_vector_from_realman_pose(
        self, realman_tcp_pose: np.ndarray, gripper_width: float
    ) -> np.ndarray:
        if self._reference_realman_tcp_pose is None:
            raise RuntimeError("RealmanPika reference TCP pose is not initialized.")
        realman_relative_pose = realman_tcp_relative_pose_between(
            self._reference_realman_tcp_pose, realman_tcp_pose
        )
        pika_relative_pose = realman_tcp_relative_pose_to_pika_relative_pose(realman_relative_pose)
        return np.concatenate([pika_relative_pose, np.array([gripper_width], dtype=np.float64)])

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

        current_realman_tcp_pose = self._read_realman_tcp_pose()
        target = self._clip_relative_action(self._action_to_vector(action))
        realman_relative_pose = pika_relative_pose_to_realman_tcp_relative_pose(target[:6])
        realman_target_pose = apply_realman_tcp_relative_pose(current_realman_tcp_pose, realman_relative_pose)

        current_state = None
        target_state = None
        timeout_s = None
        if self.config.progress_gate_enabled:
            current_gripper_width = self._read_gripper_width()
            current_state = self._state_vector_from_realman_pose(
                current_realman_tcp_pose, current_gripper_width
            )
            target_state = self._state_vector_from_realman_pose(realman_target_pose, target[6])
            pos_distance, rot_distance = _pose_distance(current_state[:6], target_state[:6])
            gripper_distance = abs(float(target_state[6] - current_state[6]))
            arm_duration = max(
                pos_distance / self.config.max_pos_speed,
                rot_distance / self.config.max_rot_speed,
            )
            gripper_speed_m_s = self.config.gripper_move_max_speed_mm_s / 1000.0
            gripper_duration = gripper_distance / gripper_speed_m_s
            timeout_s = max(
                arm_duration + float(self.config.robot_action_latency),
                gripper_duration + float(self.config.gripper_action_latency),
            )

        now = time.time()
        self.arm.schedule_waypoint(
            realman_target_pose, target_time=now + float(self.config.robot_action_latency)
        )
        self.gripper.schedule_waypoint(target[6], target_time=now + float(self.config.gripper_action_latency))

        if self.config.progress_gate_enabled:
            assert target_state is not None
            assert timeout_s is not None
            self._active_action_target = target_state
            self._active_action_started_at = time.monotonic()
            self._active_action_timeout_s = max(
                self.config.progress_min_timeout_s,
                timeout_s + self.config.progress_timeout_margin_s,
            )
            self._active_action_settle_count = 0
        return {key: float(value) for key, value in zip(STATE_ACTION_KEYS, target, strict=True)}

    @property
    def requires_action_acknowledgement(self) -> bool:
        return self.config.progress_gate_enabled

    def get_action_execution_status(
        self, observation: RobotObservation | None = None
    ) -> ActionExecutionStatus:
        target = self._active_action_target
        started_at = self._active_action_started_at
        timeout_s = self._active_action_timeout_s
        if not self.config.progress_gate_enabled or target is None or started_at is None:
            return ActionExecutionStatus(active=False)

        if observation is None:
            current = self._read_state_vector()
        else:
            missing = [key for key in STATE_ACTION_KEYS if key not in observation]
            if missing:
                raise ValueError(f"Missing RealmanPika progress observation keys: {missing}.")
            current = np.array(
                [_as_latest_float(observation[key]) for key in STATE_ACTION_KEYS],
                dtype=np.float64,
            )

        position_error, rotation_error = _pose_distance(current[:6], target[:6])
        gripper_error = abs(float(current[6] - target[6]))
        within_tolerance = (
            position_error <= self.config.progress_position_tolerance_m
            and rotation_error <= self.config.progress_rotation_tolerance_rad
            and (
                not self.config.progress_require_gripper_target
                or gripper_error <= self.config.progress_gripper_tolerance_m
            )
        )
        self._active_action_settle_count = self._active_action_settle_count + 1 if within_tolerance else 0
        reached = self._active_action_settle_count >= self.config.progress_settle_samples
        elapsed_s = time.monotonic() - started_at
        timed_out = not reached and timeout_s is not None and elapsed_s >= timeout_s
        return ActionExecutionStatus(
            active=True,
            reached=reached,
            timed_out=timed_out,
            elapsed_s=elapsed_s,
            timeout_s=timeout_s,
            position_error=position_error,
            rotation_error=rotation_error,
            gripper_error=gripper_error,
        )

    def acknowledge_action_execution(self) -> None:
        self._active_action_target = None
        self._active_action_started_at = None
        self._active_action_timeout_s = None
        self._active_action_settle_count = 0

    @check_if_not_connected
    def hold_position(self) -> None:
        if self.arm is None or self.gripper is None:
            raise RuntimeError("RealmanPika is not connected.")
        current_pose = self._read_realman_tcp_pose()
        current_gripper_width = self._read_gripper_width()
        self.arm.servol(current_pose, duration=0.0)
        self.gripper.schedule_waypoint(current_gripper_width, target_time=time.time())
        self.acknowledge_action_execution()

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
        self.acknowledge_action_execution()

    @check_if_not_connected
    def disconnect(self) -> None:
        self._disconnect_best_effort()
        logger.info("%s disconnected.", self)
