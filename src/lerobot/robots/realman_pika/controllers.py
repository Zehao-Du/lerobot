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

from __future__ import annotations

import enum
import logging
import multiprocessing as mp
import os
import queue
import socket
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Iterable
from ctypes import CFUNCTYPE
from pathlib import Path
from typing import Any

import numpy as np

from .trajectory import (
    PoseTrajectoryInterpolator,
    realman_euler_pose_to_rotvec_pose,
    rotvec_pose_to_realman_euler_pose,
)

logger = logging.getLogger(__name__)


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic) -> None:
    t_now = time_func()
    if t_end - t_now > slack_time:
        time.sleep(t_end - t_now - slack_time)
    while time_func() < t_end:
        pass


def _append_sys_paths(paths: Iterable[str | Path | None]) -> None:
    for path in paths:
        if path is None:
            continue
        path_str = str(path)
        if path_str and path_str not in sys.path:
            sys.path.append(path_str)


def _check_vector(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape:
        raise RuntimeError(f"Expected {name} shape {shape}, got {value.shape}.")
    return value


def _infer_local_ip_for_target(target_ip: str) -> str:
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((target_ip, 9))
        return sock.getsockname()[0]
    except OSError as e:
        raise RuntimeError(
            f"Failed to infer local UDP target IP for RealMan robot {target_ip}. "
            "Pass robot_udp_target_ip explicitly."
        ) from e
    finally:
        if sock is not None:
            sock.close()


class _LatestStateMixin:
    def _init_state_cache(self, maxlen: int) -> None:
        self._state_cache: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def _drain_state_queue(self) -> None:
        while True:
            try:
                state = self.state_queue.get_nowait()
            except queue.Empty:
                return
            self._state_cache.append(state)

    def get_state(self, k: int | None = None) -> dict[str, Any]:
        self._drain_state_queue()
        if not self._state_cache:
            raise RuntimeError(f"{self.__class__.__name__} has not received state yet.")
        if k is None:
            return self._state_cache[-1]
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}.")
        states = list(self._state_cache)[-k:]
        return _stack_state_dicts(states)

    def get_all_state(self) -> dict[str, Any]:
        self._drain_state_queue()
        if not self._state_cache:
            raise RuntimeError(f"{self.__class__.__name__} has not received state yet.")
        return _stack_state_dicts(list(self._state_cache))


def _stack_state_dicts(states: list[dict[str, Any]]) -> dict[str, Any]:
    if len(states) == 1:
        return states[0]
    out: dict[str, Any] = {}
    for key in states[0]:
        out[key] = np.stack([state[key] for state in states], axis=0)
    return out


class RealmanCommand(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class RealmanInterpolationController(mp.Process, _LatestStateMixin):
    """Small RealMan pose controller with lazy SDK import.

    Parent process API matches the subset used by ``RealmanPika``:
    ``start()``, ``stop()``, ``schedule_waypoint()``, and ``get_state()``.
    """

    def __init__(
        self,
        *,
        robot_ip: str,
        robot_port: int,
        level: int = 3,
        mode: int = 2,
        frequency: int = 125,
        max_pos_speed: float = 0.15,
        max_rot_speed: float = 0.4,
        launch_timeout: float = 15.0,
        joint_dim: int = 7,
        command_mode: str = "movep_canfd",
        interpolation_mode: str = "trajectory",
        state_read_retries: int = 3,
        state_read_retry_delay: float = 0.01,
        max_consecutive_state_read_failures: int = 10,
        use_udp_state: bool = True,
        udp_port: int = 8888,
        udp_cycle: int = 2,
        udp_target_ip: str | None = None,
        udp_state_timeout: float = 0.2,
        realman_sdk_root: str | Path | None = None,
        realman_rm_api_dir: str | Path | None = None,
        get_max_k: int | None = None,
        verbose: bool = False,
    ):
        if not robot_ip:
            raise ValueError("RealmanInterpolationController requires robot_ip.")
        if robot_port <= 0:
            raise ValueError(f"robot_port must be > 0, got {robot_port}.")
        if not 0 < frequency <= 500:
            raise ValueError(f"frequency must be in (0, 500], got {frequency}.")
        if max_pos_speed <= 0:
            raise ValueError(f"max_pos_speed must be > 0, got {max_pos_speed}.")
        if max_rot_speed <= 0:
            raise ValueError(f"max_rot_speed must be > 0, got {max_rot_speed}.")
        if joint_dim <= 0:
            raise ValueError(f"joint_dim must be > 0, got {joint_dim}.")
        if command_mode not in ("movep_canfd", "movep_follow", "movel", "movej_p"):
            raise ValueError(f"Unsupported command_mode: {command_mode!r}.")
        if interpolation_mode not in ("trajectory", "none"):
            raise ValueError(f"Unsupported interpolation_mode: {interpolation_mode!r}.")
        if use_udp_state and udp_target_ip is None:
            udp_target_ip = _infer_local_ip_for_target(robot_ip)

        super().__init__(name="RealmanInterpolationController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.level = level
        self.mode = mode
        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.joint_dim = joint_dim
        self.command_mode = command_mode
        self.interpolation_mode = interpolation_mode
        self.state_read_retries = state_read_retries
        self.state_read_retry_delay = state_read_retry_delay
        self.max_consecutive_state_read_failures = max_consecutive_state_read_failures
        self.use_udp_state = use_udp_state
        self.udp_port = udp_port
        self.udp_cycle = udp_cycle
        self.udp_target_ip = udp_target_ip
        self.udp_state_timeout = udp_state_timeout
        self.realman_sdk_root = realman_sdk_root
        self.realman_rm_api_dir = realman_rm_api_dir
        self.verbose = verbose

        self.command_queue: mp.Queue = mp.Queue(maxsize=256)
        self.state_queue: mp.Queue = mp.Queue(maxsize=1024)
        self.error_queue: mp.Queue = mp.Queue()
        self.ready_event = mp.Event()
        self._init_state_cache(maxlen=get_max_k or int(frequency * 5))

    def start(self, wait: bool = True) -> None:
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait: bool = True) -> None:
        self.command_queue.put({"cmd": RealmanCommand.STOP.value})
        if wait:
            self.stop_wait()

    def start_wait(self) -> None:
        deadline = time.monotonic() + self.launch_timeout
        while time.monotonic() < deadline:
            try:
                message = self.error_queue.get_nowait()
            except queue.Empty:
                message = None
            if message is not None:
                raise RuntimeError(f"RealMan controller process failed during startup:\n{message}")
            if self.ready_event.is_set():
                if not self.is_alive():
                    raise RuntimeError("RealMan controller process exited before becoming ready.")
                return
            if not self.is_alive():
                raise RuntimeError("RealMan controller process exited before becoming ready.")
            time.sleep(0.05)
        raise RuntimeError(f"RealMan controller did not become ready within {self.launch_timeout:.1f}s.")

    def stop_wait(self) -> None:
        self.join(timeout=self.launch_timeout)
        if self.is_alive():
            self.terminate()
            self.join(timeout=1.0)

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set() and self.is_alive()

    def schedule_waypoint(self, pose: np.ndarray, target_time: float) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,):
            raise ValueError(f"Expected RealMan waypoint shape (6,), got {pose.shape}.")
        self.command_queue.put(
            {
                "cmd": RealmanCommand.SCHEDULE_WAYPOINT.value,
                "target_pose": pose,
                "target_time": float(target_time),
            }
        )

    def servol(self, pose: np.ndarray, duration: float) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,):
            raise ValueError(f"Expected RealMan servol pose shape (6,), got {pose.shape}.")
        if duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}.")
        self.command_queue.put(
            {
                "cmd": RealmanCommand.SERVOL.value,
                "target_pose": pose,
                "duration": float(duration),
            }
        )

    def _import_realman_sdk(self):
        _append_sys_paths([self.realman_sdk_root, self.realman_rm_api_dir])
        try:
            from Robotic_Arm.rm_robot_interface import (  # type: ignore
                RoboticArm,
                rm_realtime_arm_joint_state_t,
                rm_realtime_push_config_t,
                rm_thread_mode_e,
            )
        except ImportError as e:
            raise ImportError(
                "Missing RealMan SDK. Set robot.realman_sdk_root / robot.realman_rm_api_dir "
                "or install Robotic_Arm in the active environment."
            ) from e
        return RoboticArm, rm_realtime_arm_joint_state_t, rm_realtime_push_config_t, rm_thread_mode_e

    def _connect_robot(self, RoboticArm, rm_thread_mode_e):
        robot = RoboticArm(rm_thread_mode_e(self.mode))
        handle = robot.rm_create_robot_arm(self.robot_ip, self.robot_port, self.level)
        if handle.id == -1:
            raise RuntimeError(f"Failed to connect to RealMan arm at {self.robot_ip}:{self.robot_port}.")
        return robot

    def _setup_udp_state(self, robot, rm_realtime_arm_joint_state_t, rm_realtime_push_config_t) -> None:
        self._udp_state_lock = threading.Lock()
        self._last_udp_state = None
        self._last_udp_state_time = None

        def callback(state_data):
            waypoint = state_data.waypoint
            pose = np.array(
                [
                    waypoint.position.x,
                    waypoint.position.y,
                    waypoint.position.z,
                    waypoint.euler.rx,
                    waypoint.euler.ry,
                    waypoint.euler.rz,
                ],
                dtype=np.float64,
            )
            joint_deg = np.asarray(state_data.joint_status.joint_position, dtype=np.float64)
            joint = np.deg2rad(joint_deg[: self.joint_dim])
            with self._udp_state_lock:
                self._last_udp_state = (pose, joint)
                self._last_udp_state_time = time.time()

        config = rm_realtime_push_config_t(
            cycle=self.udp_cycle,
            enable=True,
            port=self.udp_port,
            force_coordinate=-1,
            ip=self.udp_target_ip,
        )
        ret = robot.rm_set_realtime_push(config)
        if ret != 0:
            raise RuntimeError(f"Failed to enable RealMan UDP realtime push, error code: {ret}.")

        callback_type = CFUNCTYPE(None, rm_realtime_arm_joint_state_t)
        self._udp_callback = callback_type(callback)
        self._udp_callback_ref = self._udp_callback
        robot.rm_realtime_arm_state_call_back(self._udp_callback)

    def _get_udp_arm_state(self) -> tuple[np.ndarray, np.ndarray]:
        deadline = time.time() + self.udp_state_timeout
        while time.time() < deadline:
            with self._udp_state_lock:
                state = self._last_udp_state
                state_time = self._last_udp_state_time
            if state is not None and state_time is not None and time.time() - state_time <= self.udp_state_timeout:
                pose, joint = state
                return _check_vector("udp_pose", pose, (6,)), _check_vector(
                    "udp_joint", joint, (self.joint_dim,)
                )
            time.sleep(0.001)
        raise RuntimeError(f"Timed out waiting for RealMan UDP state for {self.udp_state_timeout:.3f}s.")

    def _get_tcp_arm_state(self, robot) -> tuple[np.ndarray, np.ndarray]:
        last_ret = None
        for attempt in range(self.state_read_retries):
            ret, raw_state = robot.rm_get_current_arm_state()
            if ret == 0:
                return _check_vector("pose", raw_state["pose"], (6,)), _check_vector(
                    "joint", raw_state["joint"], (self.joint_dim,)
                )
            last_ret = ret
            if attempt < self.state_read_retries - 1 and self.state_read_retry_delay > 0:
                time.sleep(self.state_read_retry_delay)
        raise RuntimeError(
            f"Failed to get RealMan arm state after {self.state_read_retries} attempts, "
            f"last error code: {last_ret}."
        )

    def _get_arm_state(self, robot) -> tuple[np.ndarray, np.ndarray]:
        if self.use_udp_state:
            return self._get_udp_arm_state()
        return self._get_tcp_arm_state(robot)

    def _send_pose_command(self, robot, pose: np.ndarray) -> int:
        pose_list = np.asarray(pose, dtype=np.float64).tolist()
        if self.command_mode == "movep_canfd":
            return robot.rm_movep_canfd(pose_list, follow=True, trajectory_mode=0, radio=0)
        if self.command_mode == "movep_follow":
            return robot.rm_movep_follow(pose_list)
        if self.command_mode == "movej_p":
            return robot.rm_movej_p(pose_list, 5, 0, 1, 0)
        return robot.rm_movel(pose_list, 5, 0, 1, 0)

    def _put_state(
        self,
        actual_pose: np.ndarray,
        actual_joint: np.ndarray,
        target_pose: np.ndarray,
        prev_pose: np.ndarray | None,
        prev_joint: np.ndarray | None,
        prev_time: float | None,
        timestamp: float,
    ) -> None:
        tcp_speed = np.zeros((6,), dtype=np.float64)
        joint_vel = np.zeros((self.joint_dim,), dtype=np.float64)
        if prev_pose is not None and prev_joint is not None and prev_time is not None:
            dt = max(timestamp - prev_time, 1e-6)
            tcp_speed = (actual_pose - prev_pose) / dt
            joint_vel = (actual_joint - prev_joint) / dt
        state = {
            "ActualTCPPose": actual_pose,
            "ActualTCPSpeed": tcp_speed,
            "ActualQ": actual_joint,
            "ActualQd": joint_vel,
            "TargetTCPPose": target_pose,
            "TargetTCPSpeed": np.zeros((6,), dtype=np.float64),
            "TargetQ": actual_joint,
            "TargetQd": np.zeros((self.joint_dim,), dtype=np.float64),
            "robot_receive_timestamp": timestamp,
            "robot_timestamp": timestamp,
        }
        try:
            self.state_queue.put_nowait(state)
        except queue.Full:
            try:
                self.state_queue.get_nowait()
            except queue.Empty:
                pass
            self.state_queue.put_nowait(state)

    def run(self) -> None:
        if hasattr(os, "sched_setscheduler"):
            pass
        robot = None
        try:
            RoboticArm, rm_realtime_arm_joint_state_t, rm_realtime_push_config_t, rm_thread_mode_e = (
                self._import_realman_sdk()
            )
            robot = self._connect_robot(RoboticArm, rm_thread_mode_e)
            if self.use_udp_state:
                self._setup_udp_state(robot, rm_realtime_arm_joint_state_t, rm_realtime_push_config_t)

            curr_pose, curr_joint = self._get_arm_state(robot)
            curr_interp_pose = realman_euler_pose_to_rotvec_pose(curr_pose)
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_interp_pose])
            direct_target_pose = curr_pose.copy()
            pending_direct_waypoints: list[tuple[float, np.ndarray]] = []
            prev_pose = None
            prev_joint = None
            prev_time = None
            consecutive_failures = 0
            dt = 1.0 / self.frequency
            iter_idx = 0
            t_start = time.monotonic()
            keep_running = True

            while keep_running:
                t_now = time.monotonic()

                if self.interpolation_mode == "none":
                    while pending_direct_waypoints and pending_direct_waypoints[0][0] <= t_now:
                        _, direct_target_pose = pending_direct_waypoints.pop(0)
                    command_pose = direct_target_pose
                else:
                    command_pose_interp = pose_interp(t_now)
                    command_pose = rotvec_pose_to_realman_euler_pose(command_pose_interp)
                ret = self._send_pose_command(robot, command_pose)
                if ret != 0:
                    raise RuntimeError(f"RealMan pose command failed with error code: {ret}.")

                recv_time = time.time()
                try:
                    curr_pose, curr_joint = self._get_arm_state(robot)
                except RuntimeError as e:
                    consecutive_failures += 1
                    if self.verbose:
                        print(
                            "[RealmanInterpolationController] "
                            f"state read failed {consecutive_failures}/"
                            f"{self.max_consecutive_state_read_failures}: {e}"
                        )
                    if consecutive_failures >= self.max_consecutive_state_read_failures:
                        raise RuntimeError(
                            f"RealMan state read failed {consecutive_failures} consecutive times."
                        ) from e
                else:
                    consecutive_failures = 0
                    self._put_state(
                        curr_pose,
                        curr_joint,
                        command_pose,
                        prev_pose,
                        prev_joint,
                        prev_time,
                        recv_time,
                    )
                    prev_pose = curr_pose
                    prev_joint = curr_joint
                    prev_time = recv_time

                commands = []
                while True:
                    try:
                        commands.append(self.command_queue.get_nowait())
                    except queue.Empty:
                        break

                for command in commands:
                    if command["cmd"] == RealmanCommand.STOP.value:
                        keep_running = False
                        break
                    if command["cmd"] == RealmanCommand.SERVOL.value:
                        target_pose = np.asarray(command["target_pose"], dtype=np.float64)
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        if self.interpolation_mode == "none":
                            pending_direct_waypoints.append((t_insert, target_pose))
                            pending_direct_waypoints.sort(key=lambda x: x[0])
                        else:
                            target_interp_pose = realman_euler_pose_to_rotvec_pose(target_pose)
                            pose_interp = pose_interp.drive_to_waypoint(
                                pose=target_interp_pose,
                                time=t_insert,
                                curr_time=curr_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed,
                            )
                            last_waypoint_time = t_insert
                        if self.verbose:
                            print(
                                "[RealmanInterpolationController] "
                                f"New pose target:{target_pose} duration:{duration}s"
                            )
                    elif command["cmd"] == RealmanCommand.SCHEDULE_WAYPOINT.value:
                        target_pose = np.asarray(command["target_pose"], dtype=np.float64)
                        target_time = float(command["target_time"])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        if self.interpolation_mode == "none":
                            if target_time <= curr_time:
                                direct_target_pose = target_pose
                            else:
                                pending_direct_waypoints.append((target_time, target_pose))
                                pending_direct_waypoints.sort(key=lambda x: x[0])
                        else:
                            target_interp_pose = realman_euler_pose_to_rotvec_pose(target_pose)
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_interp_pose,
                                time=target_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time,
                            )
                            last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                iter_idx += 1
                precise_wait(t_start + iter_idx * dt, time_func=time.monotonic)
                if iter_idx == 1:
                    self.ready_event.set()

        except BaseException:
            self.error_queue.put(traceback.format_exc())
            self.ready_event.set()
            raise
        finally:
            if robot is not None:
                robot.rm_delete_robot_arm()


class PikaCommand(enum.Enum):
    STOP = 0
    SCHEDULE_WAYPOINT = 1


class PikaController(mp.Process, _LatestStateMixin):
    def __init__(
        self,
        *,
        serial_port: str,
        frequency: int = 30,
        move_max_speed: float = 200.0,
        get_max_k: int | None = None,
        launch_timeout: float = 3.0,
        receive_latency: float = 0.0,
        use_meters: bool = True,
        min_width: float = 0.0,
        max_width: float = 0.09,
        init_velocity: float = 0.1,
        pika_sdk_root: str | Path | None = None,
        verbose: bool = False,
    ):
        if not serial_port:
            raise ValueError("PikaController requires serial_port.")
        if frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {frequency}.")
        if min_width >= max_width:
            raise ValueError(f"min_width must be smaller than max_width, got {min_width} >= {max_width}.")

        super().__init__(name="PikaController")
        self.serial_port = serial_port
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.min_width = min_width * self.scale if use_meters else min_width
        self.max_width = max_width * self.scale if use_meters else max_width
        self.init_velocity = init_velocity
        self.pika_sdk_root = pika_sdk_root
        self.verbose = verbose

        self.command_queue: mp.Queue = mp.Queue(maxsize=1024)
        self.state_queue: mp.Queue = mp.Queue(maxsize=1024)
        self.error_queue: mp.Queue = mp.Queue()
        self.ready_event = mp.Event()
        self._init_state_cache(maxlen=get_max_k or int(frequency * 10))

    def start(self, wait: bool = True) -> None:
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait: bool = True) -> None:
        self.command_queue.put({"cmd": PikaCommand.STOP.value})
        if wait:
            self.stop_wait()

    def start_wait(self) -> None:
        deadline = time.monotonic() + self.launch_timeout
        while time.monotonic() < deadline:
            try:
                message = self.error_queue.get_nowait()
            except queue.Empty:
                message = None
            if message is not None:
                raise RuntimeError(f"Pika controller process failed during startup:\n{message}")
            if self.ready_event.is_set():
                if not self.is_alive():
                    raise RuntimeError("Pika controller process exited before becoming ready.")
                return
            if not self.is_alive():
                raise RuntimeError("Pika controller process exited before becoming ready.")
            time.sleep(0.02)
        raise RuntimeError(f"Pika controller did not become ready within {self.launch_timeout:.1f}s.")

    def stop_wait(self) -> None:
        self.join(timeout=self.launch_timeout)
        if self.is_alive():
            self.terminate()
            self.join(timeout=1.0)

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set() and self.is_alive()

    def schedule_waypoint(self, pos: float | np.ndarray, target_time: float) -> None:
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos_arr.shape != (1,):
            raise ValueError(f"Expected gripper waypoint shape (1,), got {pos_arr.shape}.")
        self.command_queue.put(
            {
                "cmd": PikaCommand.SCHEDULE_WAYPOINT.value,
                "target_pos": float(pos_arr[0]),
                "target_time": float(target_time),
            }
        )

    def _import_pika_sdk(self):
        _append_sys_paths([self.pika_sdk_root])
        try:
            from pika.gripper import Gripper  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Missing Pika SDK. Set robot.pika_sdk_root or install the pika package in the active environment."
            ) from e
        return Gripper

    def _connect_gripper(self):
        Gripper = self._import_pika_sdk()
        gripper = Gripper(port=self.serial_port)
        if not gripper.connect():
            raise RuntimeError(f"Failed to connect to the Pika gripper at {self.serial_port}.")
        if not gripper.enable():
            raise RuntimeError("Failed to enable the Pika gripper.")
        if self.init_velocity is not None:
            gripper.set_velocity(self.init_velocity)
        return gripper

    def _put_state(self, position_raw: float, prev_pos_raw: float, prev_time: float, receive_time: float) -> None:
        dt = max(receive_time - prev_time, 1e-6)
        velocity_raw = (position_raw - prev_pos_raw) / dt
        state = {
            "gripper_state": 0,
            "gripper_position": position_raw / self.scale,
            "gripper_velocity": velocity_raw / self.scale,
            "gripper_force": 0.0,
            "gripper_measure_timestamp": receive_time,
            "gripper_receive_timestamp": receive_time,
            "gripper_timestamp": receive_time - self.receive_latency,
        }
        try:
            self.state_queue.put_nowait(state)
        except queue.Full:
            try:
                self.state_queue.get_nowait()
            except queue.Empty:
                pass
            self.state_queue.put_nowait(state)

    def run(self) -> None:
        gripper = None
        try:
            gripper = self._connect_gripper()
            curr_pos = float(gripper.get_gripper_distance())
            prev_pos = curr_pos
            prev_time = time.time()
            pending_pos = None
            pending_time = None
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            self.ready_event.set()

            while keep_running:
                t_now = time.monotonic()
                while True:
                    try:
                        command = self.command_queue.get_nowait()
                    except queue.Empty:
                        break
                    if command["cmd"] == PikaCommand.STOP.value:
                        keep_running = False
                        break
                    if command["cmd"] == PikaCommand.SCHEDULE_WAYPOINT.value:
                        pending_pos = float(np.clip(command["target_pos"] * self.scale, self.min_width, self.max_width))
                        pending_time = time.monotonic() - time.time() + float(command["target_time"])

                if pending_pos is not None and pending_time is not None and t_now >= pending_time:
                    gripper.set_gripper_distance(pending_pos)
                    pending_pos = None
                    pending_time = None

                receive_time = time.time()
                curr_pos = float(gripper.get_gripper_distance())
                self._put_state(curr_pos, prev_pos, prev_time, receive_time)
                prev_pos = curr_pos
                prev_time = receive_time

                iter_idx += 1
                precise_wait(t_start + iter_idx / self.frequency, time_func=time.monotonic)

        except BaseException:
            self.error_queue.put(traceback.format_exc())
            self.ready_event.set()
            raise
        finally:
            if self.verbose:
                logger.info("Pika controller disconnected from %s", self.serial_port)
