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

"""Interactive RealMan arm + Pika gripper controller smoke test.

This is the LeRobot-side equivalent of pika-dp's ``pika_env.py`` main menu,
but it only starts the arm and gripper controllers. It does not start cameras,
policy inference, replay buffers, or recording.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.robots.realman_pika.config_realman_pika import (
    DEFAULT_GRIPPER_SERIAL_PORT,
    DEFAULT_REALMAN_IP,
    DEFAULT_REALMAN_PORT,
    RealmanPikaConfig,
)
from lerobot.robots.realman_pika.controllers import PikaController, RealmanInterpolationController
from lerobot.robots.realman_pika.transforms import (
    apply_realman_tcp_relative_pose,
    pika_relative_pose_to_realman_tcp_relative_pose,
    realman_tcp_relative_pose_between,
    realman_tcp_relative_pose_to_pika_relative_pose,
)
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepLatencyResult:
    onset_from_submit_s: float
    onset_from_target_time_s: float
    reached_from_submit_s: float
    reached_from_target_time_s: float


def _parse_six_floats(raw: str) -> np.ndarray:
    values = raw.replace(",", " ").split()
    if len(values) != 6:
        raise ValueError("Expected 6 numbers: x y z rx ry rz")
    return np.array([float(value) for value in values], dtype=np.float64)


def _parse_one_float(raw: str) -> float:
    values = raw.replace(",", " ").split()
    if len(values) != 1:
        raise ValueError("Expected 1 number.")
    return float(values[0])


def _as_latest_float(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == ():
        return float(arr)
    return float(arr.reshape(-1)[-1])


def _as_latest_pose(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (6,):
        return arr
    if arr.ndim >= 2 and arr.shape[-1] == 6:
        return arr.reshape(-1, 6)[-1]
    raise ValueError(f"Expected pose with trailing dimension 6, got {arr.shape}.")


def _as_latest_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim >= 2:
        return arr.reshape(-1, arr.shape[-1])[-1]
    raise ValueError(f"Expected vector, got {arr.shape}.")


def _retry_get_state(controller, label: str, timeout_s: float = 3.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return controller.get_state()
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for {label} state.") from last_error


def _latest_realman_tcp_pose(arm: RealmanInterpolationController) -> np.ndarray:
    state = _retry_get_state(arm, "RealMan")
    return _as_latest_pose(state["ActualTCPPose"])


def _latest_pika_relative_state(
    arm: RealmanInterpolationController, reference_realman_tcp_pose: np.ndarray
) -> np.ndarray:
    realman_relative = realman_tcp_relative_pose_between(
        reference_realman_tcp_pose, _latest_realman_tcp_pose(arm)
    )
    return realman_tcp_relative_pose_to_pika_relative_pose(realman_relative)


def _latest_joint_pos(arm: RealmanInterpolationController) -> np.ndarray:
    state = _retry_get_state(arm, "RealMan")
    return _as_latest_vector(state["ActualQ"])


def _latest_gripper_width(gripper: PikaController) -> float:
    state = _retry_get_state(gripper, "Pika")
    return _as_latest_float(state["gripper_position"])


def _format_pose(pose: np.ndarray) -> str:
    return np.array2string(np.asarray(pose, dtype=np.float64), precision=6, suppress_small=True)


def _print_state(
    arm: RealmanInterpolationController,
    gripper: PikaController,
    reference_realman_tcp_pose: np.ndarray,
) -> None:
    pika_relative = _latest_pika_relative_state(arm, reference_realman_tcp_pose)
    realman_tcp = _latest_realman_tcp_pose(arm)
    joint = _latest_joint_pos(arm)
    width_m = _latest_gripper_width(gripper)

    print("pika_tcp_relative_state [x y z rx ry rz]:", _format_pose(pika_relative))
    print("realman_tcp_pose        [x y z rx ry rz]:", _format_pose(realman_tcp))
    print("joint_rad:", _format_pose(joint))
    print(f"gripper_width: {width_m:.6f} m ({width_m * 1000:.2f} mm)")


def _measure_step_latency(
    *,
    read_sample: Callable[[], tuple[np.ndarray, float]],
    schedule: Callable[[float], None],
    target: np.ndarray,
    schedule_delay_s: float,
    onset_threshold: float,
    target_tolerance: float,
    timeout_s: float,
    poll_interval_s: float,
    required_reached_samples: int = 2,
) -> StepLatencyResult:
    initial, initial_timestamp = read_sample()
    initial = np.asarray(initial, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if initial.shape != target.shape:
        raise ValueError(f"Initial and target shapes differ: {initial.shape} != {target.shape}.")
    if np.linalg.norm(target - initial) <= onset_threshold:
        raise ValueError("Target displacement must be larger than the onset threshold.")

    submit_time = time.time()
    target_time = submit_time + schedule_delay_s
    schedule(target_time)

    onset_timestamp = None
    reached_timestamp = None
    reached_samples = 0
    deadline = time.monotonic() + timeout_s
    last_timestamp = initial_timestamp
    while time.monotonic() < deadline:
        value, timestamp = read_sample()
        value = np.asarray(value, dtype=np.float64)
        if timestamp <= last_timestamp:
            time.sleep(poll_interval_s)
            continue
        last_timestamp = timestamp

        if onset_timestamp is None and np.linalg.norm(value - initial) >= onset_threshold:
            onset_timestamp = timestamp

        if np.linalg.norm(value - target) <= target_tolerance:
            reached_samples += 1
            if reached_samples >= required_reached_samples:
                reached_timestamp = timestamp
                break
        else:
            reached_samples = 0
        time.sleep(poll_interval_s)

    if onset_timestamp is None:
        raise TimeoutError(f"No motion detected within {timeout_s:.2f}s.")
    if reached_timestamp is None:
        raise TimeoutError(f"Motion started but target was not reached within {timeout_s:.2f}s.")

    return StepLatencyResult(
        onset_from_submit_s=onset_timestamp - submit_time,
        onset_from_target_time_s=onset_timestamp - target_time,
        reached_from_submit_s=reached_timestamp - submit_time,
        reached_from_target_time_s=reached_timestamp - target_time,
    )


def _print_latency_result(label: str, trial: int, result: StepLatencyResult) -> None:
    print(
        f"{label} trial {trial}: "
        f"onset_from_submit={result.onset_from_submit_s * 1000:.1f} ms, "
        f"onset_vs_target_time={result.onset_from_target_time_s * 1000:+.1f} ms, "
        f"reached_from_submit={result.reached_from_submit_s * 1000:.1f} ms, "
        f"reached_vs_target_time={result.reached_from_target_time_s * 1000:+.1f} ms"
    )


def _print_latency_summary(
    label: str, results: list[StepLatencyResult], expected_motion_s: float | None = None
) -> None:
    onset_ms = np.array([result.onset_from_submit_s for result in results]) * 1000
    reached_ms = np.array([result.reached_from_submit_s for result in results]) * 1000
    print(f"\n{label} latency summary ({len(results)} trials)")
    print(f"  onset  : median={np.median(onset_ms):.1f} ms, P95={np.percentile(onset_ms, 95):.1f} ms")
    print(f"  reached: median={np.median(reached_ms):.1f} ms, P95={np.percentile(reached_ms, 95):.1f} ms")
    if expected_motion_s is not None:
        overhead_ms = reached_ms - expected_motion_s * 1000
        print(f"  kinematic minimum: {expected_motion_s * 1000:.1f} ms")
        print(
            f"  reached minus kinematic minimum: "
            f"median={np.median(overhead_ms):.1f} ms, P95={np.percentile(overhead_ms, 95):.1f} ms"
        )


def _confirm_latency_test(args: argparse.Namespace) -> None:
    print("\nAutomatic hardware latency test")
    print(f"  devices: {args.latency_test}")
    print(f"  trials: {args.latency_trials}")
    if args.latency_test in ("gripper", "both"):
        print(f"  gripper step: {args.latency_gripper_step_mm:.1f} mm")
        print(f"  gripper scheduled delay: {args.action_latency * 1000:.1f} ms")
        print(f"  gripper feedback resolution: about {1000 / args.gripper_frequency:.1f} ms")
    if args.latency_test in ("arm", "both"):
        print(f"  arm step: {args.latency_arm_step_mm:.1f} mm along RealMan base X")
        print("  arm command mode: immediate servol (speed-limited trajectory)")
        print(f"  arm max position speed: {args.max_pos_speed:.4f} m/s")
        print(f"  arm max rotation speed: {args.max_rot_speed:.4f} rad/s (not exercised by X-only test)")
        expected_detection_s = (
            max(args.latency_arm_step_mm - args.latency_arm_tolerance_mm, 0.0)
            / 1000
            / args.max_pos_speed
        )
        print(f"  arm kinematic minimum to tolerance: {expected_detection_s * 1000:.1f} ms")
        print(f"  arm feedback resolution: about {1000 / args.robot_command_frequency:.1f} ms")
    response = input("Clear the workspace and type 'yes' to start motion: ").strip().lower()
    if response != "yes":
        raise RuntimeError("Latency test cancelled.")


def _run_gripper_latency_test(
    gripper: PikaController, args: argparse.Namespace
) -> list[StepLatencyResult]:
    def read_sample() -> tuple[np.ndarray, float]:
        state = _retry_get_state(gripper, "Pika")
        value = np.array([_as_latest_float(state["gripper_position"])], dtype=np.float64)
        timestamp = _as_latest_float(state["gripper_receive_timestamp"])
        return value, timestamp

    results = []
    step_m = args.latency_gripper_step_mm / 1000.0
    for trial in range(1, args.latency_trials + 1):
        initial, _ = read_sample()
        direction = -1.0 if initial[0] - step_m >= args.gripper_min_width_m else 1.0
        target = np.array([initial[0] + direction * step_m], dtype=np.float64)
        target[0] = np.clip(target[0], args.gripper_min_width_m, args.gripper_max_width_m)
        print(f"gripper trial {trial}: {initial[0] * 1000:.2f} -> {target[0] * 1000:.2f} mm")

        result = _measure_step_latency(
            read_sample=read_sample,
            schedule=lambda target_time: gripper.schedule_waypoint(target[0], target_time),
            target=target,
            schedule_delay_s=args.action_latency,
            onset_threshold=args.latency_gripper_onset_mm / 1000.0,
            target_tolerance=args.latency_gripper_tolerance_mm / 1000.0,
            timeout_s=args.latency_timeout,
            poll_interval_s=1.0 / (args.gripper_frequency * 2),
        )
        results.append(result)
        _print_latency_result("gripper", trial, result)

        _measure_step_latency(
            read_sample=read_sample,
            schedule=lambda target_time: gripper.schedule_waypoint(initial[0], target_time),
            target=initial,
            schedule_delay_s=args.action_latency,
            onset_threshold=args.latency_gripper_onset_mm / 1000.0,
            target_tolerance=args.latency_gripper_tolerance_mm / 1000.0,
            timeout_s=args.latency_timeout,
            poll_interval_s=1.0 / (args.gripper_frequency * 2),
        )
        time.sleep(args.latency_rest)
    return results


def _run_arm_latency_test(
    arm: RealmanInterpolationController, args: argparse.Namespace
) -> list[StepLatencyResult]:
    def read_sample() -> tuple[np.ndarray, float]:
        state = _retry_get_state(arm, "RealMan")
        pose = _as_latest_pose(state["ActualTCPPose"])
        timestamp = _as_latest_float(state["robot_receive_timestamp"])
        return pose[:3], timestamp

    results = []
    step_m = args.latency_arm_step_mm / 1000.0
    for trial in range(1, args.latency_trials + 1):
        initial_pose = _latest_realman_tcp_pose(arm)
        target_pose = initial_pose.copy()
        target_pose[0] += step_m
        print(f"arm trial {trial}: base X {initial_pose[0]:.6f} -> {target_pose[0]:.6f} m")

        result = _measure_step_latency(
            read_sample=read_sample,
            schedule=lambda _target_time: arm.servol(target_pose, duration=0.0),
            target=target_pose[:3],
            schedule_delay_s=0.0,
            onset_threshold=args.latency_arm_onset_mm / 1000.0,
            target_tolerance=args.latency_arm_tolerance_mm / 1000.0,
            timeout_s=args.latency_timeout,
            poll_interval_s=1.0 / (args.robot_command_frequency * 2),
        )
        results.append(result)
        _print_latency_result("arm", trial, result)

        _measure_step_latency(
            read_sample=read_sample,
            schedule=lambda _target_time: arm.servol(initial_pose, duration=0.0),
            target=initial_pose[:3],
            schedule_delay_s=0.0,
            onset_threshold=args.latency_arm_onset_mm / 1000.0,
            target_tolerance=args.latency_arm_tolerance_mm / 1000.0,
            timeout_s=args.latency_timeout,
            poll_interval_s=1.0 / (args.robot_command_frequency * 2),
        )
        time.sleep(args.latency_rest)
    return results


def _submit_action(
    arm: RealmanInterpolationController,
    gripper: PikaController,
    pika_relative: np.ndarray,
    gripper_width_m: float,
    action_latency: float,
) -> None:
    if action_latency < 0:
        raise ValueError(f"action_latency must be >= 0, got {action_latency}.")
    current_realman_tcp_pose = _latest_realman_tcp_pose(arm)
    realman_relative = pika_relative_pose_to_realman_tcp_relative_pose(pika_relative)
    realman_tcp_target = apply_realman_tcp_relative_pose(current_realman_tcp_pose, realman_relative)
    target_time = time.time() + action_latency
    arm.schedule_waypoint(realman_tcp_target, target_time=target_time)
    gripper.schedule_waypoint(gripper_width_m, target_time=target_time)

    print(f"Scheduled target at t+{action_latency:.3f}s")
    print("current_realman_tcp_absolute    :", _format_pose(current_realman_tcp_pose))
    print("model_pika_tcp_local_relative   :", _format_pose(pika_relative))
    print("robot_realman_tcp_local_relative:", _format_pose(realman_relative))
    print("robot_realman_base_delta        :", _format_pose(realman_tcp_target - current_realman_tcp_pose))
    print("robot_realman_tcp_absolute      :", _format_pose(realman_tcp_target))
    print("gripper_width                  :", f"{gripper_width_m:.6f} m ({gripper_width_m * 1000:.2f} mm)")


def _make_arg_parser() -> argparse.ArgumentParser:
    cfg_defaults = RealmanPikaConfig()
    parser = argparse.ArgumentParser(
        description="Interactive RealMan arm + Pika gripper test using LeRobot controllers."
    )
    parser.add_argument("--robot_ip", default=DEFAULT_REALMAN_IP)
    parser.add_argument("--robot_port", type=int, default=DEFAULT_REALMAN_PORT)
    parser.add_argument("--robot_level", type=int, default=cfg_defaults.robot_level)
    parser.add_argument("--robot_mode", type=int, default=cfg_defaults.robot_mode)
    parser.add_argument("--robot_joint_dim", type=int, default=cfg_defaults.robot_joint_dim)
    parser.add_argument("--robot_state_read_retries", type=int, default=cfg_defaults.robot_state_read_retries)
    parser.add_argument(
        "--robot_state_read_retry_delay",
        type=float,
        default=cfg_defaults.robot_state_read_retry_delay_sec,
    )
    parser.add_argument(
        "--robot_max_consecutive_state_read_failures",
        type=int,
        default=cfg_defaults.robot_max_consecutive_state_read_failures,
    )
    parser.add_argument("--no_robot_udp_state", action="store_true", help="Disable RealMan UDP state.")
    parser.add_argument("--robot_udp_port", type=int, default=cfg_defaults.robot_udp_port)
    parser.add_argument("--robot_udp_cycle", type=int, default=cfg_defaults.robot_udp_cycle)
    parser.add_argument("--robot_udp_target_ip", default=cfg_defaults.robot_udp_target_ip)
    parser.add_argument("--robot_udp_state_timeout", type=float, default=cfg_defaults.robot_udp_state_timeout_sec)
    parser.add_argument("--robot_launch_timeout", type=float, default=cfg_defaults.robot_launch_timeout)
    parser.add_argument("--robot_command_frequency", type=int, default=cfg_defaults.robot_command_frequency)
    parser.add_argument(
        "--robot_command_mode",
        default=cfg_defaults.robot_command_mode,
        choices=("movep_canfd", "movep_follow", "movel", "movej_p"),
    )
    parser.add_argument(
        "--robot_interpolation_mode",
        default=cfg_defaults.robot_interpolation_mode,
        choices=("trajectory", "none"),
    )
    parser.add_argument(
        "--max_pos_speed", "--robot.max_pos_speed", dest="max_pos_speed", type=float, default=cfg_defaults.max_pos_speed
    )
    parser.add_argument(
        "--max_rot_speed", "--robot.max_rot_speed", dest="max_rot_speed", type=float, default=cfg_defaults.max_rot_speed
    )
    parser.add_argument("--realman_sdk_root", type=Path, default=cfg_defaults.realman_sdk_root)
    parser.add_argument("--realman_rm_api_dir", type=Path, default=cfg_defaults.realman_rm_api_dir)

    parser.add_argument("--gripper_serial_port", default=DEFAULT_GRIPPER_SERIAL_PORT)
    parser.add_argument("--pika_sdk_root", type=Path, default=cfg_defaults.pika_sdk_root)
    parser.add_argument("--gripper_frequency", type=int, default=cfg_defaults.gripper_frequency)
    parser.add_argument("--gripper_move_max_speed_mm_s", type=float, default=cfg_defaults.gripper_move_max_speed_mm_s)
    parser.add_argument("--gripper_min_width_m", type=float, default=cfg_defaults.gripper_min_width_m)
    parser.add_argument("--gripper_max_width_m", type=float, default=cfg_defaults.gripper_max_width_m)
    parser.add_argument("--gripper_init_velocity", type=float, default=cfg_defaults.gripper_init_velocity)
    parser.add_argument("--gripper_launch_timeout", type=float, default=cfg_defaults.gripper_launch_timeout)

    parser.add_argument("--action_latency", type=float, default=cfg_defaults.action_latency)
    parser.add_argument("--latency_test", choices=("arm", "gripper", "both"))
    parser.add_argument("--latency_trials", type=int, default=5)
    parser.add_argument("--latency_timeout", type=float, default=5.0)
    parser.add_argument("--latency_rest", type=float, default=0.5)
    parser.add_argument("--latency_arm_step_mm", type=float, default=5.0)
    parser.add_argument("--latency_arm_onset_mm", type=float, default=0.2)
    parser.add_argument("--latency_arm_tolerance_mm", type=float, default=0.5)
    parser.add_argument("--latency_gripper_step_mm", type=float, default=5.0)
    parser.add_argument("--latency_gripper_onset_mm", type=float, default=0.5)
    parser.add_argument("--latency_gripper_tolerance_mm", type=float, default=1.0)
    return parser


def _validate_latency_args(args: argparse.Namespace) -> None:
    positive_names = (
        "latency_trials",
        "latency_timeout",
        "latency_arm_step_mm",
        "latency_arm_onset_mm",
        "latency_arm_tolerance_mm",
        "latency_gripper_step_mm",
        "latency_gripper_onset_mm",
        "latency_gripper_tolerance_mm",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be > 0.")
    if args.action_latency < 0:
        raise ValueError("--action_latency must be >= 0.")
    if args.latency_rest < 0:
        raise ValueError("--latency_rest must be >= 0.")
    if args.latency_arm_step_mm <= args.latency_arm_onset_mm:
        raise ValueError("--latency_arm_step_mm must exceed --latency_arm_onset_mm.")
    if args.latency_gripper_step_mm <= args.latency_gripper_onset_mm:
        raise ValueError("--latency_gripper_step_mm must exceed --latency_gripper_onset_mm.")


def _make_controllers(args: argparse.Namespace) -> tuple[RealmanInterpolationController, PikaController]:
    arm = RealmanInterpolationController(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        level=args.robot_level,
        mode=args.robot_mode,
        frequency=args.robot_command_frequency,
        max_pos_speed=args.max_pos_speed,
        max_rot_speed=args.max_rot_speed,
        launch_timeout=args.robot_launch_timeout,
        joint_dim=args.robot_joint_dim,
        command_mode=args.robot_command_mode,
        interpolation_mode=args.robot_interpolation_mode,
        state_read_retries=args.robot_state_read_retries,
        state_read_retry_delay=args.robot_state_read_retry_delay,
        max_consecutive_state_read_failures=args.robot_max_consecutive_state_read_failures,
        use_udp_state=not args.no_robot_udp_state,
        udp_port=args.robot_udp_port,
        udp_cycle=args.robot_udp_cycle,
        udp_target_ip=args.robot_udp_target_ip,
        udp_state_timeout=args.robot_udp_state_timeout,
        realman_sdk_root=args.realman_sdk_root,
        realman_rm_api_dir=args.realman_rm_api_dir,
    )
    gripper = PikaController(
        serial_port=args.gripper_serial_port,
        frequency=args.gripper_frequency,
        move_max_speed=args.gripper_move_max_speed_mm_s,
        launch_timeout=args.gripper_launch_timeout,
        use_meters=True,
        min_width=args.gripper_min_width_m,
        max_width=args.gripper_max_width_m,
        init_velocity=args.gripper_init_velocity,
        pika_sdk_root=args.pika_sdk_root,
    )
    return arm, gripper


def _print_menu() -> None:
    print("\nSelect next action:")
    print("1. get state")
    print("2. move Pika TCP relative offset from current TCP")
    print("3. move RealMan TCP relative offset from current TCP")
    print("4. record init pose")
    print("5. get gripper")
    print("6. set gripper width")
    print("7. exit")


def main() -> None:
    init_logging()
    args = _make_arg_parser().parse_args()
    _validate_latency_args(args)
    arm, gripper = _make_controllers(args)

    try:
        logger.info("Starting RealMan controller...")
        arm.start(wait=True)
        logger.info("Starting Pika gripper controller...")
        gripper.start(wait=True)

        reference_realman_tcp_pose = _latest_realman_tcp_pose(arm)
        print("Ready.")
        print("Reference RealMan TCP pose:", _format_pose(reference_realman_tcp_pose))

        if args.latency_test is not None:
            _confirm_latency_test(args)
            if args.latency_test in ("gripper", "both"):
                _print_latency_summary("gripper", _run_gripper_latency_test(gripper, args))
            if args.latency_test in ("arm", "both"):
                expected_motion_s = (
                    max(args.latency_arm_step_mm - args.latency_arm_tolerance_mm, 0.0)
                    / 1000
                    / args.max_pos_speed
                )
                _print_latency_summary(
                    "arm", _run_arm_latency_test(arm, args), expected_motion_s=expected_motion_s
                )
            return

        while True:
            _print_menu()
            command = input("Enter command: ").strip().lower()

            try:
                if command in ("1", "get state", "get_state"):
                    _print_state(arm, gripper, reference_realman_tcp_pose)

                elif command in ("2", "move pika relative", "pika", "offset"):
                    raw_pose = input("Enter Pika TCP relative offset [dx dy dz drx dry drz]: ").strip()
                    pika_relative = _parse_six_floats(raw_pose)
                    width_m = _latest_gripper_width(gripper)
                    _submit_action(arm, gripper, pika_relative, width_m, args.action_latency)

                elif command in ("3", "move realman relative", "realman"):
                    raw_pose = input("Enter RealMan TCP relative offset [dx dy dz drx dry drz]: ").strip()
                    realman_relative = _parse_six_floats(raw_pose)
                    pika_relative = realman_tcp_relative_pose_to_pika_relative_pose(realman_relative)
                    width_m = _latest_gripper_width(gripper)
                    _submit_action(arm, gripper, pika_relative, width_m, args.action_latency)

                elif command in ("4", "record init pose", "record_init_pose", "record init"):
                    reference_realman_tcp_pose = _latest_realman_tcp_pose(arm)
                    print("Recorded reference RealMan TCP pose:", _format_pose(reference_realman_tcp_pose))

                elif command in ("5", "get gripper", "get_gripper"):
                    width_m = _latest_gripper_width(gripper)
                    print(f"gripper_width: {width_m:.6f} m ({width_m * 1000:.2f} mm)")

                elif command in ("6", "set gripper width", "set_gripper_width", "set gripper"):
                    raw_width = input("Enter gripper width in mm: ").strip()
                    width_m = _parse_one_float(raw_width) / 1000.0
                    width_m = float(np.clip(width_m, args.gripper_min_width_m, args.gripper_max_width_m))
                    pika_relative = np.zeros((6,), dtype=np.float64)
                    _submit_action(arm, gripper, pika_relative, width_m, args.action_latency)

                elif command in ("7", "exit", "quit", "q"):
                    break

                else:
                    print("Unknown command. Use 1, 2, 3, 4, 5, 6, or 7.")
            except Exception as e:  # noqa: BLE001
                logger.error("Command failed: %s", e)

    finally:
        logger.info("Stopping controllers...")
        for controller in (gripper, arm):
            try:
                if controller.is_alive():
                    controller.stop(wait=True)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to stop %s cleanly: %s", controller.name, e)


if __name__ == "__main__":
    main()
