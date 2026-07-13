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

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig, Cv2Backends
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig

DEFAULT_REALMAN_IP = "192.168.1.18"
DEFAULT_REALMAN_PORT = 8080
DEFAULT_GRIPPER_SERIAL_PORT = "/dev/ttyUSB60"
DEFAULT_REALSENSE_SERIAL = "419122270755"
DEFAULT_FISHEYE_DEVICE = "/dev/video60"


def default_realman_pika_cameras() -> dict[str, CameraConfig]:
    return {
        "rgb": RealSenseCameraConfig(
            serial_number_or_name=DEFAULT_REALSENSE_SERIAL,
            width=640,
            height=480,
            fps=30,
        ),
        "fisheye": OpenCVCameraConfig(
            index_or_path=Path(DEFAULT_FISHEYE_DEVICE),
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
            backend=Cv2Backends.V4L2,
        ),
    }


@RobotConfig.register_subclass("realman_pika")
@dataclass
class RealmanPikaConfig(RobotConfig):
    """RealMan arm + Pika gripper robot configuration.

    Coordinates exposed to policies are the Pika gripper frame:
    ``[x, y, z, rx, ry, rz, gripper_width]`` where rotations are rotvecs and
    gripper width is meters.
    """

    robot_ip: str = DEFAULT_REALMAN_IP
    robot_port: int = DEFAULT_REALMAN_PORT
    robot_level: int = 3
    robot_mode: int = 2
    robot_joint_dim: int = 7

    robot_command_frequency: int = 125
    robot_command_mode: str = "movep_canfd"
    robot_interpolation_mode: str = "trajectory"
    robot_launch_timeout: float = 15.0
    robot_state_read_retries: int = 3
    robot_state_read_retry_delay_sec: float = 0.01
    robot_max_consecutive_state_read_failures: int = 10
    robot_use_udp_state: bool = True
    robot_udp_port: int = 8888
    robot_udp_cycle: int = 2
    robot_udp_target_ip: str | None = None
    robot_udp_state_timeout_sec: float = 0.2

    realman_sdk_root: Path | None = Path("/home/ubuntu/Documents/CodeField/zehao/Control_Tools")
    realman_rm_api_dir: Path | None = Path(
        "/home/ubuntu/Documents/CodeField/zehao/Control_Tools/RM_API2/Python"
    )

    gripper_serial_port: str = DEFAULT_GRIPPER_SERIAL_PORT
    pika_sdk_root: Path | None = Path("/home/ubuntu/Documents/CodeField/zehao/pika_sdk")
    gripper_frequency: int = 30
    gripper_move_max_speed_mm_s: float = 200.0
    gripper_min_width_m: float = 0.0
    gripper_max_width_m: float = 0.09
    gripper_init_velocity: float = 0.1
    gripper_launch_timeout: float = 3.0

    # More conservative than the legacy pika_env.py defaults.
    max_pos_speed: float = 0.15
    max_rot_speed: float = 0.4
    max_relative_pos: float = 0.03
    max_relative_rot: float = 0.15

    action_latency: float = 0.1
    robot_action_latency: float | None = None
    gripper_action_latency: float | None = None

    # Hardware-progress gating keeps sensing at rollout FPS but advances the
    # policy action queue only after the arm and gripper reach their targets.
    progress_gate_enabled: bool = False
    progress_position_tolerance_m: float = 0.0005
    progress_rotation_tolerance_rad: float = 0.02
    progress_gripper_tolerance_m: float = 0.001
    # Keep False for grasping: contact can prevent the gripper from reaching a
    # commanded closed width even though the grasp action has completed.
    progress_require_gripper_target: bool = False
    progress_settle_samples: int = 2
    progress_timeout_enabled: bool = True
    progress_timeout_margin_s: float = 0.3
    progress_min_timeout_s: float = 0.5

    cameras: dict[str, CameraConfig] = field(default_factory=default_realman_pika_cameras)
    disable_cameras_on_connect: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.robot_ip:
            raise ValueError("RealmanPikaConfig requires robot_ip.")
        if self.robot_port <= 0:
            raise ValueError(f"robot_port must be > 0, got {self.robot_port}.")
        if self.robot_joint_dim <= 0:
            raise ValueError(f"robot_joint_dim must be > 0, got {self.robot_joint_dim}.")
        if not self.gripper_serial_port:
            raise ValueError("RealmanPikaConfig requires gripper_serial_port.")
        if self.gripper_min_width_m >= self.gripper_max_width_m:
            raise ValueError(
                "gripper_min_width_m must be smaller than gripper_max_width_m, "
                f"got {self.gripper_min_width_m} >= {self.gripper_max_width_m}."
            )
        if self.max_relative_pos <= 0:
            raise ValueError(f"max_relative_pos must be > 0, got {self.max_relative_pos}.")
        if self.max_relative_rot <= 0:
            raise ValueError(f"max_relative_rot must be > 0, got {self.max_relative_rot}.")
        if self.gripper_move_max_speed_mm_s <= 0:
            raise ValueError(
                f"gripper_move_max_speed_mm_s must be > 0, got {self.gripper_move_max_speed_mm_s}."
            )
        if self.progress_position_tolerance_m <= 0:
            raise ValueError(
                f"progress_position_tolerance_m must be > 0, got {self.progress_position_tolerance_m}."
            )
        if self.progress_rotation_tolerance_rad <= 0:
            raise ValueError(
                f"progress_rotation_tolerance_rad must be > 0, got {self.progress_rotation_tolerance_rad}."
            )
        if self.progress_gripper_tolerance_m <= 0:
            raise ValueError(
                f"progress_gripper_tolerance_m must be > 0, got {self.progress_gripper_tolerance_m}."
            )
        if self.progress_settle_samples <= 0:
            raise ValueError(f"progress_settle_samples must be > 0, got {self.progress_settle_samples}.")
        if self.progress_timeout_margin_s < 0:
            raise ValueError(f"progress_timeout_margin_s must be >= 0, got {self.progress_timeout_margin_s}.")
        if self.progress_min_timeout_s <= 0:
            raise ValueError(f"progress_min_timeout_s must be > 0, got {self.progress_min_timeout_s}.")
        if self.robot_action_latency is None:
            self.robot_action_latency = self.action_latency
        if self.gripper_action_latency is None:
            self.gripper_action_latency = self.action_latency
