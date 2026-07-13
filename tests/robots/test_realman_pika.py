import time

import numpy as np
import pytest

from lerobot.robots.realman_pika.config_realman_pika import RealmanPikaConfig
import lerobot.robots.realman_pika.realman_pika as realman_pika_module
from lerobot.robots.realman_pika.realman_pika import STATE_ACTION_KEYS, RealmanPika
from lerobot.robots.realman_pika.transforms import (
    apply_realman_tcp_relative_pose,
    pika_gripper_pose_to_realman_tcp_pose,
    pika_relative_pose_to_realman_tcp_relative_pose,
    realman_tcp_relative_pose_to_pika_relative_pose,
    realman_tcp_pose_to_pika_gripper_pose,
)


class FakeCamera:
    width = 640
    height = 480
    is_connected = False

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def read_latest(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)


class FakeArm:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_ready = False
        self.scheduled = []
        self.pose = np.zeros((6,), dtype=np.float64)
        FakeArm.instances.append(self)

    def start(self, wait=True):
        self.is_ready = True

    def stop(self, wait=True):
        self.is_ready = False

    def get_state(self):
        return {"ActualTCPPose": self.pose}

    def schedule_waypoint(self, pose, target_time):
        self.scheduled.append((np.asarray(pose, dtype=np.float64), target_time))


class FakeGripper:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_ready = False
        self.scheduled = []
        self.position = 0.04
        FakeGripper.instances.append(self)

    def start(self, wait=True):
        self.is_ready = True

    def stop(self, wait=True):
        self.is_ready = False

    def get_state(self):
        return {"gripper_position": self.position}

    def schedule_waypoint(self, pos, target_time):
        self.scheduled.append((float(np.asarray(pos).reshape(-1)[0]), target_time))


def _patch_fakes(monkeypatch):
    FakeArm.instances.clear()
    FakeGripper.instances.clear()
    monkeypatch.setattr(
        realman_pika_module,
        "make_cameras_from_configs",
        lambda configs: {name: FakeCamera() for name in configs},
    )
    monkeypatch.setattr(realman_pika_module, "RealmanInterpolationController", FakeArm)
    monkeypatch.setattr(realman_pika_module, "PikaController", FakeGripper)


def test_feature_schema_and_order(monkeypatch, tmp_path):
    _patch_fakes(monkeypatch)
    robot = RealmanPika(RealmanPikaConfig(calibration_dir=tmp_path))

    assert list(robot.action_features) == list(STATE_ACTION_KEYS)
    assert list(robot.observation_features) == [*STATE_ACTION_KEYS, "rgb", "fisheye"]
    assert robot.observation_features["rgb"] == (480, 640, 3)
    assert robot.observation_features["fisheye"] == (480, 640, 3)


def test_realman_pika_pose_roundtrip():
    pose = np.array([0.32, -0.11, 0.25, 0.21, -0.17, 0.33], dtype=np.float64)
    pika_pose = realman_tcp_pose_to_pika_gripper_pose(pose)
    recovered = pika_gripper_pose_to_realman_tcp_pose(pika_pose)

    np.testing.assert_allclose(recovered, pose, atol=1e-8)


def test_realman_pika_relative_pose_roundtrip():
    relative = np.array([0.02, -0.01, 0.03, 0.04, -0.02, 0.01], dtype=np.float64)
    pika_relative = realman_tcp_relative_pose_to_pika_relative_pose(relative)
    recovered = pika_relative_pose_to_realman_tcp_relative_pose(pika_relative)

    np.testing.assert_allclose(recovered, relative, atol=1e-8)


def test_send_action_clips_and_schedules(monkeypatch, tmp_path):
    _patch_fakes(monkeypatch)
    cfg = RealmanPikaConfig(
        calibration_dir=tmp_path,
        max_relative_pos=0.02,
        max_relative_rot=0.05,
        action_latency=0.2,
    )
    robot = RealmanPika(cfg)
    robot.connect()

    current = robot._read_state_vector()
    np.testing.assert_allclose(current[:6], np.zeros(6), atol=1e-8)
    requested = current.copy()
    requested[:3] = np.array([0.5, -0.5, 0.5])
    requested[3:6] = np.array([1.0, -1.0, 1.0])
    requested[6] = 1.0
    action = {key: float(value) for key, value in zip(STATE_ACTION_KEYS, requested, strict=True)}

    before = time.time()
    sent = robot.send_action(action)
    after = time.time()
    sent_vec = np.array([sent[key] for key in STATE_ACTION_KEYS], dtype=np.float64)

    np.testing.assert_allclose(sent_vec[:3], np.array([0.02, -0.02, 0.02]))
    np.testing.assert_allclose(sent_vec[3:6], np.array([0.05, -0.05, 0.05]))
    assert sent_vec[6] == cfg.gripper_max_width_m

    arm = FakeArm.instances[-1]
    gripper = FakeGripper.instances[-1]
    assert len(arm.scheduled) == 1
    assert len(gripper.scheduled) == 1
    expected_realman_relative = pika_relative_pose_to_realman_tcp_relative_pose(sent_vec[:6])
    expected_realman_target = apply_realman_tcp_relative_pose(arm.pose, expected_realman_relative)
    np.testing.assert_allclose(arm.scheduled[0][0], expected_realman_target)
    assert gripper.scheduled[0][0] == cfg.gripper_max_width_m
    assert before + cfg.action_latency <= arm.scheduled[0][1] <= after + cfg.action_latency
    assert before + cfg.action_latency <= gripper.scheduled[0][1] <= after + cfg.action_latency

    robot.disconnect()


def test_realman_trajectory_interpolator_matches_umi_schedule_semantics():
    pytest.importorskip("scipy")
    from lerobot.robots.realman_pika.trajectory import (
        PoseTrajectoryInterpolator,
        realman_euler_pose_to_rotvec_pose,
        rotvec_pose_to_realman_euler_pose,
    )

    start_pose = np.zeros((6,), dtype=np.float64)
    target_pose = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float64)
    interp = PoseTrajectoryInterpolator(
        times=[0.0],
        poses=[realman_euler_pose_to_rotvec_pose(start_pose)],
    )

    scheduled = interp.schedule_waypoint(
        pose=realman_euler_pose_to_rotvec_pose(target_pose),
        time=1.0,
        max_pos_speed=0.1,
        max_rot_speed=1.0,
        curr_time=0.0,
        last_waypoint_time=0.0,
    )

    assert scheduled.times[-1] == pytest.approx(2.0)
    mid_pose = rotvec_pose_to_realman_euler_pose(scheduled(1.0))
    final_pose = rotvec_pose_to_realman_euler_pose(scheduled(2.0))
    np.testing.assert_allclose(mid_pose[:3], [0.1, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(final_pose, target_pose, atol=1e-8)
