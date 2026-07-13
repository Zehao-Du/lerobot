import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

import lerobot.robots.realman_pika.realman_pika as realman_pika_module
from lerobot.robots.realman_pika.config_realman_pika import RealmanPikaConfig
from lerobot.robots.realman_pika.controllers import _pop_latest_due_waypoint
from lerobot.robots.realman_pika.realman_pika import STATE_ACTION_KEYS, RealmanPika
from lerobot.robots.realman_pika.transforms import (
    apply_realman_tcp_relative_pose,
    pika_gripper_pose_to_realman_tcp_pose,
    pika_relative_pose_to_realman_tcp_relative_pose,
    realman_tcp_pose_to_pika_gripper_pose,
    realman_tcp_relative_pose_to_pika_relative_pose,
)
from lerobot.scripts import lerobot_realman_pika_test as hardware_test_module


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
        self.servoed = []
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

    def servol(self, pose, duration):
        self.servoed.append((np.asarray(pose, dtype=np.float64), duration))


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


def test_progress_gate_waits_for_settled_hardware_target(monkeypatch, tmp_path):
    _patch_fakes(monkeypatch)
    cfg = RealmanPikaConfig(
        calibration_dir=tmp_path,
        progress_gate_enabled=True,
        progress_settle_samples=2,
        max_relative_pos=0.005,
        max_pos_speed=0.01,
    )
    robot = RealmanPika(cfg)
    robot.connect()

    action = dict.fromkeys(STATE_ACTION_KEYS, 0.0)
    action["eef_x.pos"] = 0.005
    action["gripper.pos"] = 0.04
    robot.send_action(action)

    assert robot.requires_action_acknowledgement
    target = robot._active_action_target
    assert target is not None
    initial_obs = dict.fromkeys(STATE_ACTION_KEYS, 0.0)
    initial_obs["gripper.pos"] = 0.04
    status = robot.get_action_execution_status(initial_obs)
    assert status.active
    assert not status.reached
    assert status.position_error == pytest.approx(0.005)

    target_obs = {key: float(value) for key, value in zip(STATE_ACTION_KEYS, target, strict=True)}
    assert not robot.get_action_execution_status(target_obs).reached
    reached = robot.get_action_execution_status(target_obs)
    assert reached.reached
    assert not reached.timed_out

    robot.acknowledge_action_execution()
    assert not robot.get_action_execution_status(target_obs).active
    robot.disconnect()


def test_progress_gate_times_out_and_can_hold_current_pose(monkeypatch, tmp_path):
    _patch_fakes(monkeypatch)
    cfg = RealmanPikaConfig(
        calibration_dir=tmp_path,
        progress_gate_enabled=True,
        progress_min_timeout_s=0.01,
        progress_timeout_margin_s=0.0,
        max_relative_pos=0.005,
        max_pos_speed=0.01,
    )
    robot = RealmanPika(cfg)
    robot.connect()

    action = dict.fromkeys(STATE_ACTION_KEYS, 0.0)
    action["eef_x.pos"] = 0.005
    action["gripper.pos"] = 0.04
    robot.send_action(action)
    assert robot._active_action_timeout_s == pytest.approx(0.6)
    robot._active_action_started_at = time.monotonic() - 1.0

    initial_obs = dict.fromkeys(STATE_ACTION_KEYS, 0.0)
    initial_obs["gripper.pos"] = 0.04
    status = robot.get_action_execution_status(initial_obs)
    assert status.timed_out

    robot.hold_position()
    arm = FakeArm.instances[-1]
    assert len(arm.servoed) == 1
    np.testing.assert_allclose(arm.servoed[0][0], arm.pose)
    assert not robot.get_action_execution_status(initial_obs).active
    robot.disconnect()


def test_progress_gate_does_not_require_gripper_target_after_contact(monkeypatch, tmp_path):
    _patch_fakes(monkeypatch)
    cfg = RealmanPikaConfig(
        calibration_dir=tmp_path,
        progress_gate_enabled=True,
        progress_settle_samples=1,
        progress_require_gripper_target=False,
    )
    robot = RealmanPika(cfg)
    robot.connect()

    action = dict.fromkeys(STATE_ACTION_KEYS, 0.0)
    action["gripper.pos"] = 0.0
    robot.send_action(action)
    current_obs = dict.fromkeys(STATE_ACTION_KEYS, 0.0)
    current_obs["gripper.pos"] = 0.04

    status = robot.get_action_execution_status(current_obs)
    assert status.reached
    assert status.gripper_error == pytest.approx(0.04)
    robot.disconnect()


def test_progress_gate_can_disable_timeout(monkeypatch, tmp_path):
    _patch_fakes(monkeypatch)
    cfg = RealmanPikaConfig(
        calibration_dir=tmp_path,
        progress_gate_enabled=True,
        progress_timeout_enabled=False,
    )
    robot = RealmanPika(cfg)
    robot.connect()

    action = dict.fromkeys(STATE_ACTION_KEYS, 0.0)
    action["eef_x.pos"] = 0.005
    action["gripper.pos"] = 0.04
    robot.send_action(action)
    robot._active_action_started_at = time.monotonic() - 60.0

    current_obs = dict.fromkeys(STATE_ACTION_KEYS, 0.0)
    current_obs["gripper.pos"] = 0.04
    status = robot.get_action_execution_status(current_obs)
    assert status.active
    assert not status.reached
    assert not status.timed_out
    assert status.timeout_s is None
    robot.disconnect()


def test_pika_waypoints_remain_scheduled_when_commands_arrive_faster_than_latency():
    waypoints = deque(
        [
            (0.100, 50.0),
            (0.133, 40.0),
            (0.166, 30.0),
            (0.200, 20.0),
        ]
    )

    assert _pop_latest_due_waypoint(waypoints, 0.099) is None
    assert _pop_latest_due_waypoint(waypoints, 0.100) == 50.0
    assert _pop_latest_due_waypoint(waypoints, 0.150) == 40.0
    assert _pop_latest_due_waypoint(waypoints, 0.199) == 30.0
    assert _pop_latest_due_waypoint(waypoints, 0.200) == 20.0
    assert not waypoints


def test_pika_waypoints_skip_stale_targets_but_preserve_future_target():
    waypoints = deque([(0.100, 50.0), (0.133, 40.0), (0.166, 30.0)])

    assert _pop_latest_due_waypoint(waypoints, 0.150) == 40.0
    assert list(waypoints) == [(0.166, 30.0)]


def test_latency_measurement_uses_feedback_timestamps(monkeypatch):
    class FakeClock:
        wall_time = 100.0
        monotonic_time = 0.0

        def time(self):
            return self.wall_time

        def monotonic(self):
            return self.monotonic_time

        def sleep(self, duration):
            self.wall_time += duration
            self.monotonic_time += duration

    clock = FakeClock()
    monkeypatch.setattr(hardware_test_module.time, "time", clock.time)
    monkeypatch.setattr(hardware_test_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(hardware_test_module.time, "sleep", clock.sleep)

    samples = iter(
        [
            (np.array([0.0]), 99.99),
            (np.array([0.0]), 100.05),
            (np.array([0.3]), 100.12),
            (np.array([0.95]), 100.15),
            (np.array([0.98]), 100.16),
        ]
    )
    scheduled_times = []
    result = hardware_test_module._measure_step_latency(
        read_sample=lambda: next(samples),
        schedule=scheduled_times.append,
        target=np.array([1.0]),
        schedule_delay_s=0.1,
        onset_threshold=0.2,
        target_tolerance=0.1,
        timeout_s=1.0,
        poll_interval_s=0.01,
    )

    assert scheduled_times == [100.1]
    assert result.onset_from_submit_s == pytest.approx(0.12)
    assert result.onset_from_target_time_s == pytest.approx(0.02)
    assert result.reached_from_submit_s == pytest.approx(0.16)
    assert result.reached_from_target_time_s == pytest.approx(0.06)


def test_arm_latency_probe_uses_immediate_servol(monkeypatch):
    class FakeArm:
        def __init__(self):
            self.calls = []

        def get_state(self):
            return {
                "ActualTCPPose": np.zeros(6),
                "robot_receive_timestamp": 1.0,
            }

        def servol(self, pose, duration):
            self.calls.append((np.asarray(pose), duration))

    delays = []

    def fake_measure_step_latency(**kwargs):
        delays.append(kwargs["schedule_delay_s"])
        kwargs["schedule"](123.0)
        return hardware_test_module.StepLatencyResult(0.01, 0.01, 0.05, 0.05)

    monkeypatch.setattr(hardware_test_module, "_measure_step_latency", fake_measure_step_latency)
    arm = FakeArm()
    args = SimpleNamespace(
        latency_trials=1,
        latency_arm_step_mm=5.0,
        latency_arm_onset_mm=0.2,
        latency_arm_tolerance_mm=0.5,
        latency_timeout=1.0,
        robot_command_frequency=125,
        latency_rest=0.0,
    )

    hardware_test_module._run_arm_latency_test(arm, args)

    assert delays == [0.0, 0.0]
    assert len(arm.calls) == 2
    assert all(duration == 0.0 for _, duration in arm.calls)
    assert arm.calls[0][0][0] == pytest.approx(0.005)
    assert arm.calls[1][0][0] == pytest.approx(0.0)


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
