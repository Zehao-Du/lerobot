# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Minimal tests for the rollout module's public API."""

from __future__ import annotations

import dataclasses
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------


def test_rollout_top_level_imports():
    import lerobot.rollout

    for name in lerobot.rollout.__all__:
        assert hasattr(lerobot.rollout, name), f"Missing export: {name}"


def test_inference_submodule_imports():
    import lerobot.rollout.inference

    for name in lerobot.rollout.inference.__all__:
        assert hasattr(lerobot.rollout.inference, name), f"Missing export: {name}"


def test_strategies_submodule_imports():
    import lerobot.rollout.strategies

    for name in lerobot.rollout.strategies.__all__:
        assert hasattr(lerobot.rollout.strategies, name), f"Missing export: {name}"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_strategy_config_types():
    from lerobot.rollout import (
        ActionDebugStrategyConfig,
        BaseStrategyConfig,
        DAggerStrategyConfig,
        EpisodicStrategyConfig,
        HighlightStrategyConfig,
        SentryStrategyConfig,
    )

    assert BaseStrategyConfig().type == "base"
    assert ActionDebugStrategyConfig().type == "action_debug"
    assert SentryStrategyConfig().type == "sentry"
    assert HighlightStrategyConfig().type == "highlight"
    assert DAggerStrategyConfig().type == "dagger"
    assert EpisodicStrategyConfig().type == "episodic"


def test_dagger_config_invalid_input_device():
    from lerobot.rollout import DAggerStrategyConfig

    with pytest.raises(ValueError, match="input_device must be 'keyboard' or 'pedal'"):
        DAggerStrategyConfig(input_device="joystick")


def test_dagger_config_defaults():
    from lerobot.rollout import DAggerStrategyConfig

    cfg = DAggerStrategyConfig()
    assert cfg.num_episodes is None
    assert cfg.record_autonomous is False
    assert cfg.input_device == "keyboard"


def test_inference_config_types():
    from lerobot.rollout import HumanInLoopInferenceConfig, RTCInferenceConfig, SyncInferenceConfig

    assert SyncInferenceConfig().type == "sync"
    human_in_loop = HumanInLoopInferenceConfig()
    assert human_in_loop.type == "human_in_loop"
    assert human_in_loop.gripper_width_offset == 0.0

    rtc = RTCInferenceConfig()
    assert rtc.type == "rtc"
    assert rtc.queue_threshold == 30
    assert rtc.rtc is not None


def test_sentry_config_defaults():
    from lerobot.rollout import SentryStrategyConfig

    cfg = SentryStrategyConfig()
    assert cfg.upload_every_n_episodes == 5
    assert cfg.target_video_file_size_mb is None


# ---------------------------------------------------------------------------
# RolloutRingBuffer
# ---------------------------------------------------------------------------


def test_ring_buffer_append_and_eviction():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=0.5, max_memory_mb=100.0, fps=10.0)
    # max_frames = 5
    for i in range(8):
        buf.append({"val": i})
    assert len(buf) == 5


def test_ring_buffer_drain():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    for i in range(3):
        buf.append({"val": i})
    frames = buf.drain()
    assert len(frames) == 3
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_clear():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    buf.append({"val": 1})
    buf.clear()
    assert len(buf) == 0
    assert buf.estimated_bytes == 0


def test_ring_buffer_tensor_bytes():
    from lerobot.rollout.ring_buffer import RolloutRingBuffer

    buf = RolloutRingBuffer(max_seconds=1.0, max_memory_mb=100.0, fps=10.0)
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes
    buf.append({"tensor": t})
    assert buf.estimated_bytes >= 400


# ---------------------------------------------------------------------------
# ThreadSafeRobot
# ---------------------------------------------------------------------------


def test_thread_safe_robot_delegates(tmp_path):
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3, calibration_dir=tmp_path))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    obs = wrapper.get_observation()
    assert "motor_1.pos" in obs
    assert "motor_2.pos" in obs
    assert "motor_3.pos" in obs

    action = {"motor_1.pos": 0.0, "motor_2.pos": 1.0, "motor_3.pos": 2.0}
    result = wrapper.send_action(action)
    assert result == action

    robot.disconnect()


def test_thread_safe_robot_properties(tmp_path):
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from tests.mocks.mock_robot import MockRobot, MockRobotConfig

    robot = MockRobot(MockRobotConfig(n_motors=3, calibration_dir=tmp_path))
    robot.connect()
    wrapper = ThreadSafeRobot(robot)

    assert wrapper.name == "mock_robot"
    assert "motor_1.pos" in wrapper.observation_features
    assert "motor_1.pos" in wrapper.action_features
    assert wrapper.is_connected is True
    assert wrapper.inner is robot

    robot.disconnect()


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def test_create_strategy_dispatches():
    from lerobot.rollout import (
        ActionDebugStrategy,
        ActionDebugStrategyConfig,
        BaseStrategy,
        BaseStrategyConfig,
        DAggerStrategy,
        DAggerStrategyConfig,
        EpisodicStrategy,
        EpisodicStrategyConfig,
        SentryStrategy,
        SentryStrategyConfig,
        create_strategy,
    )

    assert isinstance(create_strategy(BaseStrategyConfig()), BaseStrategy)
    assert isinstance(create_strategy(ActionDebugStrategyConfig()), ActionDebugStrategy)
    assert isinstance(create_strategy(SentryStrategyConfig()), SentryStrategy)
    assert isinstance(create_strategy(DAggerStrategyConfig()), DAggerStrategy)
    assert isinstance(create_strategy(EpisodicStrategyConfig()), EpisodicStrategy)


def test_create_strategy_unknown_raises():
    from lerobot.rollout import create_strategy

    cfg = MagicMock()
    cfg.type = "bogus"
    with pytest.raises(ValueError, match="Unknown strategy type"):
        create_strategy(cfg)


def test_format_action_chunk_debug_prints_every_action():
    from lerobot.rollout.strategies import format_action_chunk_debug

    raw = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
    processed = raw + 1
    output = format_action_chunk_debug(
        raw,
        processed,
        ["eef_x.pos", "eef_rx.pos", "gripper.pos"],
    )

    assert "predicted 2 actions" in output
    assert "no actions sent to robot" in output
    assert "step 00" in output
    assert "step 01" in output
    assert "raw: [0.1000, 0.2000, 0.3000]" in output
    assert "pos: eef_x.pos=1.1000" in output
    assert "rot: eef_rx.pos=1.2000" in output
    assert "gripper: gripper.pos=1.3000" in output


def test_action_debug_strategy_predicts_chunk_without_sending(monkeypatch, capsys):
    from lerobot.rollout import ActionDebugStrategy, ActionDebugStrategyConfig
    from lerobot.rollout.strategies import action_debug as action_debug_module

    class FakePipeline:
        def __init__(self, transform=None):
            self.transform = transform or (lambda value: value)
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

        def __call__(self, value):
            return self.transform(value)

    class FakePolicy:
        config = SimpleNamespace(use_amp=False)

        def __init__(self):
            self.reset_count = 0
            self.observation = None

        def reset(self):
            self.reset_count += 1

        def predict_action_chunk(self, observation):
            self.observation = observation
            return torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])

    class FakeRobot:
        robot_type = "mock"

        def __init__(self):
            self.sent = []

        def get_observation(self):
            return {"x.pos": 1.0}

        def send_action(self, action):
            self.sent.append(action)

    monkeypatch.setattr(
        action_debug_module,
        "build_dataset_frame",
        lambda features, observation, prefix: {"observation.state": observation["x.pos"]},
    )
    monkeypatch.setattr(
        action_debug_module,
        "prepare_observation_for_inference",
        lambda observation, device, task, robot_type: observation,
    )

    policy = FakePolicy()
    preprocessor = FakePipeline()
    postprocessor = FakePipeline(lambda actions: actions + 1)
    robot = FakeRobot()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(device="cpu", task="inspect"),
            visual_prompt_recolorer=None,
        ),
        hardware=SimpleNamespace(robot_wrapper=robot),
        policy=SimpleNamespace(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        ),
        processors=SimpleNamespace(robot_observation_processor=lambda observation: observation),
        data=SimpleNamespace(
            hw_features={},
            ordered_action_keys=["x.pos", "gripper.pos"],
        ),
    )
    strategy = ActionDebugStrategy(ActionDebugStrategyConfig())

    strategy.setup(ctx)
    strategy.run(ctx)

    output = capsys.readouterr().out
    assert "predicted 2 actions" in output
    assert "step 00" in output
    assert "step 01" in output
    assert policy.observation["task"] == ["inspect"]
    assert robot.sent == []


# ---------------------------------------------------------------------------
# Inference factory
# ---------------------------------------------------------------------------


def test_create_inference_engine_sync():
    from lerobot.rollout import SyncInferenceConfig, SyncInferenceEngine, create_inference_engine

    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=MagicMock(),
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        robot_wrapper=MagicMock(robot_type="mock"),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["k"],
        task="test",
        fps=30.0,
        device="cpu",
    )
    assert isinstance(engine, SyncInferenceEngine)


def test_human_in_loop_engine_reuses_chunk_until_next(monkeypatch, capsys):
    from lerobot.rollout import HumanInLoopInferenceEngine

    class Policy:
        config = SimpleNamespace(use_amp=False)

        def __init__(self):
            self.calls = 0

        def predict_action_chunk(self, observation):
            self.calls += 1
            return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        def reset(self):
            pass

    policy = Policy()
    robot = MagicMock()
    engine = HumanInLoopInferenceEngine(
        policy=policy,
        preprocessor=MagicMock(side_effect=lambda value: value),
        postprocessor=MagicMock(side_effect=lambda value: value),
        robot_wrapper=robot,
        ordered_action_keys=["x.pos", "gripper.pos"],
        task="pick",
        device="cpu",
        robot_type="mock",
        gripper_width_offset=0.5,
        shutdown_event=Event(),
    )
    responses = iter(["offset 0.25", "1", "offset=1.0", "1", "next", "0"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))

    zero_obs = {"observation.state": torch.zeros(2).numpy()}
    one_obs = {"observation.state": torch.ones(2).numpy()}
    assert torch.equal(engine.get_action(zero_obs), torch.tensor([3.0, 3.75]))
    assert torch.equal(engine.get_action(zero_obs), torch.tensor([3.0, 3.0]))
    assert engine.get_action(zero_obs) is None
    assert torch.equal(engine.get_action(one_obs), torch.tensor([1.0, 1.0]))
    assert policy.calls == 2
    assert robot.set_action_reference_from_state.call_count == 2
    assert robot.set_action_reference_to_current_pose.call_count == 0
    assert robot.clear_action_reference.call_count == 1
    assert "Gripper width offset updated to 0.250000 m" in capsys.readouterr().out


def test_create_rtc_engine_resolves_gripper_lookahead_channel():
    from lerobot.rollout import RTCInferenceConfig, RTCInferenceEngine, create_inference_engine

    pipeline = MagicMock()
    pipeline.steps = []
    engine = create_inference_engine(
        RTCInferenceConfig(gripper_lookahead_steps=2),
        policy=MagicMock(),
        preprocessor=pipeline,
        postprocessor=pipeline,
        robot_wrapper=MagicMock(robot_type="mock", action_features={}),
        hw_features={},
        dataset_features={},
        ordered_action_keys=["eef_x.pos", "eef_y.pos", "gripper.pos"],
        task="test",
        fps=30.0,
        device="cpu",
    )

    assert isinstance(engine, RTCInferenceEngine)
    assert engine._gripper_action_indices == [2]


def test_rtc_inference_config_validates_timed_playback():
    from lerobot.rollout import RTCInferenceConfig

    cfg = RTCInferenceConfig(action_interval_s=0.56, action_replan_interval=1)
    assert cfg.action_interval_s == pytest.approx(0.56)

    with pytest.raises(ValueError, match="action_interval_s must be > 0"):
        RTCInferenceConfig(action_interval_s=0)
    with pytest.raises(ValueError, match="action_replan_interval must be > 0"):
        RTCInferenceConfig(action_replan_interval=0)
    with pytest.raises(ValueError, match="gripper_lookahead_steps must be >= 0"):
        RTCInferenceConfig(gripper_lookahead_steps=-1)


def test_rtc_engine_paces_action_queue(monkeypatch):
    from lerobot.policies.rtc import RTCConfig
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.rollout import RTCInferenceEngine
    from lerobot.rollout.inference import rtc as rtc_module

    clock = [10.0]
    monkeypatch.setattr(rtc_module.time, "perf_counter", lambda: clock[0])

    pipeline = MagicMock()
    pipeline.steps = []
    engine = RTCInferenceEngine(
        policy=MagicMock(),
        preprocessor=pipeline,
        postprocessor=pipeline,
        robot_wrapper=MagicMock(robot_type="mock", action_features={}),
        rtc_config=RTCConfig(),
        hw_features={},
        task="test",
        fps=30,
        device="cpu",
        action_interval_s=0.56,
    )
    engine._action_queue = ActionQueue(RTCConfig())
    actions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    engine._action_queue.merge(actions, actions, real_delay=0)

    assert torch.equal(engine.get_action(None), actions[0])
    clock[0] += 0.55
    assert engine.get_action(None) is None
    clock[0] += 0.01
    assert torch.equal(engine.get_action(None), actions[1])


def test_rtc_engine_uses_future_gripper_channel():
    from lerobot.policies.rtc import RTCConfig
    from lerobot.policies.rtc.action_queue import ActionQueue
    from lerobot.rollout import RTCInferenceEngine

    pipeline = MagicMock()
    pipeline.steps = []
    engine = RTCInferenceEngine(
        policy=MagicMock(),
        preprocessor=pipeline,
        postprocessor=pipeline,
        robot_wrapper=MagicMock(robot_type="mock", action_features={}),
        rtc_config=RTCConfig(),
        hw_features={},
        task="test",
        fps=30,
        device="cpu",
        gripper_lookahead_steps=2,
        gripper_action_indices=[2],
    )
    engine._action_queue = ActionQueue(RTCConfig())
    actions = torch.tensor(
        [
            [0.0, 10.0, 100.0],
            [1.0, 11.0, 101.0],
            [2.0, 12.0, 102.0],
        ]
    )
    engine._action_queue.merge(actions, actions, real_delay=0)

    assert torch.equal(engine.get_action(None), torch.tensor([0.0, 10.0, 102.0]))


def test_rtc_timed_playback_converts_latency_to_action_steps():
    from lerobot.rollout.inference.rtc import _latency_to_action_steps

    assert _latency_to_action_steps(0.0, 0.56) == 0
    assert _latency_to_action_steps(0.55, 0.56) == 1
    assert _latency_to_action_steps(0.57, 0.56) == 2


# ---------------------------------------------------------------------------
# Action confirmation
# ---------------------------------------------------------------------------


class _FakeInference:
    def __init__(self):
        self.action = torch.tensor([0.1, 0.2], dtype=torch.float32)

    def get_action(self, obs_frame):
        return self.action


class _FakeRobotWrapper:
    def __init__(self):
        self.sent = []

    def send_action(self, action):
        self.sent.append(action)
        return action


def _make_send_action_ctx(confirm_each_action: bool, log_controller_actions: bool = False):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            cfg=SimpleNamespace(
                confirm_each_action=confirm_each_action,
                log_controller_actions=log_controller_actions,
            ),
            shutdown_event=Event(),
        ),
        policy=SimpleNamespace(inference=_FakeInference()),
        data=SimpleNamespace(
            dataset_features={},
            ordered_action_keys=["x.pos", "y.pos"],
        ),
        processors=SimpleNamespace(
            robot_action_processor=lambda action_and_obs: action_and_obs[0],
        ),
        hardware=SimpleNamespace(robot_wrapper=_FakeRobotWrapper()),
    )


def test_send_next_action_confirms_before_dispatch(monkeypatch):
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx = _make_send_action_ctx(confirm_each_action=True)
    monkeypatch.setattr("builtins.input", lambda _: "y")

    action = send_next_action({}, {}, ctx, ActionInterpolator())

    assert action == {"x.pos": pytest.approx(0.1), "y.pos": pytest.approx(0.2)}
    assert ctx.hardware.robot_wrapper.sent == [action]


def test_send_next_action_can_skip_confirmed_action(monkeypatch):
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx = _make_send_action_ctx(confirm_each_action=True)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    action = send_next_action({}, {}, ctx, ActionInterpolator())

    assert action is None
    assert ctx.hardware.robot_wrapper.sent == []
    assert not ctx.runtime.shutdown_event.is_set()


def test_send_next_action_confirmation_can_quit(monkeypatch):
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    ctx = _make_send_action_ctx(confirm_each_action=True)
    monkeypatch.setattr("builtins.input", lambda _: "q")

    action = send_next_action({}, {}, ctx, ActionInterpolator())

    assert action is None
    assert ctx.hardware.robot_wrapper.sent == []
    assert ctx.runtime.shutdown_event.is_set()


def test_realman_pika_action_confirmation_prints_offset_debug(capsys, monkeypatch):
    from lerobot.rollout.strategies.core import _confirm_action

    ctx = _make_send_action_ctx(confirm_each_action=True)
    obs = {
        "eef_x.pos": 0.1,
        "eef_y.pos": 0.2,
        "eef_z.pos": 0.3,
        "eef_rx.pos": 0.0,
        "eef_ry.pos": 0.0,
        "eef_rz.pos": 0.0,
        "gripper.pos": 0.04,
    }
    action = {
        "eef_x.pos": 0.11,
        "eef_y.pos": 0.18,
        "eef_z.pos": 0.33,
        "eef_rx.pos": 0.01,
        "eef_ry.pos": -0.02,
        "eef_rz.pos": 0.03,
        "gripper.pos": 0.05,
    }
    monkeypatch.setattr("builtins.input", lambda _: "y")

    assert _confirm_action(ctx, action, obs) is True

    output = capsys.readouterr().out
    assert "[ActionDebug:rollout]" in output
    assert "model_pika_relative" in output
    assert "robot_realman_tcp_relative" in output
    assert "pika_absolute" not in output
    assert "gripper: 0.0500" in output


def test_send_next_action_logs_colorized_controller_action(capsys):
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.action_interpolator import ActionInterpolator

    action = {
        "eef_x.pos": 0.11,
        "eef_y.pos": 0.18,
        "eef_z.pos": 0.03,
        "eef_rx.pos": 0.01,
        "eef_ry.pos": -0.02,
        "eef_rz.pos": 0.03,
        "gripper.pos": 0.05,
    }
    ctx = _make_send_action_ctx(confirm_each_action=False, log_controller_actions=True)
    ctx.policy.inference.action = torch.tensor(list(action.values()), dtype=torch.float32)
    ctx.data.ordered_action_keys = list(action)

    assert send_next_action({}, {}, ctx, ActionInterpolator()) == pytest.approx(action)

    output = capsys.readouterr().out
    assert "[ActionDebug:controller]" in output
    assert "[sent]" in output
    assert "\033[92mpos:" in output
    assert "\033[96mrot:" in output
    assert "\033[93mgripper:" in output
    assert len(ctx.hardware.robot_wrapper.sent) == 1
    assert ctx.hardware.robot_wrapper.sent[0] == pytest.approx(action)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def test_estimate_max_episode_seconds_no_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    assert estimate_max_episode_seconds({}, fps=30.0) == 300.0


def test_estimate_max_episode_seconds_with_video():
    from lerobot.rollout.strategies import estimate_max_episode_seconds

    features = {"cam": {"dtype": "video", "shape": (480, 640, 3)}}
    result = estimate_max_episode_seconds(features, fps=30.0)
    assert result > 0
    # With a real camera, duration should differ from the fallback
    assert result != 300.0


def test_safe_push_to_hub():
    from lerobot.rollout.strategies import safe_push_to_hub

    ds = MagicMock()
    ds.num_episodes = 0
    assert safe_push_to_hub(ds) is False
    ds.push_to_hub.assert_not_called()

    ds.num_episodes = 5
    assert safe_push_to_hub(ds, tags=["test"]) is True
    ds.push_to_hub.assert_called_once_with(tags=["test"], private=False)


# ---------------------------------------------------------------------------
# DAgger state machine
# ---------------------------------------------------------------------------


def test_dagger_full_transition_cycle():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    assert events.phase == DAggerPhase.AUTONOMOUS

    # AUTONOMOUS -> PAUSED
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.AUTONOMOUS, DAggerPhase.PAUSED)

    # PAUSED -> CORRECTING
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.CORRECTING)

    # CORRECTING -> PAUSED
    events.request_transition("correction")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.CORRECTING, DAggerPhase.PAUSED)

    # PAUSED -> AUTONOMOUS
    events.request_transition("pause_resume")
    old, new = events.consume_transition()
    assert (old, new) == (DAggerPhase.PAUSED, DAggerPhase.AUTONOMOUS)


def test_dagger_invalid_transition_ignored():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("correction")  # Not valid from AUTONOMOUS
    assert events.consume_transition() is None
    assert events.phase == DAggerPhase.AUTONOMOUS


def test_dagger_events_reset():
    from lerobot.rollout.strategies import DAggerEvents, DAggerPhase

    events = DAggerEvents()
    events.request_transition("pause_resume")
    events.consume_transition()  # -> PAUSED
    events.upload_requested.set()
    events.reset()
    assert events.phase == DAggerPhase.AUTONOMOUS
    assert not events.upload_requested.is_set()


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------


def test_rollout_context_fields():
    from lerobot.rollout import RolloutContext

    field_names = {f.name for f in dataclasses.fields(RolloutContext)}
    assert field_names == {"runtime", "hardware", "policy", "processors", "data"}
