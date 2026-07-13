import torch

from lerobot.policies.smolvla.processor_smolvla import (
    SmolVLAAbsoluteEEActionProcessorStep,
    SmolVLARelativeEEProcessorStep,
    to_absolute_ee_actions,
    to_relative_ee_actions,
)
from lerobot.processor import PolicyProcessorPipeline, ProcessorStepRegistry
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


def test_smolvla_ee_registry_names_deserialize():
    config = {
        "name": "test",
        "steps": [
            {
                "registry_name": "smolvla_relative_ee_processor",
                "config": {"enabled": True, "use_relative_states": True},
            },
            {
                "registry_name": "smolvla_absolute_ee_action_processor",
                "config": {"enabled": True},
            },
        ],
    }

    pipeline = PolicyProcessorPipeline.from_config(config)

    assert "smolvla_relative_ee_processor" in ProcessorStepRegistry.list()
    assert "smolvla_absolute_ee_action_processor" in ProcessorStepRegistry.list()
    assert isinstance(pipeline.steps[0], SmolVLARelativeEEProcessorStep)
    assert isinstance(pipeline.steps[1], SmolVLAAbsoluteEEActionProcessorStep)


def test_smolvla_relative_ee_processor_deserializes_legacy_flags():
    config = {
        "name": "test",
        "steps": [
            {
                "registry_name": "smolvla_relative_ee_processor",
                "config": {"enabled_actions": True, "enabled_states": True},
            },
        ],
    }

    pipeline = PolicyProcessorPipeline.from_config(config)

    step = pipeline.steps[0]
    assert isinstance(step, SmolVLARelativeEEProcessorStep)
    assert step.enabled is True
    assert step.use_relative_states is True


def test_relative_ee_action_absolute_roundtrip():
    state = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.1, -0.2, 0.3, 0.04],
            [-0.1, 0.2, 0.1, -0.2, 0.1, -0.1, 0.02],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [
            [
                [0.25, -0.08, 0.31, 0.2, -0.1, 0.4, 0.06],
                [0.24, -0.07, 0.32, 0.15, -0.05, 0.45, 0.03],
            ],
            [
                [-0.09, 0.22, 0.13, -0.1, 0.2, -0.2, 0.08],
                [-0.08, 0.21, 0.14, -0.15, 0.15, -0.25, 0.01],
            ],
        ],
        dtype=torch.float32,
    )

    relative = to_relative_ee_actions(actions, state)
    recovered = to_absolute_ee_actions(relative, state)

    torch.testing.assert_close(recovered, actions, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(relative[..., 6], actions[..., 6])


def test_processor_steps_cache_state_and_keep_gripper_absolute():
    state = torch.tensor([[0.2, 0.1, -0.1, 0.05, -0.1, 0.2, 0.04]], dtype=torch.float32)
    action = torch.tensor([[[0.21, 0.12, -0.08, 0.1, -0.05, 0.25, 0.07]]], dtype=torch.float32)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: None,
        TransitionKey.DONE: None,
        TransitionKey.TRUNCATED: None,
        TransitionKey.INFO: None,
        TransitionKey.COMPLEMENTARY_DATA: None,
    }
    relative_step = SmolVLARelativeEEProcessorStep(enabled=True)
    absolute_step = SmolVLAAbsoluteEEActionProcessorStep(enabled=True, relative_step=relative_step)

    relative_transition = relative_step(transition)
    relative_action = relative_transition[TransitionKey.ACTION]
    assert relative_action[..., 6].item() == action[..., 6].item()

    recovered_transition = absolute_step(relative_transition)
    torch.testing.assert_close(recovered_transition[TransitionKey.ACTION], action, atol=1e-5, rtol=1e-5)


def test_relative_ee_state_mode_keeps_model_action_relative():
    state = torch.tensor([[0.2, 0.1, -0.1, 0.05, -0.1, 0.2, 0.04]], dtype=torch.float32)
    action = torch.tensor([[[0.01, 0.02, -0.03, 0.04, -0.05, 0.06, 0.07]]], dtype=torch.float32)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: None,
        TransitionKey.DONE: None,
        TransitionKey.TRUNCATED: None,
        TransitionKey.INFO: None,
        TransitionKey.COMPLEMENTARY_DATA: None,
    }
    relative_step = SmolVLARelativeEEProcessorStep(enabled=False, use_relative_states=True)
    absolute_step = SmolVLAAbsoluteEEActionProcessorStep(enabled=True, relative_step=relative_step)

    preprocessed = relative_step(transition)
    torch.testing.assert_close(preprocessed[TransitionKey.OBSERVATION][OBS_STATE], state)

    postprocessed = absolute_step({TransitionKey.ACTION: action})
    torch.testing.assert_close(postprocessed[TransitionKey.ACTION], action)
