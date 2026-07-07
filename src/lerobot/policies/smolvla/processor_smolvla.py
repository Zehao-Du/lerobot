#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NewLineTaskProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE, POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_smolvla import SmolVLAConfig


def _skew(vector: Tensor) -> Tensor:
    zero = torch.zeros_like(vector[..., 0])
    x, y, z = vector.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def _rotvec_to_matrix(rotvec: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / angle.clamp_min(torch.finfo(rotvec.dtype).eps)
    axis_skew = _skew(axis)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    eye = eye.expand(*rotvec.shape[:-1], 3, 3)
    angle = angle[..., None]
    return eye + torch.sin(angle) * axis_skew + (1.0 - torch.cos(angle)) * (axis_skew @ axis_skew)


def _matrix_to_rotvec(matrix: Tensor) -> Tensor:
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    vee = torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.sin(angle)
    scale = angle / (2.0 * sin_angle).clamp_min(torch.finfo(matrix.dtype).eps)
    rotvec = scale[..., None] * vee
    small = angle.abs() < 1e-6
    return torch.where(small[..., None], 0.5 * vee, rotvec)


def _latest_state(state: Tensor) -> Tensor:
    return state[:, -1, :] if state.ndim > 2 else state


def _absolute_ee_pose_to_relative(pose: Tensor, base_pose: Tensor) -> Tensor:
    if pose.shape[-1] < 6:
        raise ValueError(f"Expected at least 6 pose dimensions, got {pose.shape[-1]}.")

    if base_pose.device != pose.device or base_pose.dtype != pose.dtype:
        base_pose = base_pose.to(device=pose.device, dtype=pose.dtype)

    pose_xyz = pose[..., :3]
    pose_rotvec = pose[..., 3:6]
    base_xyz = base_pose[..., :3]
    base_rotvec = base_pose[..., 3:6]
    if pose.ndim == 3 and base_pose.ndim == 2:
        base_xyz = base_xyz.unsqueeze(1)
        base_rotvec = base_rotvec.unsqueeze(1)

    pose_rot = _rotvec_to_matrix(pose_rotvec)
    base_rot = _rotvec_to_matrix(base_rotvec)
    base_rot_t = base_rot.transpose(-1, -2)
    rel_xyz = torch.matmul(base_rot_t, (pose_xyz - base_xyz).unsqueeze(-1)).squeeze(-1)
    rel_rot = base_rot_t @ pose_rot
    rel_pose = pose.clone()
    rel_pose[..., :3] = rel_xyz
    rel_pose[..., 3:6] = _matrix_to_rotvec(rel_rot)
    return rel_pose


def _relative_ee_pose_to_absolute(pose: Tensor, base_pose: Tensor) -> Tensor:
    if pose.shape[-1] < 6:
        raise ValueError(f"Expected at least 6 pose dimensions, got {pose.shape[-1]}.")

    if base_pose.device != pose.device or base_pose.dtype != pose.dtype:
        base_pose = base_pose.to(device=pose.device, dtype=pose.dtype)

    rel_xyz = pose[..., :3]
    rel_rotvec = pose[..., 3:6]
    base_xyz = base_pose[..., :3]
    base_rotvec = base_pose[..., 3:6]
    if pose.ndim == 3 and base_pose.ndim == 2:
        base_xyz = base_xyz.unsqueeze(1)
        base_rotvec = base_rotvec.unsqueeze(1)

    rel_rot = _rotvec_to_matrix(rel_rotvec)
    base_rot = _rotvec_to_matrix(base_rotvec)
    abs_xyz = base_xyz + torch.matmul(base_rot, rel_xyz.unsqueeze(-1)).squeeze(-1)
    abs_rot = base_rot @ rel_rot
    abs_pose = pose.clone()
    abs_pose[..., :3] = abs_xyz
    abs_pose[..., 3:6] = _matrix_to_rotvec(abs_rot)
    return abs_pose


@ProcessorStepRegistry.register(name="smolvla_relative_ee_processor")
@dataclass
class SmolVLARelativeEEProcessorStep(ProcessorStep):
    enabled_actions: bool = False
    enabled_states: bool = False
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is None:
            return transition

        base_state = _latest_state(state)
        self._last_state = base_state.detach().clone()

        if not self.enabled_actions and not self.enabled_states:
            return transition

        new_transition = transition.copy()
        new_observation = observation.copy()

        if self.enabled_states:
            new_observation[OBS_STATE] = _absolute_ee_pose_to_relative(state, base_state)
            new_transition[TransitionKey.OBSERVATION] = new_observation

        action = transition.get(TransitionKey.ACTION)
        if self.enabled_actions and action is not None:
            new_transition[TransitionKey.ACTION] = _absolute_ee_pose_to_relative(action, base_state)

        return new_transition

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {"enabled_actions": self.enabled_actions, "enabled_states": self.enabled_states}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="smolvla_absolute_ee_action_processor")
@dataclass
class SmolVLAAbsoluteEEActionProcessorStep(ProcessorStep):
    enabled: bool = False
    relative_step: SmolVLARelativeEEProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError("SmolVLAAbsoluteEEActionProcessorStep requires a paired relative_step.")
        base_state = self.relative_step.get_cached_state()
        if base_state is None:
            raise RuntimeError("No cached observation.state found for SmolVLA relative EE action recovery.")

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = _relative_ee_pose_to_absolute(action, base_state)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_ee_step = SmolVLARelativeEEProcessorStep(
        enabled_actions=config.use_relative_ee_actions,
        enabled_states=config.use_relative_ee_states,
    )

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        relative_ee_step,
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        SmolVLAAbsoluteEEActionProcessorStep(
            enabled=config.use_relative_ee_actions, relative_step=relative_ee_step
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def ensure_smolvla_relative_ee_processors(
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    config: SmolVLAConfig,
) -> None:
    """Insert SmolVLA relative EE steps into pretrained processor pipelines when requested."""
    if not config.use_relative_ee_actions and not config.use_relative_ee_states:
        return

    relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, SmolVLARelativeEEProcessorStep)),
        None,
    )
    if relative_step is None:
        relative_step = SmolVLARelativeEEProcessorStep()
        steps = list(preprocessor.steps)
        insert_at = 0
        for idx, step in enumerate(steps):
            if isinstance(step, AddBatchDimensionProcessorStep):
                insert_at = idx + 1
                break
        steps.insert(insert_at, relative_step)
        preprocessor.steps = steps

    relative_step.enabled_actions = config.use_relative_ee_actions
    relative_step.enabled_states = config.use_relative_ee_states

    absolute_step = next(
        (step for step in postprocessor.steps if isinstance(step, SmolVLAAbsoluteEEActionProcessorStep)),
        None,
    )
    if absolute_step is None:
        absolute_step = SmolVLAAbsoluteEEActionProcessorStep()
        steps = list(postprocessor.steps)
        insert_at = len(steps)
        for idx, step in enumerate(steps):
            if isinstance(step, DeviceProcessorStep):
                insert_at = idx
                break
        steps.insert(insert_at, absolute_step)
        postprocessor.steps = steps

    absolute_step.enabled = config.use_relative_ee_actions
    absolute_step.relative_step = relative_step
