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
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.utils.constants import OBS_STATE

from .configuration_smolvla import SmolVLAConfig


def _skew(rotvec: torch.Tensor) -> torch.Tensor:
    x, y, z = rotvec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    theta2 = theta * theta
    small = theta < 1e-6
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta.clamp_min(1e-12))
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12))
    k = _skew(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(rotvec.shape[:-1] + (3, 3))
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def _matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    trace = matrix.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    scale = torch.where(
        theta < 1e-6,
        0.5 + theta * theta / 12.0,
        theta / (2.0 * sin_theta.clamp_min(1e-12)),
    )
    return scale[..., None] * vee


def to_relative_ee_actions(actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Convert absolute EE actions to state-relative EE actions.

    Dimensions 0:3 are translational offsets. Dimensions 3:6 are rotvecs
    converted by ``R_relative = R_state^-1 @ R_action``. Dimension 6
    (gripper width) remains absolute.
    """

    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    out = actions.clone()
    anchor_pos = state[..., :3]
    anchor_rot = state[..., 3:6]
    if actions.ndim == state.ndim + 1:
        anchor_pos = anchor_pos.unsqueeze(-2)

    out[..., :3] -= anchor_pos
    action_rot = _rotvec_to_matrix(out[..., 3:6])
    anchor_rot_matrix = _rotvec_to_matrix(anchor_rot)
    if actions.ndim == state.ndim + 1:
        anchor_rot_matrix = anchor_rot_matrix.unsqueeze(-3)
    relative_rot = anchor_rot_matrix.transpose(-1, -2) @ action_rot
    out[..., 3:6] = _matrix_to_rotvec(relative_rot)
    return out


def to_absolute_ee_actions(actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Convert state-relative EE actions back to absolute EE actions."""

    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    out = actions.clone()
    anchor_pos = state[..., :3]
    anchor_rot = state[..., 3:6]
    if actions.ndim == state.ndim + 1:
        anchor_pos = anchor_pos.unsqueeze(-2)

    out[..., :3] += anchor_pos
    relative_rot = _rotvec_to_matrix(out[..., 3:6])
    anchor_rot_matrix = _rotvec_to_matrix(anchor_rot)
    if actions.ndim == state.ndim + 1:
        anchor_rot_matrix = anchor_rot_matrix.unsqueeze(-3)
    absolute_rot = anchor_rot_matrix @ relative_rot
    out[..., 3:6] = _matrix_to_rotvec(absolute_rot)
    return out


@ProcessorStepRegistry.register(name="smolvla_relative_ee_processor")
@dataclass(init=False)
class SmolVLARelativeEEProcessorStep(ProcessorStep):
    enabled: bool = False
    use_relative_states: bool = False
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __init__(
        self,
        enabled: bool | None = None,
        use_relative_states: bool | None = None,
        enabled_actions: bool | None = None,
        enabled_states: bool | None = None,
    ) -> None:
        self.enabled = bool(enabled if enabled is not None else enabled_actions)
        self.use_relative_states = bool(
            use_relative_states if use_relative_states is not None else enabled_states
        )
        self._last_state = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state

        if not self.enabled and not self.use_relative_states:
            return transition

        new_transition = transition.copy()
        if self.enabled and state is not None:
            action = new_transition.get(TransitionKey.ACTION)
            if action is not None:
                new_transition[TransitionKey.ACTION] = to_relative_ee_actions(action, state)

        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "use_relative_states": self.use_relative_states}

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
        if self.relative_step.use_relative_states:
            return transition
        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError("SmolVLAAbsoluteEEActionProcessorStep has no cached observation.state.")

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = to_absolute_ee_actions(action, cached_state)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def reconnect_smolvla_ee_processors(
    preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
) -> None:
    relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, SmolVLARelativeEEProcessorStep)),
        None,
    )
    if relative_step is None:
        return
    for step in postprocessor.steps:
        if isinstance(step, SmolVLAAbsoluteEEActionProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step


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
        enabled=config.use_relative_ee_actions,
        use_relative_states=config.use_relative_ee_states,
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        NewLineTaskProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        relative_ee_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        SmolVLAAbsoluteEEActionProcessorStep(
            enabled=config.use_relative_ee_actions,
            relative_step=relative_ee_step,
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
