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

"""One-shot action-chunk inspection strategy that never commands the robot."""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch

from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.feature_utils import build_dataset_frame

from ..context import RolloutContext
from .core import _ACTION_COLORS, RolloutStrategy

logger = logging.getLogger(__name__)

_ROTATION_MARKERS = ("rx", "ry", "rz", "wx", "wy", "wz", "roll", "pitch", "yaw", "quat", "rot")


def _action_group(key: str) -> str:
    normalized = key.lower()
    if "gripper" in normalized:
        return "gripper"
    if any(marker in normalized for marker in _ROTATION_MARKERS):
        return "rot"
    return "pos"


def format_action_chunk_debug(
    raw_actions: torch.Tensor,
    processed_actions: torch.Tensor,
    action_keys: list[str],
) -> str:
    """Format every predicted action with colorized semantic groups."""
    raw = raw_actions.detach().cpu()
    processed = processed_actions.detach().cpu()
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise ValueError(f"Action debug expects batch size 1, got shape={tuple(raw.shape)}")
        raw = raw.squeeze(0)
    if processed.ndim == 3:
        if processed.shape[0] != 1:
            raise ValueError(
                f"Processed action debug expects batch size 1, got shape={tuple(processed.shape)}"
            )
        processed = processed.squeeze(0)
    if raw.ndim != 2 or processed.ndim != 2:
        raise ValueError(
            f"Action chunks must be [T, A], got raw={tuple(raw.shape)}, processed={tuple(processed.shape)}"
        )
    if raw.shape != processed.shape:
        raise ValueError(
            f"Raw and processed action shapes differ: {tuple(raw.shape)} != {tuple(processed.shape)}"
        )
    if processed.shape[1] != len(action_keys):
        raise ValueError(
            f"Action dimension ({processed.shape[1]}) does not match action keys ({len(action_keys)})"
        )

    colors = _ACTION_COLORS
    lines = [
        f"{colors['header']}[ActionChunkDebug] predicted {len(processed)} actions "
        f"(no actions sent to robot){colors['reset']}"
    ]
    for step, (raw_action, processed_action) in enumerate(zip(raw, processed, strict=True)):
        raw_values = ", ".join(f"{float(value):.4f}" for value in raw_action)
        grouped: dict[str, list[str]] = {"pos": [], "rot": [], "gripper": []}
        for key, value in zip(action_keys, processed_action, strict=True):
            grouped[_action_group(key)].append(f"{key}={float(value):.4f}")
        components = [
            f"{colors[group]}{group}: {', '.join(values)}{colors['reset']}"
            for group, values in grouped.items()
            if values
        ]
        lines.append(
            f"{colors['step']}step {step:02d}{colors['reset']} | raw: [{raw_values}] | {' '.join(components)}"
        )
    return "\n".join(lines)


class ActionDebugStrategy(RolloutStrategy):
    """Capture one current observation and print the complete policy action chunk."""

    def setup(self, ctx: RolloutContext) -> None:
        ctx.policy.policy.reset()
        ctx.policy.preprocessor.reset()
        ctx.policy.postprocessor.reset()
        logger.info("Action debug ready; robot commands are disabled")

    def run(self, ctx: RolloutContext) -> None:
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        policy = ctx.policy.policy
        device = torch.device(cfg.device or "cpu")

        obs_raw = robot.get_observation()
        obs_processed = ctx.processors.robot_observation_processor(obs_raw)
        recolorer = ctx.runtime.visual_prompt_recolorer
        if recolorer is not None:
            obs_processed = recolorer.recolor_observation_images(obs_processed)

        observation = build_dataset_frame(ctx.data.hw_features, obs_processed, prefix="observation")
        observation = prepare_observation_for_inference(
            observation,
            device,
            cfg.task,
            robot.robot_type,
        )
        observation["task"] = [cfg.task]
        observation = ctx.policy.preprocessor(observation)

        autocast_ctx = (
            torch.autocast(device_type=device.type)
            if device.type == "cuda" and policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            raw_actions = policy.predict_action_chunk(observation)
            raw_actions_for_display = raw_actions.detach().clone()
            processed_actions = ctx.policy.postprocessor(raw_actions)

        print(
            format_action_chunk_debug(
                raw_actions_for_display,
                processed_actions,
                ctx.data.ordered_action_keys,
            ),
            flush=True,
        )

    def teardown(self, ctx: RolloutContext) -> None:
        self._teardown_hardware(ctx.hardware, return_to_initial_position=False)
        logger.info("Action debug complete; no actions were sent")
