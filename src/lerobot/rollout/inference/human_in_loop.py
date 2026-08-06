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

"""Interactive action-chunk inference controlled from the terminal."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from copy import copy
from threading import Event

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.sam3_recolor import Sam3PinkBlockRecolorer

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)


def format_human_action_chunk(actions: torch.Tensor, action_keys: list[str]) -> str:
    """Format a processed ``[T, A]`` action chunk for terminal selection."""
    chunk = actions.detach().cpu()
    if chunk.ndim == 3:
        if chunk.shape[0] != 1:
            raise ValueError(f"Human-in-loop inference expects batch size 1, got {tuple(chunk.shape)}")
        chunk = chunk.squeeze(0)
    if chunk.ndim != 2:
        raise ValueError(f"Human-in-loop actions must have shape [T, A], got {tuple(chunk.shape)}")
    if chunk.shape[1] != len(action_keys):
        raise ValueError(
            f"Action dimension ({chunk.shape[1]}) does not match action keys ({len(action_keys)})"
        )

    lines = [
        f"\n[HumanInLoop] Model predicted {len(chunk)} actions (all relative to one fixed TCP reference):"
    ]
    for index, action in enumerate(chunk):
        values = ", ".join(
            f"{key}={float(value):.4f}" for key, value in zip(action_keys, action, strict=True)
        )
        lines.append(f"  [{index:02d}] {values}")
    lines.append("Commands: <index>=execute, offset <meters>=change gripper offset, next=infer, quit=stop.")
    return "\n".join(lines)


class HumanInLoopInferenceEngine(InferenceEngine):
    """Predict a full chunk, then let a human repeatedly choose actions from it."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        ordered_action_keys: list[str],
        task: str,
        device: str | None,
        robot_type: str,
        gripper_width_offset: float = 0.0,
        shutdown_event: Event | None = None,
        visual_prompt_recolorer: Sam3PinkBlockRecolorer | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._device = torch.device(device or "cpu")
        self._robot_type = robot_type
        self._gripper_width_offset = gripper_width_offset
        self._gripper_action_indices = [
            index for index, key in enumerate(ordered_action_keys) if "gripper" in key.lower()
        ]
        if not math.isfinite(gripper_width_offset):
            raise ValueError("gripper_width_offset must be finite")
        if gripper_width_offset != 0.0 and not self._gripper_action_indices:
            raise ValueError("gripper_width_offset requires an action key containing 'gripper'")
        self._shutdown_event = shutdown_event
        self._visual_prompt_recolorer = visual_prompt_recolorer
        self._base_actions: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None

    def start(self) -> None:
        logger.info("Human-in-loop inference started; waiting for terminal action selection")

    def stop(self) -> None:
        self._robot.clear_action_reference()
        logger.info("Human-in-loop inference stopped")

    def reset(self) -> None:
        self._robot.clear_action_reference()
        self._base_actions = None
        self._actions = None
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def _predict_action_chunk(self, obs_frame: dict) -> None:
        # Anchor the chunk to the exact state snapshot the model will consume.
        # A separate hardware read here could drift while the arm is moving.
        state = obs_frame.get(OBS_STATE)
        if state is None:
            self._robot.set_action_reference_to_current_pose()
        else:
            self._robot.set_action_reference_from_state(state)
        observation = copy(obs_frame)
        autocast_ctx = (
            torch.autocast(device_type=self._device.type)
            if self._device.type == "cuda" and self._policy.config.use_amp
            else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            if self._visual_prompt_recolorer is not None:
                observation = self._visual_prompt_recolorer.recolor_observation_images(observation)
            observation = prepare_observation_for_inference(
                observation, self._device, self._task, self._robot_type
            )
            observation = self._preprocessor(observation)
            actions = self._policy.predict_action_chunk(observation)
            actions = self._postprocessor(actions)

        self._base_actions = actions.squeeze(0).detach().cpu().clone()
        self._apply_gripper_width_offset()
        print(format_human_action_chunk(self._actions, self._ordered_action_keys), flush=True)

    def _apply_gripper_width_offset(self) -> None:
        if self._base_actions is None:
            self._actions = None
            return
        self._actions = self._base_actions.clone()
        if self._gripper_width_offset != 0.0:
            self._actions[:, self._gripper_action_indices] -= self._gripper_width_offset

    def _set_gripper_width_offset(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("gripper width offset must be finite")
        if value != 0.0 and not self._gripper_action_indices:
            raise ValueError("gripper width offset requires an action key containing 'gripper'")
        self._gripper_width_offset = value
        self._apply_gripper_width_offset()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        if obs_frame is None:
            return None
        if self._actions is None:
            self._predict_action_chunk(obs_frame)

        while True:
            try:
                response = input("human-in-loop> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                logger.warning("Terminal input closed; requesting rollout shutdown")
                if self._shutdown_event is not None:
                    self._shutdown_event.set()
                return None

            if response in {"next", "n"}:
                self._robot.clear_action_reference()
                self._base_actions = None
                self._actions = None
                return None
            if response in {"quit", "q", "exit"}:
                self._robot.clear_action_reference()
                if self._shutdown_event is not None:
                    self._shutdown_event.set()
                return None
            if response == "offset":
                print(f"Current gripper width offset: {self._gripper_width_offset:.6f} m", flush=True)
                continue
            if response.startswith("offset ") or response.startswith("offset="):
                value_text = response.removeprefix("offset").lstrip(" =")
                try:
                    self._set_gripper_width_offset(float(value_text))
                except ValueError as error:
                    print(f"Invalid offset: {error}", flush=True)
                    continue
                print(f"Gripper width offset updated to {self._gripper_width_offset:.6f} m.", flush=True)
                print(format_human_action_chunk(self._actions, self._ordered_action_keys), flush=True)
                continue
            try:
                index = int(response)
            except ValueError:
                print("Enter an action index, 'offset <meters>', 'next', or 'quit'.", flush=True)
                continue
            if not 0 <= index < len(self._actions):
                print(f"Action index must be between 0 and {len(self._actions) - 1}.", flush=True)
                continue
            return self._actions[index].clone()
