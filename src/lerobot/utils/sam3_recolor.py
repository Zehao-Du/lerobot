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

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_SAM3_CHECKPOINT = Path("/home/ubuntu/Documents/CodeField/zehao/lerobot/outputs/sam3")
DEFAULT_PINK_BLOCK_PROMPTS = (
    "pink block",
    "pink square block",
    "pink cube",
    "pink small cube",
    "pink square",
)
DEFAULT_CAMERA_KEYS = ("rgb", "fisheye", "observation.images.rgb", "observation.images.fisheye")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    component_ids = np.arange(1, num_labels)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    best_component = int(component_ids[np.argmax(component_areas)])
    return labels == best_component


def clean_mask(mask: np.ndarray, min_area: int = 64) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask.copy()
    kernel = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    if cleaned.sum() < min_area:
        return np.zeros_like(cleaned, dtype=bool)
    return _largest_component(cleaned.astype(bool))


def pink_hsv_mask(image_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lower_1 = np.array([135, 40, 40], dtype=np.uint8)
    upper_1 = np.array([179, 255, 255], dtype=np.uint8)
    lower_2 = np.array([125, 20, 80], dtype=np.uint8)
    upper_2 = np.array([170, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_1, upper_1) | cv2.inRange(hsv, lower_2, upper_2)
    return clean_mask(mask > 0)


def recolor_masked_region(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    target_hue: int = 120,
    alpha: float = 0.9,
    preserve_luminance: bool = True,
) -> np.ndarray:
    if not mask.any():
        return image_rgb.copy()

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    recolored_hsv = hsv.copy()
    recolored_hsv[..., 0][mask] = float(target_hue)
    recolored_hsv[..., 1][mask] = np.maximum(recolored_hsv[..., 1][mask], 180.0)
    if not preserve_luminance:
        recolored_hsv[..., 2][mask] = np.maximum(recolored_hsv[..., 2][mask], 200.0)

    recolored_rgb = cv2.cvtColor(recolored_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    out = image_rgb.astype(np.float32).copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * recolored_rgb.astype(np.float32)[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


class Sam3PinkBlockRecolorer:
    def __init__(
        self,
        checkpoint: Path = DEFAULT_SAM3_CHECKPOINT,
        *,
        device: str | None = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        alpha: float = 0.9,
        preserve_luminance: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.alpha = alpha
        self.preserve_luminance = preserve_luminance

        from transformers import Sam3Model, Sam3Processor

        logger.info("Loading SAM3 from %s on %s", self.checkpoint, self.device)
        self.model = Sam3Model.from_pretrained(self.checkpoint, local_files_only=True).to(self.device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(self.checkpoint, local_files_only=True)

    def _segment_prompt_masks(self, image_rgb: np.ndarray, prompts: tuple[str, ...]) -> np.ndarray:
        pil_image = Image.fromarray(image_rgb, mode="RGB")
        masks: list[np.ndarray] = []
        for prompt in prompts:
            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs["original_sizes"].detach().cpu().tolist(),
            )[0]
            result_masks = results.get("masks")
            if result_masks is None:
                continue
            if isinstance(result_masks, torch.Tensor):
                result_masks = result_masks.detach().cpu().numpy()
            result_masks = np.asarray(result_masks, dtype=bool)
            if result_masks.ndim == 2:
                result_masks = result_masks[None, ...]
            for mask in result_masks:
                if mask.any():
                    masks.append(mask)

        if not masks:
            return np.zeros(image_rgb.shape[:2], dtype=bool)
        union_mask = np.any(np.stack(masks, axis=0), axis=0)
        return clean_mask(union_mask)

    def recolor_image(
        self,
        image_rgb: np.ndarray,
        prompts: tuple[str, ...] = DEFAULT_PINK_BLOCK_PROMPTS,
    ) -> tuple[np.ndarray, np.ndarray]:
        sam_mask = self._segment_prompt_masks(image_rgb, prompts)
        pink_mask = pink_hsv_mask(image_rgb)

        if sam_mask.any() and pink_mask.any():
            overlap = sam_mask & pink_mask
            final_mask = clean_mask(overlap) if overlap.any() else pink_mask
        elif pink_mask.any():
            final_mask = pink_mask
        else:
            final_mask = sam_mask

        recolored = recolor_masked_region(
            image_rgb,
            final_mask,
            alpha=self.alpha,
            preserve_luminance=self.preserve_luminance,
        )
        return recolored, final_mask

    def recolor_observation_images(
        self,
        observation: dict,
        image_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS,
    ) -> dict:
        out = dict(observation)
        for key, image in list(out.items()):
            if key not in image_keys and not any(key.endswith(suffix) for suffix in ("rgb", "fisheye")):
                continue
            image_np = np.asarray(image)
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                continue
            recolored, _ = self.recolor_image(image_np)
            out[key] = recolored
        return out
