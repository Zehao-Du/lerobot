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

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.realman_pika.config_realman_pika import (
    DEFAULT_FISHEYE_DEVICE,
    DEFAULT_REALSENSE_SERIAL,
    RealmanPikaConfig,
)
from lerobot.utils.sam3_recolor import DEFAULT_PINK_BLOCK_PROMPTS, Sam3PinkBlockRecolorer

logger = logging.getLogger(__name__)


def capture_and_recolor(
    checkpoint: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    alpha: float = 0.9,
    preserve_luminance: bool = True,
    warmup_sec: float = 1.0,
    rgb_serial: str = DEFAULT_REALSENSE_SERIAL,
    fisheye_device: str = DEFAULT_FISHEYE_DEVICE,
) -> dict[str, Path]:
    cfg = RealmanPikaConfig(
        gripper_serial_port="/dev/ttyUSB60",
    )
    cfg.cameras["rgb"].serial_number_or_name = rgb_serial
    cfg.cameras["fisheye"].index_or_path = Path(fisheye_device)
    cameras = make_cameras_from_configs(cfg.cameras)
    output_dir.mkdir(parents=True, exist_ok=True)

    recolorer = Sam3PinkBlockRecolorer(
        checkpoint,
        device=device,
        threshold=threshold,
        mask_threshold=mask_threshold,
        alpha=alpha,
        preserve_luminance=preserve_luminance,
    )

    try:
        for name, cam in cameras.items():
            logger.info("Connecting camera %s", name)
            cam.connect()
        if warmup_sec > 0:
            time.sleep(warmup_sec)

        outputs: dict[str, Path] = {}
        for name, cam in cameras.items():
            image_rgb = cam.read_latest()
            recolored, mask = recolorer.recolor_image(image_rgb)

            input_path = output_dir / f"{name}_input.png"
            output_path = output_dir / f"{name}_output.png"
            mask_path = output_dir / f"{name}_mask.png"
            Image.fromarray(image_rgb, mode="RGB").save(input_path)
            Image.fromarray(recolored, mode="RGB").save(output_path)
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

            outputs[f"{name}_input"] = input_path
            outputs[f"{name}_output"] = output_path
            outputs[f"{name}_mask"] = mask_path
            logger.info("Saved %s input/output/mask to %s", name, output_dir)
        return outputs
    finally:
        for cam in cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture rgb+fisheye images, recolor the pink block to blue with SAM3.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/ubuntu/Documents/CodeField/zehao/lerobot/outputs/sam3"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--no-preserve-luminance", action="store_true")
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument("--rgb-serial", default=DEFAULT_REALSENSE_SERIAL)
    parser.add_argument("--fisheye-device", default=DEFAULT_FISHEYE_DEVICE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp") / "sam3_recolor_test",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    outputs = capture_and_recolor(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        alpha=args.alpha,
        preserve_luminance=not args.no_preserve_luminance,
        warmup_sec=args.warmup_sec,
        rgb_serial=args.rgb_serial,
        fisheye_device=args.fisheye_device,
    )
    print("Saved files:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
