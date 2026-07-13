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

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np

logger = logging.getLogger(__name__)


class RolloutCameraRecorder:
    """Lightweight rollout camera recorder.

    H.264 mode is directly playable. Raw RGB mode is lower overhead during
    control but must be converted after rollout.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        camera_keys: list[str],
        fps: float,
        filename_prefix: str = "",
        session_id: str | None = None,
        raw_rgb: bool = False,
        convert_raw_to_mp4: bool = False,
        keep_raw: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir) / (session_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.camera_keys = tuple(camera_keys)
        self.fps = float(fps)
        self.filename_prefix = filename_prefix
        self.raw_rgb = raw_rgb
        self.convert_raw_to_mp4 = convert_raw_to_mp4
        self.keep_raw = keep_raw
        self._writers: dict[str, subprocess.Popen | BinaryIO] = {}
        self._paths: dict[str, Path] = {}
        self._metadata: dict[str, dict] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def paths(self) -> dict[str, Path]:
        return dict(self._paths)

    def write(self, observation: dict) -> None:
        for key in self.camera_keys:
            image = observation.get(key)
            if image is None:
                continue
            image_np = np.asarray(image)
            if image_np.ndim != 3 or image_np.shape[2] != 3:
                continue
            writer = self._writers.get(key)
            if writer is None:
                writer = self._open_writer(key, image_np)
                self._writers[key] = writer
            payload = np.ascontiguousarray(image_np).tobytes()
            if self.raw_rgb:
                writer.write(payload)
                self._metadata[key]["frames"] += 1
            else:
                writer.stdin.write(payload)

    def close(self) -> None:
        for key, writer in self._writers.items():
            if self.raw_rgb:
                writer.close()
            else:
                if writer.stdin:
                    writer.stdin.close()
                return_code = writer.wait(timeout=10)
                if return_code != 0:
                    logger.warning("ffmpeg recorder for %s exited with code %s", key, return_code)
        self._writers.clear()
        if self.raw_rgb and self._metadata:
            metadata_path = self.output_dir / "metadata.json"
            metadata_path.write_text(json.dumps(self._metadata, indent=2), encoding="utf-8")
            if self.convert_raw_to_mp4:
                self._convert_raw_recordings_to_mp4()
        if self._paths:
            logger.info("Camera recordings saved: %s", self._paths)

    def _convert_raw_recordings_to_mp4(self) -> None:
        for key, meta in self._metadata.items():
            raw_path = Path(meta["path"])
            mp4_path = raw_path.with_suffix(".mp4")
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{meta['width']}x{meta['height']}",
                "-r",
                f"{float(meta['fps']):g}",
                "-i",
                str(raw_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4_path),
            ]
            logger.info("Converting raw camera recording %s to %s", raw_path, mp4_path)
            subprocess.run(cmd, check=True)
            self._paths[key] = mp4_path
            if not self.keep_raw:
                raw_path.unlink(missing_ok=True)

    def _open_writer(self, key: str, image_rgb: np.ndarray) -> subprocess.Popen | BinaryIO:
        height, width = image_rgb.shape[:2]
        safe_key = key.replace("/", "_").replace(".", "_")
        if self.raw_rgb:
            path = self.output_dir / f"{self.filename_prefix}{safe_key}.rgb"
            writer = path.open("wb")
            self._paths[key] = path
            self._metadata[key] = {
                "path": str(path),
                "width": width,
                "height": height,
                "fps": self.fps,
                "pix_fmt": "rgb24",
                "frames": 0,
                "convert_to_mp4": (
                    f"ffmpeg -y -f rawvideo -pix_fmt rgb24 -s {width}x{height} -r {self.fps:g} "
                    f"-i {path} -an -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p "
                    f"-movflags +faststart {path.with_suffix('.mp4')}"
                ),
            }
            logger.info("Recording raw RGB camera %s to %s", key, path)
            return writer

        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(
                f"Camera recorder requires even frame size for yuv420p, got {width}x{height} for {key}."
            )
        path = self.output_dir / f"{self.filename_prefix}{safe_key}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if writer.stdin is None:
            writer.kill()
            raise RuntimeError(f"Failed to open ffmpeg stdin for camera recorder {key}: {path}")
        self._paths[key] = path
        logger.info("Recording camera %s to %s", key, path)
        return writer
