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

import numbers

import numpy as np


def _require_scipy():
    try:
        import scipy.interpolate as si
        import scipy.spatial.transform as st
    except ImportError as e:
        raise ImportError(
            "RealMan trajectory interpolation requires scipy. Install scipy in the active environment."
        ) from e
    return si, st


def realman_euler_pose_to_rotvec_pose(pose: np.ndarray) -> np.ndarray:
    _, st = _require_scipy()
    pose = np.asarray(pose, dtype=np.float64)
    out = np.array(pose, dtype=np.float64, copy=True)
    out[3:] = st.Rotation.from_euler("xyz", pose[3:]).as_rotvec()
    return out


def rotvec_pose_to_realman_euler_pose(pose: np.ndarray) -> np.ndarray:
    _, st = _require_scipy()
    pose = np.asarray(pose, dtype=np.float64)
    out = np.array(pose, dtype=np.float64, copy=True)
    out[3:] = st.Rotation.from_rotvec(pose[3:]).as_euler("xyz")
    return out


def rotation_distance(a, b) -> float:
    return (b * a.inv()).magnitude()


def pose_distance(start_pose: np.ndarray, end_pose: np.ndarray) -> tuple[float, float]:
    _, st = _require_scipy()
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


class PoseTrajectoryInterpolator:
    """UMI pose trajectory interpolator used by the original Pika RealMan stack."""

    def __init__(self, times: np.ndarray, poses: np.ndarray):
        si, st = _require_scipy()
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:, :3]
            rot = st.Rotation.from_rotvec(poses[:, 3:])

            self.pos_interp = si.interp1d(times, pos, axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        return self.pos_interp.x

    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        n = len(self.times)
        poses = np.zeros((n, 6))
        poses[:, :3] = self.pos_interp.y
        poses[:, 3:] = self.rot_interp(self.times).as_rotvec()
        return poses

    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)

    def drive_to_waypoint(
        self,
        pose: np.ndarray,
        time: float,
        curr_time: float,
        max_pos_speed: float = np.inf,
        max_rot_speed: float = np.inf,
    ) -> "PoseTrajectoryInterpolator":
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        time = max(time, curr_time)

        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator(times, poses)

    def schedule_waypoint(
        self,
        pose: np.ndarray,
        time: float,
        max_pos_speed: float = np.inf,
        max_rot_speed: float = np.inf,
        curr_time: float | None = None,
        last_waypoint_time: float | None = None,
    ) -> "PoseTrajectoryInterpolator":
        assert max_pos_speed > 0
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                return self
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)

        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)
        return PoseTrajectoryInterpolator(times, poses)

    def __call__(self, t: numbers.Number | np.ndarray) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            pose[:, :3] = self.pos_interp(t)
            pose[:, 3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose
