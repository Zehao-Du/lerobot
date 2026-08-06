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

import math

import numpy as np

from lerobot.utils.rotation import Rotation


# =================== RealMan/Pika Frame Calibration (Edit Here) =================== #
# Pika gripper origin expressed in the RealMan TCP frame.
# Translation is in millimeters. Rotation values are [x, y, z] in degrees,
# applied as fixed-axis (extrinsic) rotations in Y-X-Z order.
PIKA_GRIPPER_TRANSLATION_IN_REALMAN_TCP_MM = np.array([0.0, 0.0, 5.0], dtype=np.float64)
PIKA_GRIPPER_ROTATION_IN_REALMAN_TCP_DEG = np.array([0.0, -90.0, 180.0], dtype=np.float64)
# =================== End RealMan/Pika Frame Calibration ============================ #


def _make_transform(rot: np.ndarray, trans: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    if trans is not None:
        out[:3, 3] = np.asarray(trans, dtype=np.float64)
    return out


def _euler_xyz_to_mat(euler: np.ndarray) -> np.ndarray:
    rx, ry, rz = euler
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def _extrinsic_yxz_to_mat(euler_xyz: np.ndarray) -> np.ndarray:
    """Convert [rx, ry, rz] using fixed-axis rotations in Y-X-Z order."""
    rx, ry, rz = euler_xyz
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_x @ rot_y


def _mat_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
    sy = np.clip(-rot[2, 0], -1.0, 1.0)
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-9:
        rx = math.atan2(rot[2, 1], rot[2, 2])
        rz = math.atan2(rot[1, 0], rot[0, 0])
    else:
        rx = 0.0
        rz = math.atan2(-rot[0, 1], rot[1, 1])
    return np.array([rx, ry, rz], dtype=np.float64)


def realman_euler_pose_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected RealMan TCP pose shape (6,), got {pose.shape}.")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _euler_xyz_to_mat(pose[3:])
    out[:3, 3] = pose[:3]
    return out


def mat_to_realman_euler_pose(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected transform matrix shape (4, 4), got {mat.shape}.")
    pose = np.zeros((6,), dtype=np.float64)
    pose[:3] = mat[:3, 3]
    pose[3:] = _mat_to_euler_xyz(mat[:3, :3])
    return pose


def rotvec_pose_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"Expected rotvec pose shape (6,), got {pose.shape}.")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    out[:3, 3] = pose[:3]
    return out


def mat_to_rotvec_pose(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected transform matrix shape (4, 4), got {mat.shape}.")
    pose = np.zeros((6,), dtype=np.float64)
    pose[:3] = mat[:3, 3]
    pose[3:] = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return pose


T_REALMAN_TCP_PIKA_GRIPPER = _make_transform(
    _extrinsic_yxz_to_mat(np.deg2rad(PIKA_GRIPPER_ROTATION_IN_REALMAN_TCP_DEG)),
    PIKA_GRIPPER_TRANSLATION_IN_REALMAN_TCP_MM / 1000.0,
)
if not np.allclose(T_REALMAN_TCP_PIKA_GRIPPER[:3, 0], [0.0, 0.0, 1.0], atol=1e-9):
    raise RuntimeError("Pika gripper x-axis must align with RealMan TCP +z-axis.")
T_PIKA_GRIPPER_REALMAN_TCP = np.linalg.inv(T_REALMAN_TCP_PIKA_GRIPPER)


def realman_tcp_pose_to_pika_gripper_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 1:
        return mat_to_rotvec_pose(realman_euler_pose_to_mat(pose) @ T_REALMAN_TCP_PIKA_GRIPPER)
    if pose.ndim != 2 or pose.shape[-1] != 6:
        raise ValueError(f"Expected RealMan TCP pose shape (..., 6), got {pose.shape}.")
    return np.stack([realman_tcp_pose_to_pika_gripper_pose(x) for x in pose], axis=0)


def pika_gripper_pose_to_realman_tcp_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 1:
        return mat_to_realman_euler_pose(rotvec_pose_to_mat(pose) @ T_PIKA_GRIPPER_REALMAN_TCP)
    if pose.ndim != 2 or pose.shape[-1] != 6:
        raise ValueError(f"Expected Pika gripper pose shape (..., 6), got {pose.shape}.")
    return np.stack([pika_gripper_pose_to_realman_tcp_pose(x) for x in pose], axis=0)


def realman_tcp_relative_pose_to_pika_relative_pose(pose: np.ndarray) -> np.ndarray:
    """Convert a RealMan TCP local relative pose into the Pika TCP local frame.

    Relative poses are frame deltas, not absolute world poses. If ``T_rp`` is
    the fixed RealMan-TCP-to-Pika-TCP transform, the same physical delta in the
    Pika TCP frame is ``inv(T_rp) @ delta_realman @ T_rp``.
    """

    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 1:
        delta_realman = realman_euler_pose_to_mat(pose)
        return mat_to_rotvec_pose(T_PIKA_GRIPPER_REALMAN_TCP @ delta_realman @ T_REALMAN_TCP_PIKA_GRIPPER)
    if pose.ndim != 2 or pose.shape[-1] != 6:
        raise ValueError(f"Expected RealMan TCP relative pose shape (..., 6), got {pose.shape}.")
    return np.stack([realman_tcp_relative_pose_to_pika_relative_pose(x) for x in pose], axis=0)


def pika_relative_pose_to_realman_tcp_relative_pose(pose: np.ndarray) -> np.ndarray:
    """Convert a Pika TCP local relative pose into the RealMan TCP local frame."""

    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim == 1:
        delta_pika = rotvec_pose_to_mat(pose)
        return mat_to_realman_euler_pose(T_REALMAN_TCP_PIKA_GRIPPER @ delta_pika @ T_PIKA_GRIPPER_REALMAN_TCP)
    if pose.ndim != 2 or pose.shape[-1] != 6:
        raise ValueError(f"Expected Pika TCP relative pose shape (..., 6), got {pose.shape}.")
    return np.stack([pika_relative_pose_to_realman_tcp_relative_pose(x) for x in pose], axis=0)


def realman_tcp_relative_pose_between(reference_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """Return the RealMan TCP local delta taking ``reference_pose`` to ``target_pose``."""

    reference_pose = np.asarray(reference_pose, dtype=np.float64)
    target_pose = np.asarray(target_pose, dtype=np.float64)
    delta = np.linalg.inv(realman_euler_pose_to_mat(reference_pose)) @ realman_euler_pose_to_mat(target_pose)
    return mat_to_realman_euler_pose(delta)


def apply_realman_tcp_relative_pose(current_pose: np.ndarray, relative_pose: np.ndarray) -> np.ndarray:
    """Apply a RealMan TCP local relative pose to an absolute RealMan TCP pose."""

    current_pose = np.asarray(current_pose, dtype=np.float64)
    relative_pose = np.asarray(relative_pose, dtype=np.float64)
    target = realman_euler_pose_to_mat(current_pose) @ realman_euler_pose_to_mat(relative_pose)
    return mat_to_realman_euler_pose(target)
