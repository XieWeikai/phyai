"""ARX X5 2025 / AC One RobotStatus and RGB camera codecs, without ROS imports."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import NDArray

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))


def decode_arm(message: Any, side: str) -> dict[str, NDArray[np.float64]]:
    """Keep the driver's fixed six-joint order and raw gripper motor radians."""
    values = np.asarray(message.joint_pos, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("ARX RobotStatus.joint_pos must contain seven finite values")
    return {
        f"joint_position_{side}": values[:6].copy(),
        f"gripper_{side}": values[6:].copy(),
    }


def encode_arm(values: Any, side: str, message_type: type[Any]) -> Any:
    """Build a remote_slave target without clipping, scaling or delta decoding."""
    joints = np.asarray(values[f"joint_position_{side}"], dtype=np.float64)
    gripper = np.asarray(values[f"gripper_{side}"], dtype=np.float64)
    if joints.shape != (6,) or gripper.shape != (1,):
        raise ValueError("ARX commands require six joints and one raw gripper value")
    target = np.concatenate([joints, gripper])
    if not np.isfinite(target).all():
        raise ValueError("ARX commands must be finite")
    message = message_type()
    message.joint_pos = target.tolist()
    message.joint_vel = [0.0] * 7
    message.joint_cur = [0.0] * 7
    return message


def decode_compressed_rgb(message: Any) -> NDArray[np.uint8]:
    """Decode the JPEG/PNG color stream; H.264 and depth streams are not supported."""
    from PIL import Image

    with Image.open(BytesIO(bytes(message.data))) as image:
        if image.format not in {"JPEG", "PNG"} or image.mode not in {"RGB", "RGBA"}:
            raise ValueError("ARX compressed camera requires a JPEG/PNG color image")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
