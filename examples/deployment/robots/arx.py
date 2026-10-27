"""ARX X5 2025 / AC One ROS bindings and optional workbench authority handoff."""

from __future__ import annotations

import math
from functools import partial
from dataclasses import field, dataclass

from phyai_robot import (
    StateSlot,
    CommandSlot,
    FeatureSpec,
    VectorAdapter,
    CompositeRobot,
)
from phyai_robot.images import resize_with_pad
from phyai_robot.remote import PolicySpec, Embodiment
from phyai_robot.backends.ros2 import (
    RosCommand,
    Ros2Backend,
    RosObservation,
    image_backends,
)
from phyai_robot.backends.ros2.codecs import decode_rgb_image
from phyai_robot.backends.ros2.backend import SERVICE_POLL_MARGIN_S
from phyai_robot._compat import ExceptionGroup

from .arx_codecs import JOINT_NAMES, decode_arm, encode_arm, decode_compressed_rgb

SIDES = ("left", "right")


@dataclass
class ArxConfig:
    """Native topics and controller interface; model processing stays on the server."""

    domain_id: int | None = None
    node_name: str = "phyai_arx"
    io_timeout_s: float = 0.1
    camera_transport: str = "compressed"
    camera_topics: dict[str, str] = field(
        default_factory=lambda: {
            "base_camera": "/camera/camera_h/color/image_rect_raw/compressed",
            "left_wrist_camera": "/camera/camera_l/color/image_rect_raw/compressed",
            "right_wrist_camera": "/camera/camera_r/color/image_rect_raw/compressed",
        }
    )
    model_cameras: dict[str, str] = field(
        default_factory=lambda: {
            "base": "base_camera",
            "left_wrist": "left_wrist_camera",
            "right_wrist": "right_wrist_camera",
        }
    )
    state_topics: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/arm_slave_l_status",
            "right": "/arm_slave_r_status",
        }
    )
    command_topics: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/arm_master_l_status",
            "right": "/arm_master_r_status",
        }
    )
    controller_interface: str = "external"
    controller_nodes: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/arm_slave_l",
            "right": "/arm_slave_r",
        }
    )
    service_timeout_s: float = 3.0
    qos: dict[str, str] = field(
        default_factory=lambda: {
            "camera": "sensor_data",
            "state": "sensor_data",
            "command": "reliable",
        }
    )

    def validate(self) -> None:
        if self.domain_id is not None and not 0 <= self.domain_id <= 232:
            raise ValueError("robot.domain_id must be in [0, 232] or null")
        if (
            not self.node_name
            or not self.node_name.replace("_", "").isalnum()
            or self.node_name[0].isdigit()
        ):
            raise ValueError("robot.node_name must be a valid ROS node name")
        for name in ("io_timeout_s", "service_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"robot.{name} must be positive and finite")
        if self.camera_transport not in {"compressed", "raw"}:
            raise ValueError("robot.camera_transport must be compressed or raw")
        if self.controller_interface not in {"external", "workbench"}:
            raise ValueError("robot.controller_interface must be external or workbench")
        for name in ("state_topics", "command_topics", "controller_nodes"):
            if set(getattr(self, name)) != set(SIDES):
                raise ValueError(f"robot.{name} must map left and right")
        for name in (
            "camera_topics",
            "state_topics",
            "command_topics",
            "controller_nodes",
        ):
            values = list(getattr(self, name).values())
            if (
                not values
                or len(set(values)) != len(values)
                or any(not v.startswith("/") or v == "/" for v in values)
            ):
                raise ValueError(f"robot.{name} must use distinct absolute names")
        if set(self.command_topics.values()) & (
            set(self.state_topics.values()) | set(self.camera_topics.values())
        ):
            raise ValueError("ARX command topics must differ from observation topics")
        if not self.model_cameras or not set(self.model_cameras.values()) <= set(
            self.camera_topics
        ):
            raise ValueError("robot.model_cameras must name camera_topics fields")
        if set(self.camera_topics) & set(ACTION_SCHEMA):
            raise ValueError(
                "ARX camera fields must differ from joint and gripper fields"
            )
        if {"camera", "state", "command"} - set(self.qos):
            raise ValueError("robot.qos needs camera, state and command")


ACTION_SCHEMA = {
    **{
        f"joint_position_{side}": FeatureSpec(
            (6,), "float64", unit="rad", names=JOINT_NAMES
        )
        for side in SIDES
    },
    **{
        f"gripper_{side}": FeatureSpec(
            (1,), "float64", unit="rad", names=("motor_position",)
        )
        for side in SIDES
    },
}


def observation_schema(config: ArxConfig, image_size: int) -> dict[str, FeatureSpec]:
    return {
        **{
            key: FeatureSpec((image_size, image_size, 3), "uint8")
            for key in config.camera_topics
        },
        **ACTION_SCHEMA,
    }


def make_adapter(
    config: ArxConfig, observations: dict[str, FeatureSpec]
) -> VectorAdapter:
    fields = [
        key for side in SIDES for key in (f"joint_position_{side}", f"gripper_{side}")
    ]
    return VectorAdapter(
        observation_schema=observations,
        action_schema=ACTION_SCHEMA,
        cameras=config.model_cameras,
        state=[StateSlot(key) for key in fields],
        commands=[CommandSlot(key) for key in fields],
    )


class WorkbenchController:
    """Select policy input on resident controllers without enabling or homing motors."""

    def __init__(self, config: ArxConfig) -> None:
        self.config = config
        self.selected = False

    def call(self, backend: Ros2Backend, side: str, action: str) -> None:
        from std_srvs.srv import Trigger

        backend.call_service(
            Trigger,
            self.config.controller_nodes[side].rstrip("/") + "/" + action,
            Trigger.Request(),
            timeout_s=self.config.service_timeout_s,
        )

    def activate(self, backend: Ros2Backend) -> None:
        for side in SIDES:
            self.call(backend, side, "health_check")
        # Remember partial activation too, so cleanup revokes both gates on failure.
        self.selected = True
        for side in SIDES:
            self.call(backend, side, "select_inference")
        for side in SIDES:
            self.call(backend, side, "remember_fold")

    def stop(self, backend: Ros2Backend) -> None:
        if not self.selected:
            return
        errors = []
        for side in SIDES:
            try:
                # Reselecting inference closes its policy gate and retains the target.
                self.call(backend, side, "select_inference")
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup(
                "ARX controller gates did not both acknowledge stop", errors
            )
        self.selected = False


def build(config: ArxConfig, spec: PolicySpec) -> Embodiment:
    """Build lazy ROS backends; no handles or controller services open here."""
    config.validate()
    from sensor_msgs.msg import Image, CompressedImage
    from arx5_arm_msg.msg import RobotStatus

    image_size = int(spec.metadata["image_size"])
    ros = dict(domain_id=config.domain_id, io_timeout_s=config.io_timeout_s)
    states, commands = {}, {}
    for side in SIDES:
        fields = (f"joint_position_{side}", f"gripper_{side}")
        states[fields] = RosObservation(
            config.state_topics[side],
            RobotStatus,
            partial(decode_arm, side=side),
            config.qos["state"],
            recover_on_next_message=True,
        )
        commands[fields] = RosCommand(
            config.command_topics[side],
            RobotStatus,
            partial(encode_arm, side=side, message_type=RobotStatus),
            config.qos["command"],
            stamp=True,
        )
    controller = (
        WorkbenchController(config)
        if config.controller_interface == "workbench"
        else None
    )
    joints = Ros2Backend(
        observations=states,
        actions=commands,
        on_stop=controller.stop if controller else lambda backend: None,
        node_name=config.node_name,
        stop_timeout_s=2 * (config.service_timeout_s + SERVICE_POLL_MARGIN_S) + 1,
        **ros,
    )
    compressed = config.camera_transport == "compressed"
    decode = decode_compressed_rgb if compressed else decode_rgb_image
    cameras = image_backends(
        config.camera_topics,
        message_type=CompressedImage if compressed else Image,
        decode=lambda message: resize_with_pad(decode(message), image_size),
        node_name=config.node_name,
        qos=config.qos["camera"],
        **ros,
    )
    schema = observation_schema(config, image_size)
    robot = CompositeRobot(
        observation_schema=schema,
        action_schema=ACTION_SCHEMA,
        backends=[*cameras, joints],
    )
    return Embodiment(
        robot,
        make_adapter(config, schema),
        partial(controller.activate, joints) if controller else None,
    )
