"""Tianji CompositeRobot factory, ROS codecs, and controller lifecycle.

This module owns the embodiment contract, not model inference or a control loop.
Each action is a complete, one-shot publication. RobotDeployment owns repetition
and interpolation; CompositeRobot owns schema validation and backend stop/close.
A separate camera backend keeps image conversion off the command worker.
"""

from __future__ import annotations

import math
import time
from typing import Any
from functools import lru_cache
from threading import Thread
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from phyai_robot import (
    Action,
    FeatureSpec,
    Observation,
    CompositeRobot,
    ObservationNotReady,
)
from numpy.typing import NDArray
from phyai_robot.backends.ros2 import RosCommand, Ros2Backend, RosObservation

# These dimensions come from the live Tianji camera topics used by this deployment.
# The phyai-robot schema is intentionally fixed, so a change in camera mode or
# resolution should fail loudly instead of silently producing a different input.
CAMERA_HEIGHT = 1984
CAMERA_WIDTH = 2560
ARM_DOF = 7
GRIPPER_FEEDBACK_SIZE = 5

# Tianji's command mux and control-mode service values.  INPUT_USER selects the
# /tj/control/user/* joint command path; it is independent from the joint
# position/impedance mode selected by /tj/control/set_mode.
INPUT_IDLE = 0
INPUT_USER = 3
MODE_JOINT_POSITION = 1
MODE_CARTESIAN_IMPEDANCE = 2
MODE_JOINT_IMPEDANCE = 3

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, ARM_DOF + 1))
EEF_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")
WRENCH_NAMES = ("fx", "fy", "fz", "tx", "ty", "tz")
GRIPPER_FEEDBACK_NAMES = (
    "position",
    "velocity",
    "torque",
    "mos_temperature",
    "motor_temperature",
)

# Observation names are application-level names, not ROS topic names.  The
# RosObservation table below owns the topic-to-field mapping and performs all
# message decoding into these exact shapes and dtypes.
OBSERVATION_SCHEMA: Mapping[str, FeatureSpec] = {
    "head_camera": FeatureSpec((CAMERA_HEIGHT, CAMERA_WIDTH, 3), "uint8"),
    "left_wrist_camera": FeatureSpec((CAMERA_HEIGHT, CAMERA_WIDTH, 3), "uint8"),
    "right_wrist_camera": FeatureSpec((CAMERA_HEIGHT, CAMERA_WIDTH, 3), "uint8"),
    "joint_position_left": FeatureSpec(
        (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
    ),
    "joint_position_right": FeatureSpec(
        (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
    ),
    "eef_left": FeatureSpec((7,), "float64", names=EEF_NAMES),
    "eef_right": FeatureSpec((7,), "float64", names=EEF_NAMES),
    "wrench_left": FeatureSpec((6,), "float64", names=WRENCH_NAMES),
    "wrench_right": FeatureSpec((6,), "float64", names=WRENCH_NAMES),
    "gripper_feedback_left": FeatureSpec(
        (GRIPPER_FEEDBACK_SIZE,),
        "float32",
        names=GRIPPER_FEEDBACK_NAMES,
    ),
    "gripper_feedback_right": FeatureSpec(
        (GRIPPER_FEEDBACK_SIZE,),
        "float32",
        names=GRIPPER_FEEDBACK_NAMES,
    ),
}

# Every action field is mandatory for every control tick.  In particular, a
# caller cannot omit the gripper values: it must decide what complete command
# to send, which is the contract expected by CompositeRobot.
ACTION_SCHEMA: Mapping[str, FeatureSpec] = {
    "joint_position_left": FeatureSpec(
        (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
    ),
    "joint_position_right": FeatureSpec(
        (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
    ),
    "gripper_left": FeatureSpec(
        (1,), "float32", unit="normalized", names=("target_position",)
    ),
    "gripper_right": FeatureSpec(
        (1,), "float32", unit="normalized", names=("target_position",)
    ),
}


@dataclass
class TianjiRobotConfig:
    """Tianji ROS connection and controller-startup settings only.

    Mode/input services are device protocol details, so their configuration and
    validation stay next to the factory rather than in the shared YAML loader.
    Model normalization and control-loop timing are configured elsewhere.
    """

    mode: int = MODE_JOINT_IMPEDANCE
    velocity_ratio: int | None = None
    node_name: str = "phyai_tianji_robot"
    service_timeout_s: float = 3.0
    io_timeout_s: float = 0.1

    def validate(self) -> None:
        """Reject incompatible joint-control modes before creating ROS resources."""
        if self.mode not in (MODE_JOINT_POSITION, MODE_JOINT_IMPEDANCE):
            raise ValueError(
                "robot.mode must be 1 (joint position) or 3 (joint impedance)"
            )
        if self.velocity_ratio is not None and not 0 <= self.velocity_ratio <= 100:
            raise ValueError("robot.velocity_ratio must be in [0, 100] or null")
        if not self.node_name.strip():
            raise ValueError("robot.node_name must be non-empty")
        for name in ("service_timeout_s", "io_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"robot.{name} must be finite and positive")


class TianjiControl:
    """Own the service node used to prepare and safely stop Tianji control.

    This is intentionally separate from the Robot backend.  It does not
    publish actions or run a periodic loop; it only calls the mode/input
    services required by the device protocol.
    """

    def __init__(
        self,
        *,
        node_name: str = "phyai_tianji_control",
        service_timeout_s: float = 3.0,
        read_only: bool = True,
    ) -> None:
        if service_timeout_s <= 0:
            raise ValueError("service_timeout_s must be positive")
        self._read_only = read_only
        self._node_name = node_name
        self._service_timeout_s = service_timeout_s
        self._context: Any = None
        self._node: Any = None
        self._executor: Any = None
        self._thread: Thread | None = None
        self._spin_error: BaseException | None = None
        self._closed = False

    def connect(self) -> None:
        """Start the private ROS executor and create Tianji service clients."""
        if self._closed:
            raise RuntimeError("TianjiControl is closed")
        if self._thread is not None:
            return

        # Use a private rclpy Context instead of rclpy.init()/shutdown().  This
        # lets TianjiControl coexist with Ros2Backend, which owns another
        # private ROS context and executor.
        from rclpy.node import Node
        from std_srvs.srv import Trigger
        from rclpy.context import Context
        from marvin_msgs.srv import Int
        from rclpy.executors import SingleThreadedExecutor

        context = Context()
        context.init(args=[])
        node: Any = None
        executor: Any = None
        try:
            node = Node(
                self._node_name,
                context=context,
                use_global_arguments=False,
            )
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            self._context = context
            self._node = node
            self._executor = executor
            # The Int client type is shared by mode, input, and velocity services.
            # These are service clients only; no periodic command publishing is
            # started here.
            self._mode_client = node.create_client(Int, "/tj/control/set_mode")
            self._input_client = node.create_client(Int, "/tj/control/set_input")
            self._velocity_client = node.create_client(Int, "/tj/control/set_vel_ratio")
            self._ready_client = node.create_client(Trigger, "/tj/control/set_ready")
            self._thread = Thread(
                target=self._spin,
                name="TianjiControlROS",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            if executor is not None:
                executor.shutdown()
            if node is not None:
                node.destroy_node()
            context.try_shutdown()
            raise

    def _spin(self) -> None:
        try:
            assert self._executor is not None
            self._executor.spin()
        except BaseException as error:  # noqa: BLE001 - propagate through calls
            self._spin_error = error

    def _require_connected(self) -> None:
        if self._closed or self._thread is None:
            raise RuntimeError("TianjiControl is not connected")
        if self._spin_error is not None:
            raise RuntimeError("Tianji ROS executor stopped") from self._spin_error

    def _call(self, client: Any, request: Any) -> Any:
        if self._read_only:
            raise RuntimeError("Read-only TianjiControl cannot call mutating services")
        self._require_connected()
        remaining = self._service_timeout_s
        if not client.wait_for_service(timeout_sec=remaining):
            raise TimeoutError("Tianji service did not become available")
        # The executor thread completes this future.  Polling with a monotonic
        # deadline avoids depending on the exact Future.result(timeout=...)
        # implementation provided by the installed ROS2 version.
        future = client.call_async(request)
        deadline = time.monotonic() + self._service_timeout_s
        while not future.done():
            if time.monotonic() >= deadline:
                future.cancel()
                raise TimeoutError("Tianji service call timed out")
            time.sleep(0.002)
        response = future.result()
        if response is None:
            raise RuntimeError("Tianji service returned no response")
        if hasattr(response, "success") and not response.success:
            message = getattr(response, "message", "")
            raise RuntimeError(f"Tianji service rejected request: {message}")
        return response

    @staticmethod
    def _integer_request(value: int) -> Any:
        from marvin_msgs.srv import Int

        request = Int.Request()
        request.data = int(value)
        return request

    def set_ready(self) -> Any:
        """Enable the Tianji controller through ``/tj/control/set_ready``."""
        from std_srvs.srv import Trigger

        return self._call(self._ready_client, Trigger.Request())

    def set_mode(self, mode: int) -> Any:
        """Set Tianji mode: 1 position, 2 Cartesian impedance, 3 joint impedance."""
        if mode not in {
            MODE_JOINT_POSITION,
            MODE_CARTESIAN_IMPEDANCE,
            MODE_JOINT_IMPEDANCE,
        }:
            raise ValueError(f"unsupported Tianji mode: {mode}")
        return self._call(self._mode_client, self._integer_request(mode))

    def set_input(self, input_mode: int) -> Any:
        """Select the command mux input; 3 is User/Custom and 0 is idle."""
        if input_mode not in {INPUT_IDLE, INPUT_USER}:
            raise ValueError(f"unsupported Tianji input mode: {input_mode}")
        return self._call(self._input_client, self._integer_request(input_mode))

    def set_velocity_ratio(self, ratio: int) -> Any:
        """Set Tianji's controller velocity ratio as an integer percentage."""
        if not 0 <= ratio <= 100:
            raise ValueError("velocity ratio must be in [0, 100]")
        return self._call(self._velocity_client, self._integer_request(ratio))

    def prepare_user_control(
        self,
        *,
        mode: int = MODE_JOINT_IMPEDANCE,
        velocity_ratio: int | None = None,
    ) -> None:
        """Set ready, mode, optional velocity, and User/Custom input in order."""
        # Explicit opt-in: read-only sessions do not mutate controller state,
        # even when CompositeRobot closes its backend.
        self._read_only = False
        self.set_ready()
        self.set_mode(mode)
        if velocity_ratio is not None:
            self.set_velocity_ratio(velocity_ratio)
        self.set_input(INPUT_USER)

    def stop_input(self, _node: Any = None) -> None:
        """Return the command mux to idle; used by ``CompositeRobot.stop``.

        Ros2Backend invokes its ``on_stop`` callback on the backend worker.
        The callback is intentionally small and does not publish a fake zero
        action: Tianji's input mux is the device-level safe-state mechanism.
        """
        if not self._read_only:
            self.set_input(INPUT_IDLE)

    def close(self) -> None:
        """Release only this helper's ROS context; do not call global shutdown."""
        if self._closed:
            return
        self._closed = True
        executor = self._executor
        node = self._node
        context = self._context
        thread = self._thread
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if context is not None:
            context.try_shutdown()
        if thread is not None:
            thread.join(self._service_timeout_s)
            if thread.is_alive():
                raise TimeoutError("Tianji control executor did not stop")


def _decode_image(message: Any) -> NDArray[np.uint8]:
    # sensor_msgs/Image carries NV12 as a Y plane followed by an interleaved
    # half-resolution UV plane.  We keep the ROS message buffer read-only and
    # return a detached RGB array, as required by phyai-robot snapshots.
    height = int(message.height)
    width = int(message.width)
    if height != CAMERA_HEIGHT or width != CAMERA_WIDTH:
        raise ValueError(
            f"unexpected Tianji camera shape: {(height, width)}, "
            f"expected {(CAMERA_HEIGHT, CAMERA_WIDTH)}"
        )
    raw = np.frombuffer(message.data, dtype=np.uint8)
    expected_size = height * width * 3 // 2
    if raw.size != expected_size:
        raise ValueError(
            f"unexpected NV12 data size: {raw.size}, expected {expected_size}"
        )
    frame = raw.reshape((height * 3 // 2, width))
    # Prefer OpenCV for the high-rate camera path.  The NumPy implementation
    # below is a dependency-free fallback for environments with an incompatible
    # OpenCV/NumPy binary combination.
    cv2 = _load_cv2()
    if cv2 is not None:
        try:
            return np.asarray(cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12)).copy()
        except Exception:  # noqa: BLE001 - fall back to the dependency-free decoder
            return _decode_nv12_numpy(frame, height, width)
    return _decode_nv12_numpy(frame, height, width)


@lru_cache(maxsize=1)
def _load_cv2() -> Any | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def _decode_nv12_numpy(
    frame: NDArray[np.uint8], height: int, width: int
) -> NDArray[np.uint8]:
    y_plane = frame[:height].astype(np.float32)
    uv_plane = frame[height:].reshape((height // 2, width // 2, 2))
    u = np.repeat(np.repeat(uv_plane[:, :, 0], 2, axis=0), 2, axis=1)
    v = np.repeat(np.repeat(uv_plane[:, :, 1], 2, axis=0), 2, axis=1)
    red = y_plane + 1.402 * (v - 128.0)
    green = y_plane - 0.344136 * (u - 128.0) - 0.714136 * (v - 128.0)
    blue = y_plane + 1.772 * (u - 128.0)
    rgb = np.stack((red, green, blue), axis=-1)
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _decode_joint(message: Any, side: int) -> NDArray[np.float64]:
    # Tianji publishes both arms in one Jointfeedback message.  The velocity
    # and effort arrays are deliberately ignored; only arm_positions are part
    # of this robot's observation contract.
    positions = np.asarray(message.arm_positions, dtype=np.float64)
    expected_size = ARM_DOF * 2
    if positions.shape != (expected_size,):
        raise ValueError(
            f"unexpected joint feedback length: {positions.shape}, "
            f"expected {(expected_size,)}"
        )
    start = side * ARM_DOF
    return positions[start : start + ARM_DOF].copy()


def _decode_eef(message: Any) -> NDArray[np.float64]:
    # Convert geometry_msgs/PoseStamped to the stable seven-element layout
    # documented by OBSERVATION_SCHEMA.
    pose = message.pose
    return np.asarray(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )


def _decode_wrench(message: Any) -> NDArray[np.float64]:
    # WrenchStamped is flattened as force xyz followed by torque xyz.
    wrench = message.wrench
    return np.asarray(
        [
            wrench.force.x,
            wrench.force.y,
            wrench.force.z,
            wrench.torque.x,
            wrench.torque.y,
            wrench.torque.z,
        ],
        dtype=np.float64,
    )


def _decode_gripper_feedback(message: Any) -> NDArray[np.float32]:
    # Float32MultiArray order is position, velocity, torque, MOS temperature,
    # and motor temperature, as documented by the Tianji interface.
    values = np.asarray(message.data, dtype=np.float32)
    if values.shape != (GRIPPER_FEEDBACK_SIZE,):
        raise ValueError(
            f"unexpected gripper feedback length: {values.shape}, "
            f"expected {(GRIPPER_FEEDBACK_SIZE,)}"
        )
    return values.copy()


def _stamp_header(header: Any) -> None:
    stamp_ns = time.time_ns()
    header.stamp.sec = stamp_ns // 1_000_000_000
    header.stamp.nanosec = stamp_ns % 1_000_000_000


def _encode_joint(values: NDArray[Any]) -> Any:
    from marvin_msgs.msg import JointcmdArm

    # RosCommand.encode is called once per complete CompositeRobot action.
    # The controller accepts a JointcmdArm message; it does not mean that the
    # command has completed physically when publish() returns.
    message = JointcmdArm()
    _stamp_header(message.header)
    message.positions = np.asarray(values, dtype=np.float64).tolist()
    return message


def _encode_gripper(values: NDArray[Any]) -> Any:
    from std_msgs.msg import Float32

    # The schema uses shape (1,) so the action remains an array like every
    # other phyai-robot field; the ROS message itself contains one scalar.
    message = Float32()
    message.data = float(values[0])
    return message


def _make_reliable_qos() -> Any:
    # EEF and gripper feedback publishers were observed with reliable QoS; a
    # matching profile prevents DDS endpoint incompatibility.  Cameras, joint
    # feedback, and wrench data use qos_profile_sensor_data below.
    from rclpy.qos import QoSProfile, HistoryPolicy, DurabilityPolicy, ReliabilityPolicy

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _validate_action(action: Action) -> None:
    # CompositeRobot has already checked completeness, shapes, dtypes, and
    # finite values.  This guard adds the device-specific normalized gripper
    # range before Ros2Backend publishes anything.
    for key in ("gripper_left", "gripper_right"):
        value = float(action[key][0])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} target must be in [0, 1], got {value}")


def resize_rgb(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Resize RGB with PIL bilinear interpolation and centered black padding.

    The decoder caches small images so observation snapshots do not copy native
    camera frames. This only changes image geometry; model normalization and
    tokenization remain in the adapter. The same operation is used for any
    non-resized image passed directly to the adapter.
    """
    from PIL import Image

    height, width = image.shape[:2]
    ratio = max(width / size, height / size)
    resized = Image.fromarray(image).resize(
        (int(width / ratio), int(height / ratio)), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (size, size))
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.array(padded)


def make_tianji_robot(
    control: TianjiControl,
    *,
    node_name: str = "phyai_tianji_robot",
    io_timeout_s: float = 0.1,
    image_size: int = 224,
) -> CompositeRobot:
    """Construct the standard robot without connecting or selecting input mode.

    The left eye is head_camera; the right eye is unused. The three RGB cameras,
    both joint-position vectors, EEF poses, wrenches, and five-value gripper
    feedback remain in Observation. Only images are resized to image_size.
    The caller must keep control connected until after robot.stop/close so the
    command backend can request input=0 during shutdown.
    """
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float32, Float32MultiArray
    from marvin_msgs.msg import JointcmdArm, Jointfeedback
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseStamped, WrenchStamped

    reliable_qos = _make_reliable_qos()
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size must be a positive integer")

    def image_decoder(message: Any) -> NDArray[np.uint8]:
        return resize_rgb(_decode_image(message), image_size)

    # Each mapping key is a phyai-robot field.  Ros2Backend owns one ROS
    # subscription per mapping entry and stores the latest detached Sample.
    # The joint feedback topic appears twice because one ROS message decodes
    # into two independent schema fields.
    observations = {
        "head_camera": RosObservation(
            "/camera/left_eye/image_nv12", Image, image_decoder, qos_profile_sensor_data
        ),
        "left_wrist_camera": RosObservation(
            "/camera/left_wrist/image_nv12",
            Image,
            image_decoder,
            qos_profile_sensor_data,
        ),
        "right_wrist_camera": RosObservation(
            "/camera/right_wrist/image_nv12",
            Image,
            image_decoder,
            qos_profile_sensor_data,
        ),
        "joint_position_left": RosObservation(
            "/tj/info/joint_feedback",
            Jointfeedback,
            lambda message: _decode_joint(message, 0),
            qos_profile_sensor_data,
        ),
        "joint_position_right": RosObservation(
            "/tj/info/joint_feedback",
            Jointfeedback,
            lambda message: _decode_joint(message, 1),
            qos_profile_sensor_data,
        ),
        "eef_left": RosObservation(
            "/tj/info/eef_left", PoseStamped, _decode_eef, reliable_qos
        ),
        "eef_right": RosObservation(
            "/tj/info/eef_right", PoseStamped, _decode_eef, reliable_qos
        ),
        "wrench_left": RosObservation(
            "/tj/info/wrench_left",
            WrenchStamped,
            _decode_wrench,
            qos_profile_sensor_data,
        ),
        "wrench_right": RosObservation(
            "/tj/info/wrench_right",
            WrenchStamped,
            _decode_wrench,
            qos_profile_sensor_data,
        ),
        "gripper_feedback_left": RosObservation(
            "/info/gripper_feedback_L",
            Float32MultiArray,
            _decode_gripper_feedback,
            reliable_qos,
        ),
        "gripper_feedback_right": RosObservation(
            "/info/gripper_feedback_R",
            Float32MultiArray,
            _decode_gripper_feedback,
            reliable_qos,
        ),
    }
    # Each action mapping is one publisher and one encoder.  CompositeRobot
    # validates the full action first, then Ros2Backend publishes each of these
    # four messages exactly once in the caller's control tick.
    actions = {
        "joint_position_left": RosCommand(
            "/tj/control/user/joint_cmd_A",
            JointcmdArm,
            _encode_joint,
            qos_profile_sensor_data,
        ),
        "joint_position_right": RosCommand(
            "/tj/control/user/joint_cmd_B",
            JointcmdArm,
            _encode_joint,
            qos_profile_sensor_data,
        ),
        "gripper_left": RosCommand(
            "/control/gripperValueL", Float32, _encode_gripper, reliable_qos
        ),
        "gripper_right": RosCommand(
            "/control/gripperValueR", Float32, _encode_gripper, reliable_qos
        ),
    }

    # The device-specific stop callback is kept outside Ros2Backend's topic
    # publishing path.  CompositeRobot.stop() invokes it before the backend
    # releases its ROS context.
    schema = dict(OBSERVATION_SCHEMA)
    camera_keys = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    backends = [
        Ros2Backend(
            observations={key: observations.pop(key) for key in camera_keys},
            actions={},
            node_name=f"{node_name}_cameras",
            io_timeout_s=io_timeout_s,
        )
    ]
    for key in camera_keys:
        schema[key] = FeatureSpec((image_size, image_size, 3), "uint8")
    backends.append(
        Ros2Backend(
            observations=observations,
            actions=actions,
            on_stop=control.stop_input,
            node_name=node_name,
            io_timeout_s=io_timeout_s,
        )
    )
    return CompositeRobot(
        observation_schema=schema,
        action_schema=ACTION_SCHEMA,
        backends=tuple(backends),
        action_guard=_validate_action,
    )


def wait_for_observation(
    robot: CompositeRobot, *, timeout_s: float = 10.0, max_age_s: float | None = None
) -> Observation:
    """Wait for every field, optionally requiring fresh samples after model warmup."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    deadline = time.monotonic() + timeout_s
    while True:
        # Ros2Backend connects immediately, but each ROS publisher may deliver
        # its first message at a different time.  Retry until the full
        # CompositeRobot Observation can be constructed.
        try:
            observation = robot.get_observation()
            if max_age_s is not None:
                _check_observation_age(observation, max_age_s=max_age_s)
            return observation
        except (ObservationNotReady, TimeoutError) as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Tianji observation was not complete/fresh before the timeout"
                ) from error
            time.sleep(0.01)


def _check_observation_age(observation: Observation, max_age_s: float = 0.5) -> None:
    """Reject missing/stale data before warming up or enabling the controller."""
    now = time.monotonic_ns()
    for key, sample in observation.samples.items():
        if not 0 <= now - sample.received_at_ns <= max_age_s * 1e9:
            raise TimeoutError(f"Stale observation before inference: {key}")
