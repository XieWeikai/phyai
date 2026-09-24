"""Tianji ROS2 control and RLinf pi0.5 deployment example.

Run the commands below from the PhyAI checkout root. Host-specific installation
paths belong in your shell configuration, not in this example. Set these shell
variables before the environment setup:

* ``TIANJI_ROS_SETUP``: the host's ROS setup script, including Tianji interfaces.
* ``TIANJI_ROS_PYTHON``: the host Python with rclpy and camera message packages;
  its Python minor version must match the PhyAI environment.
* ``CUDA_HOME``: the CUDA toolkit root whose ptxas supports the target GPU.
  On Thor, use the host CUDA 13 toolkit rather than Triton's bundled assembler.

Prepare the shared PhyAI/ROS environment::

    source "${TIANJI_ROS_SETUP:?Set TIANJI_ROS_SETUP for this host}"
    source .venv/bin/activate
    ROS_SITE=$("${TIANJI_ROS_PYTHON:?Set TIANJI_ROS_PYTHON for this host}" -c \
        'import site; print(site.getsitepackages()[0])')
    export PYTHONPATH="$ROS_SITE${PYTHONPATH:+:$PYTHONPATH}"
    export TRITON_PTXAS_BLACKWELL_PATH="${CUDA_HOME:?Set CUDA_HOME}/bin/ptxas"

The activated PhyAI environment supplies Python, PhyAI, phyai-robot, and Ninja.
PYTHONPATH adds the host's matching ROS packages; ROS is not reinstalled into
this environment. Use ``uv run --no-sync`` if invoking through uv so it does not
replace the host's prepared GPU packages.

Check codecs without connecting to ROS, then read live observations without
publishing commands or changing control modes::

    python examples/robot/tianji.py --self-test
    python examples/robot/tianji.py --observe-seconds 2

Run the model once on live observations, without moving the robot::

    python examples/robot/tianji.py --policy-dry-run \
        --task "Plug in the Ethernet cable"

The default checkpoint is ``Path.home() / "models" /
"rlinf-pi05-tianji-step3810"``. To use another checkpoint, pass a path relative
to the checkout, for example ``--checkpoint checkpoints/tianji``. The adapter
passes ``--task`` to the policy along with left-eye, left-wrist, and right-wrist
RGB images and 16 arm/gripper position values. The right eye is unused. EEF,
wrench, and complete gripper feedback remain available in Observation.

For a bounded live test, first check the workspace, emergency stop, controller
ownership, and dry-run predictions. Stop other command publishers and release
any existing gripper reservation through its owner. Then run::

    python examples/robot/tianji.py --run-policy --prepare-user-control \
        --mode 3 --task "Plug in the Ethernet cable" \
        --action-fps 25 --control-hz 200 --execution-horizon 20 --steps 160

This generates 50 source actions and retains the first 20. Without missed
control ticks, each chunk sends 160 complete interpolated actions over about
0.8 seconds. ``--steps`` counts actual sends, not model actions or skipped ticks.
Increase it explicitly for a longer trial: 1600 sends cover ten chunks when no
ticks are missed, with observation/inference gaps between them. The model still
predicts all 50 actions regardless of execution horizon.

To keep running until you stop it, replace ``--steps`` with ``--continuous``::

    python examples/robot/tianji.py --run-policy --prepare-user-control \
        --mode 3 --task "Plug in the Ethernet cable" \
        --action-fps 25 --control-hz 200 --execution-horizon 20 \
        --max-control-lateness-ms 10 --continuous

Press Ctrl-C in that terminal to stop. ``--continuous`` and ``--steps`` cannot
be combined. Continuous operation still stops on safety failures; it does not
restart itself or override controller ownership checks.

``--action-fps`` sets source-action spacing; ``--control-hz`` independently sets
the requested Robot.send_action frequency. The concurrent queue stores source
targets, not expanded control ticks. The control loop linearly interpolates
arm/gripper fields and holds the last retained target for its final interval.
Only after the queue drains does the worker read a fresh observation and infer
again. An empty queue sends NO commands; it is not an error and does not call
Robot.stop. This is stop-and-refill execution, not RTC or cross-chunk blending.

``--prepare-user-control --mode 3`` calls ready, selects joint impedance mode 3,
and then selects User/Custom input 3. Mode 1 (joint position) also accepts these
joint targets; mode 2 does not. Model loading and warmup finish before enabling
control, and warmup predictions are discarded. Normal completion, Ctrl-C, and
handled failures use CompositeRobot's stop/close path to request input 0.
Stopping is not a homing command or a substitute for the hardware emergency
stop. Read-only sessions do not change input at cleanup. This example never
takes over an existing gripper reservation or clears a hardware fault.

The Robot remains a standard CompositeRobot with Ros2Backend sessions. Each
send_action publishes one complete action once; repeated publication belongs
to RobotDeployment, not the Robot. All four action fields, including both
grippers, are published each tick without changing the driver's motor-loop
frequency. ``--max-control-lateness-ms`` defaults to 10: a send may start up to
10 ms after its scheduled deadline, independent of the 5 ms control period.
Missed ticks within that tolerance are skipped, not replayed in a burst. The
source trajectory clock is unchanged; a delayed chunk can have fewer sends.
This threshold measures host-side submission lateness, not delivery latency
inside ROS or the robot. A blocking send cannot be interrupted by this check.
Python/ROS is not hard real-time: lateness above the limit, expired predictions,
or sensor timeouts stop deployment. Timing errors report wakeup, lock,
preparation, and previous-send durations. Check timing before a longer live trial.
"""

from __future__ import annotations

import json
import math
import time
import argparse
from typing import Any
from pathlib import Path
from functools import lru_cache
from threading import Lock, Event, Thread
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from phyai_robot import (
    Action,
    ActionChunk,
    FeatureSpec,
    Observation,
    CompositeRobot,
    RobotDeployment,
    DeploymentOptions,
    ObservationNotReady,
)
from numpy.typing import NDArray
from phyai_robot.backends.ros2 import RosCommand, Ros2Backend, RosObservation

# These dimensions come from the live Tianji camera topics discovered on Marvin.
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

# The CLI exposes only a conservative first-motion increment.  Programmatic
# users still submit complete actions through CompositeRobot.send_action().
CLI_MAX_ARM_DELTA_RAD = 0.05

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
        read_only: bool = False,
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
            self._clear_fault_client = node.create_client(
                Trigger, "/tj/control/clear_fault"
            )
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

    def clear_fault(self) -> Any:
        """Request fault clearing before preparing the controller."""
        from std_srvs.srv import Trigger

        return self._call(self._clear_fault_client, Trigger.Request())

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


def _resize_policy_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Match run_rlinf's uint8 PIL bilinear resize and centered black padding.

    The camera worker caches small RGB images to reduce snapshot copy traffic
    between chunks. The adapter still owns normalization/tokenization; its
    reference preprocessing leaves an already-224x224 image unchanged.
    """
    from PIL import Image

    height, width = image.shape[:2]
    size = 224
    ratio = max(width / size, height / size)
    resized = Image.fromarray(image).resize(
        (int(width / ratio), int(height / ratio)), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (size, size))
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.array(padded)


def _decode_policy_image(message: Any) -> NDArray[np.uint8]:
    return _resize_policy_rgb(_decode_image(message))


def make_tianji_robot(
    control: TianjiControl,
    *,
    node_name: str = "phyai_tianji_robot",
    io_timeout_s: float = 0.1,
    policy_images: bool = False,
) -> CompositeRobot:
    """Create the complete Tianji ``CompositeRobot``.

    ``left_eye`` is exposed as ``head_camera``.  The right-eye camera is
    intentionally not subscribed.  Both wrist cameras, joint positions, EEF
    poses, wrenches, and the two five-element gripper feedback messages are
    returned through the standard ``Observation`` type. With ``policy_images``,
    camera samples are RGB uint8 224x224 after the reference resize/pad transform;
    otherwise they retain the native resolution. Sensor names and non-image
    schemas stay unchanged. Heavy image callbacks use a separate ROS worker in
    policy mode so they do not directly delay the command backend's write queue.
    """
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Float32, Float32MultiArray
    from marvin_msgs.msg import JointcmdArm, Jointfeedback
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import PoseStamped, WrenchStamped

    reliable_qos = _make_reliable_qos()
    # Policy mode keeps expensive camera conversion off the command worker and
    # reduces snapshot copies to model resolution. The default observation tool
    # continues to return the original full-resolution RGB images.
    image_decoder = _decode_policy_image if policy_images else _decode_image
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
    backends: list[Ros2Backend] = []
    if policy_images:
        camera_keys = ("head_camera", "left_wrist_camera", "right_wrist_camera")
        # A separate existing Ros2Backend is sufficient; neither Robot I/O
        # concurrency nor the CompositeRobot implementation needs to change.
        backends.append(
            Ros2Backend(
                observations={key: observations.pop(key) for key in camera_keys},
                actions={},
                node_name=f"{node_name}_cameras",
                io_timeout_s=io_timeout_s,
            )
        )
        for key in camera_keys:
            schema[key] = FeatureSpec((224, 224, 3), "uint8")
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


def summarize_observation(observation: Observation) -> dict[str, dict[str, Any]]:
    """Return a compact, JSON-friendly summary without copying camera pixels.

    This is only a CLI display helper.  Application code should keep and pass
    the Observation object itself to the policy or action planner.
    """
    summary: dict[str, dict[str, Any]] = {}
    for key, sample in observation.samples.items():
        summary[key] = {
            "shape": tuple(sample.value.shape),
            "dtype": str(sample.value.dtype),
            "received_at_ns": sample.received_at_ns,
        }
        if sample.value.size <= 16:
            summary[key]["values"] = sample.value.tolist()
    return summary


def _hold_action(
    observation: Observation,
    left: float,
    right: float,
    arm_delta_left: Sequence[float] | None = None,
    arm_delta_right: Sequence[float] | None = None,
) -> Action:
    """Build a complete action from state, optionally adding tiny joint deltas.

    Holding the observed joint positions makes the first command a no-op for
    the arms.  The optional deltas are only a CLI convenience for a bounded
    smoke test; they are expressed in radians and are checked in ``main``.
    """
    joint_left = observation.samples["joint_position_left"].value.copy()
    joint_right = observation.samples["joint_position_right"].value.copy()
    if arm_delta_left is not None:
        joint_left += np.asarray(arm_delta_left, dtype=np.float64)
    if arm_delta_right is not None:
        joint_right += np.asarray(arm_delta_right, dtype=np.float64)
    return {
        "joint_position_left": joint_left,
        "joint_position_right": joint_right,
        "gripper_left": np.asarray([left], dtype=np.float32),
        "gripper_right": np.asarray([right], dtype=np.float32),
    }


# The example default resolves to the model directory in the current account.
DEFAULT_RLINF_CHECKPOINT = Path.home() / "models/rlinf-pi05-tianji-step3810"
TIANJI_MODEL_ACTION_DIM = 16
TIANJI_MODEL_ACTION_HORIZON = 50
TIANJI_MODEL_ACTION_PERIOD_S = 0.04  # 50 Hz source data with action_stride=2.
TIANJI_GRIPPER_LIMITS_RAD = (0.0, 1.6)


@lru_cache(maxsize=1)
def _rlinf_reference() -> Any:
    """Reuse the alignment runner rather than duplicate its numerical contract.

    Loading this sibling by path also supports direct script execution, where
    the repository root is not necessarily on sys.path. Heavy model imports
    stay lazy: state-only tools and schema tests do not load PyTorch or CUDA.
    """
    from importlib.util import module_from_spec, spec_from_file_location

    path = Path(__file__).resolve().parents[1] / "pi05" / "run_rlinf.py"
    spec = spec_from_file_location("_tianji_rlinf_reference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the RLinf reference runner at {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_tianji_model_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject a checkpoint with a different state/action or camera convention."""
    expected = {
        "action_dim": TIANJI_MODEL_ACTION_DIM,
        "action_horizon": TIANJI_MODEL_ACTION_HORIZON,
        "action_stride": 2,
        "camera_names": ["head_left", "left_wrist", "right_wrist"],
        "delta_mask": [True] * 7 + [False] + [True] * 7 + [False],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Unsupported Tianji checkpoint {key}: {metadata.get(key)!r}"
            )


@dataclass(frozen=True)
class TianjiPolicyRequest:
    """Preprocessed tensors from one snapshot; noise is owned by the policy."""

    processed: Any


class TianjiPi05Adapter:
    """Implement PolicyAdapter without changing the CompositeRobot contract.

    State is left arm (7 radians), left gripper, right arm (7 radians), right
    gripper. Only position feedback enters state. All three RGB cameras enter
    the vision encoder. EEF and wrench stay in Observation for future models.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        task: str,
        gripper_limits_rad: tuple[float, float] = TIANJI_GRIPPER_LIMITS_RAD,
    ) -> None:
        if not task.strip():
            raise ValueError("task must be non-empty")
        low, high = map(float, gripper_limits_rad)
        if not np.isfinite((low, high)).all() or high <= low:
            raise ValueError("gripper limits must be finite and increasing")
        self._reference = _rlinf_reference()
        self._processor, self.metadata = self._reference.load_processor(checkpoint)
        _validate_tianji_model_metadata(self.metadata)
        self._task = task.strip()
        self._gripper_limits_rad = low, high

    def state_from_observation(self, observation: Observation) -> NDArray[np.float32]:
        """Convert motor-radian feedback to the dataset's normalized gripper state.

        Observation retains all five raw gripper feedback values. Calibration
        maps position to (position-low)/(high-low); do not clip measured state:
        slight negative readings occur at the physical stop and in the dataset.
        Commands, unlike observations, are clamped to the valid [0, 1] range.
        """
        values: list[NDArray[np.float32]] = []
        low, high = self._gripper_limits_rad
        for side in ("left", "right"):
            joints = np.asarray(
                observation.samples[f"joint_position_{side}"].value, dtype=np.float32
            )
            feedback = observation.samples[f"gripper_feedback_{side}"].value
            if joints.shape != (ARM_DOF,) or feedback.shape != (GRIPPER_FEEDBACK_SIZE,):
                raise ValueError(f"Invalid {side} arm/gripper state shape")
            grip = (float(feedback[0]) - low) / (high - low)
            values.extend((joints, np.asarray([grip], dtype=np.float32)))
        state = np.concatenate(values)
        if not np.isfinite(state).all():
            raise ValueError("Tianji policy state contains a non-finite value")
        return state

    def to_request(self, observation: Observation) -> TianjiPolicyRequest:
        """Use the reference RGB resize/padding, state normalization and tokenizer.

        The CLI instruction is injected here, not stored in the Robot. The
        right eye is intentionally absent: head_left uses the left-eye camera.
        """
        payload = {
            "head_left": observation.samples["head_camera"].value,
            "left_wrist": observation.samples["left_wrist_camera"].value,
            "right_wrist": observation.samples["right_wrist_camera"].value,
            "state": self.state_from_observation(observation),
            "task": self._task,
        }
        return TianjiPolicyRequest(
            self._reference.prepare_observation(self._processor, self.metadata, payload)
        )

    def to_actions(self, result: Any, observation: Observation) -> ActionChunk:
        """Decode all 50 predictions; Deployment selects the execution prefix.

        Arm deltas are anchored to the SAME observation used for the request,
        never to newer feedback acquired while inference was running. The
        checkpoint delta_mask leaves both gripper outputs absolute.
        """
        state = self.state_from_observation(observation)
        absolute = self._reference.absolute_actions(
            self._processor, self.metadata, result, state
        )
        values = absolute.detach().cpu().numpy()
        expected = (1, TIANJI_MODEL_ACTION_HORIZON, TIANJI_MODEL_ACTION_DIM)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(
                f"Expected finite model actions with shape {expected}, "
                f"got {values.shape}"
            )
        actions: list[Action] = []
        for target in values[0]:
            actions.append(
                {
                    "joint_position_left": np.asarray(
                        target[:7], dtype=np.float64
                    ).copy(),
                    "joint_position_right": np.asarray(
                        target[8:15], dtype=np.float64
                    ).copy(),
                    "gripper_left": np.asarray(
                        [np.clip(target[7], 0.0, 1.0)], dtype=np.float32
                    ),
                    "gripper_right": np.asarray(
                        [np.clip(target[15], 0.0, 1.0)], dtype=np.float32
                    ),
                }
            )
        return ActionChunk(tuple(actions), TIANJI_MODEL_ACTION_PERIOD_S)


class TianjiPi05Policy:
    """Own the aligned PhyAI engine and a reproducible stream of diffusion noise."""

    def __init__(
        self, checkpoint: Path, *, seed: int = 42, use_cuda_graph: bool = True
    ) -> None:
        import torch
        from phyai.utils import load_config
        from phyai.models.pi05.scheduler_pi05 import PI05Request
        from phyai.models.pi05.configuration_pi05 import PI05Config

        config = load_config(checkpoint, PI05Config)
        self._torch = torch
        self._request_type = PI05Request
        self._chunk_size = int(config.chunk_size)
        self._max_action_dim = int(config.max_action_dim)
        if self._chunk_size != TIANJI_MODEL_ACTION_HORIZON:
            raise ValueError("PI05 config chunk_size must be 50")
        self._generator = torch.Generator().manual_seed(seed)
        self._engine = _rlinf_reference().make_engine(
            checkpoint, use_cuda_graph=use_cuda_graph
        )
        self._lock = Lock()
        self._close_requested = Event()
        self._closed = False

    def predict(self, request: TianjiPolicyRequest) -> Any:
        """Run one synchronous prediction; RobotDeployment owns the worker loop."""
        with self._lock:
            if self._close_requested.is_set():
                raise RuntimeError("TianjiPi05Policy is closed")
            try:
                noise = self._torch.randn(
                    1,
                    self._chunk_size,
                    self._max_action_dim,
                    generator=self._generator,
                    dtype=self._torch.float32,
                )
                processed = request.processed
                with self._torch.inference_mode():
                    # Detach output from any reusable CUDA-graph buffers before
                    # returning it to adapter postprocessing.
                    return (
                        self._engine.step(
                            self._request_type(
                                pixel_values=processed.pixel_values,
                                input_ids=processed.input_ids,
                                lang_lens=processed.lang_lens,
                                noise=noise,
                            )
                        )
                        .detach()
                        .float()
                        .cpu()
                    )
            finally:
                if self._close_requested.is_set():
                    self._close_engine()

    def _close_engine(self) -> None:
        if not self._closed:
            self._engine.close()
            self._closed = True

    def close(self) -> None:
        """Do not destroy CUDA resources underneath a timed-out predictor.

        Hardware is stopped before this call. If inference is still active,
        the worker releases its own engine in predict's finally block.
        """
        self._close_requested.set()
        if self._lock.acquire(blocking=False):
            try:
                self._close_engine()
            finally:
                self._lock.release()


def _policy_options(args: argparse.Namespace) -> DeploymentOptions:
    """Keep source actions in the queue; interpolate lazily on the control thread.

    With action_fps=25, execution_horizon=20 and control_hz=200, a chunk
    occupies 20 queue entries and normally produces 160 sends over 0.8 seconds.
    Missed control ticks within the lateness budget are skipped, not replayed.
    Observation/inference occurs only in the command-free gap between chunks.
    """
    capacity = args.max_queued_actions
    if capacity is None:
        capacity = args.execution_horizon
    return DeploymentOptions(
        control_hz=args.control_hz,
        max_control_lateness_s=args.max_control_lateness_ms / 1000,
        action_hz=args.action_fps,
        max_sample_age_s=0.5,
        startup_timeout_s=10.0,
        observation_timeout_s=args.observation_timeout,
        max_prediction_age_s=args.max_prediction_age,
        max_queued_actions=capacity,
        execution_horizon=args.execution_horizon,
        interpolate_keys=frozenset(ACTION_SCHEMA),
    )


def _check_control_available(gripper_limits_rad: Sequence[float]) -> None:
    """Fail closed instead of taking over a reserved or unhealthy gripper.

    This endpoint is read-only. Release any existing reservation through its
    owner/UI before running; this example never clears faults or steals leases.
    The check is not an atomic lease: the operator must also stop other command
    publishers before enabling User input.
    """
    from urllib.request import urlopen

    with urlopen(
        "http://127.0.0.1:8080/api/v1/gripper/web/status", timeout=3
    ) as response:
        payload = json.load(response)
    if payload.get("ok") is not True or not isinstance(payload.get("data"), dict):
        raise RuntimeError("Invalid gripper status response")
    status = payload["data"]
    if status.get("reserved"):
        raise RuntimeError(
            "Gripper is reserved; release its current session in the UI first"
        )
    if not status.get("available") or not status.get("healthy"):
        raise RuntimeError("Gripper status is unavailable or unhealthy")
    calibration = np.asarray(status.get("calibration"), dtype=np.float64)
    if calibration.shape != (2, 2) or not np.allclose(calibration, gripper_limits_rad):
        raise RuntimeError(
            f"Gripper calibration differs from --gripper-limits-rad: {calibration}"
        )


def _check_observation_age(observation: Observation, max_age_s: float = 0.5) -> None:
    """Reject missing/stale data before warming up or enabling the controller."""
    now = time.monotonic_ns()
    for key, sample in observation.samples.items():
        if not 0 <= now - sample.received_at_ns <= max_age_s * 1e9:
            raise TimeoutError(f"Stale observation before inference: {key}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate schemas and codecs without connecting to ROS",
    )
    parser.add_argument(
        "--observe-seconds",
        type=float,
        default=2.0,
        help="keep the ROS observation session open for this many seconds",
    )
    parser.add_argument(
        "--observation-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for the complete observation",
    )
    parser.add_argument(
        "--prepare-user-control",
        action="store_true",
        help="call set_ready, set_mode, and set_input(3)",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=MODE_JOINT_IMPEDANCE,
        choices=(MODE_JOINT_POSITION, MODE_CARTESIAN_IMPEDANCE, MODE_JOINT_IMPEDANCE),
        help="Tianji mode used by --prepare-user-control (default: 3)",
    )
    parser.add_argument(
        "--velocity-ratio",
        type=int,
        default=None,
        help="optional Tianji velocity ratio [0, 100]",
    )
    parser.add_argument(
        "--write-once",
        action="store_true",
        help="publish one complete action, never start a periodic publisher",
    )
    parser.add_argument(
        "--arm-delta-left",
        type=float,
        nargs=ARM_DOF,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="optional one-shot left-arm delta in radians (max abs 0.05)",
    )
    parser.add_argument(
        "--arm-delta-right",
        type=float,
        nargs=ARM_DOF,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="optional one-shot right-arm delta in radians (max abs 0.05)",
    )
    parser.add_argument(
        "--gripper-left",
        type=float,
        default=None,
        help="one-shot normalized left gripper target in [0, 1]",
    )
    parser.add_argument(
        "--gripper-right",
        type=float,
        default=None,
        help="one-shot normalized right gripper target in [0, 1]",
    )
    parser.add_argument(
        "--run-policy",
        action="store_true",
        help="run the RLinf pi0.5 policy through RobotDeployment",
    )
    parser.add_argument(
        "--policy-dry-run",
        action="store_true",
        help="run one policy inference and print actions without publishing",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="instruction injected into the PI05 policy input",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_RLINF_CHECKPOINT,
        help="converted RLinf pi0.5 checkpoint directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="CPU diffusion-noise seed; successive predictions still use fresh noise",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="use eager PI05 execution instead of CUDA graphs",
    )
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--continuous",
        action="store_true",
        help="keep running the live policy until Ctrl-C or a safety failure",
    )
    duration.add_argument(
        "--steps",
        type=int,
        default=None,
        help="number of control ticks before shutdown (default: 160, or 0.8s at 200Hz)",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=200.0,
        help="RobotDeployment command frequency",
    )
    parser.add_argument(
        "--max-control-lateness-ms",
        type=float,
        default=10.0,
        help="maximum lateness past a scheduled send, in milliseconds (default: 10)",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=20,
        help="number of source model actions to execute per prediction",
    )
    parser.add_argument(
        "--max-queued-actions",
        type=int,
        default=None,
        help="queue capacity in source model actions (default: execution horizon)",
    )
    parser.add_argument(
        "--action-fps",
        type=float,
        default=25.0,
        help="source action frequency; 25 fps means 40 ms between model targets",
    )
    parser.add_argument(
        "--max-prediction-age",
        type=float,
        default=2.0,
        help="maximum snapshot-to-command age in seconds",
    )
    parser.add_argument(
        "--gripper-limits-rad",
        type=float,
        nargs=2,
        default=TIANJI_GRIPPER_LIMITS_RAD,
        metavar=("LOW", "HIGH"),
        help="raw motor-radian feedback limits used to normalize grippers",
    )
    return parser


def _run_self_test() -> int:
    if set(OBSERVATION_SCHEMA) != {
        "head_camera",
        "left_wrist_camera",
        "right_wrist_camera",
        "joint_position_left",
        "joint_position_right",
        "eef_left",
        "eef_right",
        "wrench_left",
        "wrench_right",
        "gripper_feedback_left",
        "gripper_feedback_right",
    }:
        raise AssertionError("observation schema is incomplete")
    if set(ACTION_SCHEMA) != {
        "joint_position_left",
        "joint_position_right",
        "gripper_left",
        "gripper_right",
    }:
        raise AssertionError("action schema is incomplete")
    print("Tianji schema self-test passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.steps is None:
        args.steps = 160
    if args.self_test:
        return _run_self_test()
    if args.policy_dry_run:
        args.run_policy = True
    if args.continuous and (
        not args.run_policy or args.policy_dry_run or args.write_once
    ):
        parser.error("--continuous requires a live --run-policy session")
    if not math.isfinite(args.observe_seconds) or args.observe_seconds < 0:
        parser.error("--observe-seconds must be finite and non-negative")
    if args.policy_dry_run and (args.prepare_user_control or args.write_once):
        parser.error("--policy-dry-run cannot prepare control or send actions")
    if args.run_policy and args.write_once:
        parser.error("--run-policy cannot be combined with --write-once")
    if args.run_policy and (not args.task or not args.task.strip()):
        parser.error("policy execution requires a non-empty --task")
    if (
        args.write_once or (args.run_policy and not args.policy_dry_run)
    ) and not args.prepare_user_control:
        parser.error("sending actions requires explicit --prepare-user-control")
    if args.run_policy and args.mode == MODE_CARTESIAN_IMPEDANCE:
        parser.error(
            "joint policy commands require --mode 1 or 3, not Cartesian mode 2"
        )
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 1 <= args.execution_horizon <= TIANJI_MODEL_ACTION_HORIZON:
        parser.error("--execution-horizon must be between 1 and 50")
    try:
        options = _policy_options(args)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if args.write_once and (args.gripper_left is None or args.gripper_right is None):
        parser.error("--write-once requires both gripper targets")
    for name in ("gripper_left", "gripper_right"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    for name in ("arm_delta_left", "arm_delta_right"):
        delta = getattr(args, name)
        if delta is not None and (
            not np.isfinite(delta).all()
            or max(abs(value) for value in delta) > CLI_MAX_ARM_DELTA_RAD
        ):
            parser.error(
                f"--{name.replace('_', '-')} must be finite with "
                f"abs <= {CLI_MAX_ARM_DELTA_RAD} rad"
            )

    # Start read-only even for an intended control run. Loading/warming up the
    # model must not enable actuators, and an early exception must not stop a
    # different operator's controller. prepare_user_control explicitly opts in.
    control = TianjiControl(read_only=True)
    robot: CompositeRobot | None = None
    policy: TianjiPi05Policy | None = None
    try:
        control.connect()
        robot = make_tianji_robot(control, policy_images=args.run_policy)
        robot.connect()
        observation = wait_for_observation(robot, timeout_s=args.observation_timeout)
        print("Observation (phyai_robot.Observation):", flush=True)
        for key, details in summarize_observation(observation).items():
            print(f"  {key}: {details}", flush=True)

        if args.run_policy:
            checkpoint = args.checkpoint.expanduser().resolve()
            adapter = TianjiPi05Adapter(
                checkpoint,
                task=args.task,
                gripper_limits_rad=tuple(args.gripper_limits_rad),
            )
            policy = TianjiPi05Policy(
                checkpoint, seed=args.seed, use_cuda_graph=not args.no_cuda_graph
            )
            # Engine initialization and the first CUDA-graph execution may be
            # slow. Warm up before mode/input changes, using a NEW snapshot.
            observation = wait_for_observation(
                robot, timeout_s=args.observation_timeout, max_age_s=0.5
            )
            _check_observation_age(observation)
            start = time.monotonic()
            request = adapter.to_request(observation)
            chunk = adapter.to_actions(policy.predict(request), observation)
            elapsed = time.monotonic() - start
            state = adapter.state_from_observation(observation)
            first = chunk.actions[0]
            print(
                f"Policy warmup/dry-run: {len(chunk.actions)} actions, "
                f"dt={chunk.step_period_s}s, elapsed={elapsed:.3f}s",
                flush=True,
            )
            print(
                "Policy state (left_q, left_gripper, right_q, right_gripper): "
                f"{state.tolist()}",
                flush=True,
            )
            print(
                "First predicted action (not sent):",
                {key: value.tolist() for key, value in first.items()},
                flush=True,
            )
            if args.policy_dry_run:
                print(
                    "Read-only dry-run complete: no actions or mode/input changes.",
                    flush=True,
                )
                return 0

            _check_control_available(args.gripper_limits_rad)
            observation = wait_for_observation(
                robot, timeout_s=args.observation_timeout, max_age_s=0.5
            )
            _check_observation_age(observation)
            control.prepare_user_control(
                mode=args.mode, velocity_ratio=args.velocity_ratio
            )
            print(
                f"Tianji control prepared: mode={args.mode}, input={INPUT_USER}",
                flush=True,
            )
            print(
                f"Starting deployment: {options.control_hz:g} Hz, "
                f"execution_horizon={options.execution_horizon} model steps, "
                f"action_fps={options.action_hz:g}, "
                f"max_control_lateness_ms={options.max_control_lateness_s * 1000:g}, "
                f"queue_capacity={options.max_queued_actions} source actions. "
                "Warmup actions are discarded.",
                flush=True,
            )
            deployment = RobotDeployment(
                robot=robot, policy=policy, adapter=adapter, options=options
            )
            if args.continuous:
                print("Continuous policy running; press Ctrl-C to stop.", flush=True)
            deployment.run(max_steps=None if args.continuous else args.steps)
            if args.continuous:
                print("Continuous policy deployment stopped", flush=True)
            else:
                print(
                    f"Policy deployment completed {args.steps} control ticks",
                    flush=True,
                )
        else:
            if args.prepare_user_control:
                _check_control_available(args.gripper_limits_rad)
                _check_observation_age(observation)
                control.prepare_user_control(
                    mode=args.mode, velocity_ratio=args.velocity_ratio
                )
                print(
                    f"Tianji control prepared: mode={args.mode}, input={INPUT_USER}",
                    flush=True,
                )
            if args.write_once:
                assert args.gripper_left is not None and args.gripper_right is not None
                action = _hold_action(
                    observation,
                    args.gripper_left,
                    args.gripper_right,
                    args.arm_delta_left,
                    args.arm_delta_right,
                )
                robot.send_action(action)
                print("Sent one complete Tianji action", flush=True)
        if args.observe_seconds:
            time.sleep(args.observe_seconds)
        return 0
    finally:
        # CompositeRobot remains responsible for backend stop/close. Read-only
        # sessions make its device callback a no-op; live sessions select idle.
        try:
            if robot is not None:
                robot.close()
        finally:
            try:
                if policy is not None:
                    policy.close()
            finally:
                control.close()


if __name__ == "__main__":
    raise SystemExit(main())
