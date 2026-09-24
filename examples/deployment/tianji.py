"""Deploy a PhyAI pi0.5 policy on Tianji using the standard CompositeRobot.

Environment setup
-----------------
Run from the checkout root on the robot host. Use the prepared PhyAI environment
with CUDA, phyai-robot, and the ``deployment`` dependency group (OmegaConf). For
an already provisioned GPU environment missing only the configuration parser,
install it without replacing the host's prepared Torch/FlashInfer packages::

    uv pip install --python .venv/bin/python 'omegaconf>=2.3,<2.4'

Set host-specific shell variables outside this file: ``TIANJI_ROS_SETUP`` names
the ROS setup script including Tianji message/service packages;
``TIANJI_ROS_PYTHON`` names the host Python that can import rclpy and those
packages; ``CUDA_HOME`` names the compatible CUDA toolkit. ROS and PhyAI Python
must have the same minor version, because rclpy contains compiled extensions.
Keep the host's DDS/domain/network configuration when sourcing ROS::

    source "${TIANJI_ROS_SETUP:?Set the Tianji ROS setup script}"
    source .venv/bin/activate
    ROS_SITE=$("${TIANJI_ROS_PYTHON:?Set the host ROS Python}" -c \
        'import site; print(site.getsitepackages()[0])')
    export PYTHONPATH="$ROS_SITE${PYTHONPATH:+:$PYTHONPATH}"
    export TRITON_PTXAS_BLACKWELL_PATH="${CUDA_HOME:?Set the CUDA toolkit}/bin/ptxas"
    python -c 'import rclpy, marvin_msgs, sensor_msgs, phyai_robot, omegaconf'

This uses PhyAI's Python and adds the host's compatible ROS packages rather
than installing a second ROS distribution. On Thor the assembler must support
the GPU; use the host CUDA toolkit instead of an incompatible bundled ptxas.
Use ``uv run --no-sync`` for all commands below if launching through uv.

Configuration and launch
------------------------
Edit ``examples/deployment/configs/tianji.yaml`` or keep host-local settings in
an ignored copy. The checkpoint must contain converted pi0.5 weights/config,
tokenizer files, ``norm_stats.json``, and compatible deployment metadata::

    mkdir -p .cache
    cp examples/deployment/configs/tianji.yaml .cache/tianji.local.yaml
    # Edit policy.checkpoint in that copy; use your local checkpoint directory.
    python examples/deployment/tianji.py --config .cache/tianji.local.yaml --print-config

``--print-config`` only prints merged YAML; it does not connect to ROS, validate
checkpoint contents, or allocate GPU memory. Without it, this entry point WILL
enable control and execute model actions. Before launching, clear the workspace,
check the hardware emergency stop, stop competing command publishers, and
make sure no other controller owns the robot or grippers. This example does
not discover competing controllers or verify gripper calibration automatically;
those are operator checks. It never clears faults or homes the robot.

Start with a bounded trial, then omit max_steps for continuous deployment::

    python examples/deployment/tianji.py --config .cache/tianji.local.yaml max_steps=160
    python examples/deployment/tianji.py --config .cache/tianji.local.yaml \
        task="Plug in the Ethernet cable" deployment.execution_horizon=20

The second command uses the default ``max_steps: null`` and runs until Ctrl-C
or an error. If your YAML sets a finite bound, override it with ``max_steps=null``.
The module form, ``python -m examples.deployment.tianji``, is equivalent.

Precedence is typed schema < bundled YAML < selected YAML < dotted ``key=value``
overrides. Partial YAML files are supported. Unknown keys and incompatible types
fail before startup. Quote values with spaces and use YAML ``null``/``true``/
``false`` for nullable/boolean fields. ``--config`` is relative to your working
directory; checkpoint/kernel paths are relative to the checkout root, regardless
of YAML location. Home-directory expansion is supported. Metadata paths are
relative to the checkpoint. No change of working directory is performed.

Configuration reference
-----------------------
* ``task``: instruction injected by the adapter into each policy request.
* ``max_steps``: positive number of actual Robot.send_action calls, or null for
  continuous execution. It does not count source actions or skipped ticks.
* ``robot.mode``: 3 = joint impedance (default); 1 = joint position. Both consume
  user joint commands; mode 2 is Cartesian control and is rejected here.
* ``robot.velocity_ratio``: optional integer controller percentage, 0 through
  100; null leaves the current value unchanged. This is not a collision guard.
* ``robot.node_name``: base name for private robot/camera/control ROS nodes.
* ``robot.service_timeout_s``: per-service availability/response timeout.
* ``robot.io_timeout_s``: CompositeRobot backend read/write/stop timeout.
* ``adapter.metadata_file``: explicit filename or null to prefer
  deployment_metadata.json, otherwise require one unambiguous *_metadata.json.
  Metadata must describe 16 action dimensions, 50 steps, stride 2, the three
  expected cameras, the arm-only delta mask, and positive norm_eps.
* ``adapter.gripper_limits_rad``: [closed, open] motor positions for both grippers.
  Set these to the actual hardware calibration; there is no automatic lookup.
  State is normalized without clipping;
  predicted gripper commands are clamped to [0, 1].
* ``policy.checkpoint``: required checkpoint directory; no host-specific default.
* ``policy.kernel_policy``: kernel-selection YAML. The supplied BF16 static
  configuration preserves exact GELU/GEGLU semantics.
* ``policy.seed``: CPU diffusion-noise seed; each prediction draws fresh noise.
* ``policy.use_cuda_graph``: enable engine graph capture/replay.
* ``policy.num_threads``: PhyAI runtime CPU threads, not ROS executor threads.
* ``deployment.action_hz``: source action rate, default 25 Hz (40 ms spacing).
* ``deployment.control_hz``: independent command rate, default 200 Hz (5 ms).
* ``deployment.execution_horizon``: retained prefix, 1 through 50; default 20.
* ``deployment.max_queued_actions``: source-target capacity, at least the retained
  prefix size. It is not a capacity in interpolated 200 Hz commands.
* ``deployment.max_sample_age_s``: maximum local receive age of every snapshot
  field, including cameras, EEF, wrench, and gripper feedback.
* ``deployment.startup_timeout_s``: budget for the first executable chunk after
  enabling control. Model loading and discarded warmup happen before this.
* ``deployment.observation_timeout_s``: wait for complete/fresh sensors before
  warmup, before enabling control, and at each subsequent queue refill.
* ``deployment.max_prediction_age_s``: maximum snapshot-to-send age, including
  observation processing, inference, and execution of the retained prefix.
* ``deployment.shutdown_timeout_s``: wait for in-flight inference AFTER hardware
  stop/close. A timed-out predictor releases its engine when it returns.
* ``deployment.max_control_lateness_s``: maximum scheduled-send lateness,
  default 0.01 s (10 ms), independent of the 5 ms nominal control period.

Execution and shutdown
----------------------
The adapter uses left-eye RGB as the head camera plus both wrist RGB cameras,
not the right eye. State is left arm (7 radians), left gripper, right arm (7),
right gripper. EEF, wrench, and all gripper feedback remain in Observation but
are not model inputs yet. All 50 decoded actions are checked; arm deltas are
anchored to the SAME observation used for inference, not a later reading.

Model loading and one discarded warmup finish before service calls select
ready, the configured joint mode, optional velocity ratio, and User/Custom
input 3. RobotDeployment then reads a new observation and predicts only when
its thread-safe source-action queue is empty. The independent control loop
linearly interpolates arm/gripper targets, holds the last retained target for
one source interval, and sends nothing while the queue is empty. With defaults,
20 targets span 0.8 seconds and normally produce 160 complete commands. This is
stop-and-refill execution, not cross-chunk blending or real-time chunking.

Each Robot.send_action is a single complete write, including both grippers;
continuous publication belongs to RobotDeployment, not Robot. Missed ticks
within the lateness limit are skipped rather than replayed in a burst. The
trajectory clock is unchanged, so a delayed chunk may contain fewer sends.
The limit measures host submission lateness, not end-to-end ROS/motor latency;
a blocking send cannot be interrupted by that check. Python/ROS is not hard
real-time. Stale sensors/predictions, excessive lateness, or other failures stop
the loop. Normal completion, Ctrl-C, and handled failures use CompositeRobot's
stop/close path to request input 0 before tearing down ROS. Initialization
failures before control is enabled do not change input. Stop is not a homing
command or a replacement for the physical emergency stop.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from contextlib import ExitStack
from dataclasses import field, dataclass
from collections.abc import Sequence

# Direct script execution places only examples/deployment on sys.path. Add the
# checkout root so both invocation forms use the same package-relative modules.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from phyai_robot import RobotDeployment

from .options import DeploymentConfig
from .policies import Pi05Policy, Pi05PolicyConfig
from .configuration import parse_config, repository_path
from .robots.tianji import (
    ACTION_SCHEMA,
    TianjiControl,
    TianjiRobotConfig,
    make_tianji_robot,
    wait_for_observation,
)
from .adapters.tianji_pi05 import TianjiPi05Adapter, TianjiPi05AdapterConfig

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "tianji.yaml"


@dataclass
class TianjiDeploymentConfig:
    """Compose the schemas for this example; no device logic lives in the loader."""

    task: str = "Plug in the Ethernet cable"
    max_steps: int | None = None
    robot: TianjiRobotConfig = field(default_factory=TianjiRobotConfig)
    adapter: TianjiPi05AdapterConfig = field(default_factory=TianjiPi05AdapterConfig)
    policy: Pi05PolicyConfig = field(default_factory=Pi05PolicyConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    def validate(self) -> None:
        """Validate each module's settings before allocating model or ROS resources."""
        if not self.task.strip():
            raise ValueError("task must be non-empty")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive or null")
        self.robot.validate()
        self.adapter.validate(execution_horizon=self.deployment.execution_horizon)
        self.policy.validate()
        self.deployment.to_options()


def main(argv: Sequence[str] | None = None) -> None:
    """Compose the deployment; device protocol and model layout live elsewhere."""
    config = parse_config(
        TianjiDeploymentConfig,
        default_config=DEFAULT_CONFIG,
        argv=argv,
        validate=TianjiDeploymentConfig.validate,
    )
    options = config.deployment.to_options(interpolate_keys=ACTION_SCHEMA)
    checkpoint = repository_path(config.policy.checkpoint)
    with ExitStack() as resources:
        control = TianjiControl(
            node_name=f"{config.robot.node_name}_control",
            service_timeout_s=config.robot.service_timeout_s,
        )
        resources.callback(control.close)
        adapter = TianjiPi05Adapter(
            checkpoint,
            task=config.task,
            metadata_file=config.adapter.metadata_file,
            gripper_limits_rad=tuple(config.adapter.gripper_limits_rad),
        )
        policy = Pi05Policy(
            checkpoint,
            kernel_policy=repository_path(config.policy.kernel_policy),
            seed=config.policy.seed,
            use_cuda_graph=config.policy.use_cuda_graph,
            num_threads=config.policy.num_threads,
        )
        resources.callback(policy.close)
        robot = make_tianji_robot(
            control,
            node_name=config.robot.node_name,
            io_timeout_s=config.robot.io_timeout_s,
            image_size=adapter.image_size,
        )
        # ExitStack unwinds robot -> policy -> control even during partial
        # startup. The stop callback still needs the control service executor.
        resources.callback(robot.close)
        deployment = RobotDeployment(
            robot=robot, policy=policy, adapter=adapter, options=options
        )
        control.connect()
        robot.connect()
        observation = wait_for_observation(
            robot,
            timeout_s=options.observation_timeout_s,
            max_age_s=options.max_sample_age_s,
        )
        started = time.monotonic()
        warmup = adapter.to_actions(
            policy.predict(adapter.to_request(observation)), observation
        )
        print(
            f"Warmup: {time.monotonic() - started:.3f}s; discarded {len(warmup.actions)} actions",
            flush=True,
        )
        # Never execute warmup targets or enable control using stale startup
        # sensors. RobotDeployment obtains another fresh snapshot for its queue.
        wait_for_observation(
            robot,
            timeout_s=options.observation_timeout_s,
            max_age_s=options.max_sample_age_s,
        )
        control.prepare_user_control(
            mode=config.robot.mode, velocity_ratio=config.robot.velocity_ratio
        )
        print(
            f"Tianji input=3, mode={config.robot.mode}; source={options.action_hz:g}Hz, control={options.control_hz:g}Hz, horizon={options.execution_horizon}, lateness<={options.max_control_lateness_s * 1000:g}ms; Ctrl-C stops",
            flush=True,
        )
        deployment.run(max_steps=config.max_steps)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Cleanup has already run through RobotDeployment and ExitStack.
        print("Deployment interrupted; robot cleanup requested.", file=sys.stderr)
        raise SystemExit(130) from None
