"""YAML-friendly settings for the shared RobotDeployment control loop.

This schema is independent of robot hardware and policy layout. The entry point
chooses which continuous action fields to interpolate and checks model-specific
horizon constraints. DeploymentOptions remains the authority for timing and
queue-capacity validation; no second scheduler is implemented here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable

from phyai_robot import DeploymentOptions


@dataclass
class DeploymentConfig:
    """Source/control frequencies in Hz, timeout and age limits in seconds.

    Queue sizes and execution_horizon count source targets, not interpolated
    control ticks. The plain YAML fields intentionally omit interpolate_keys:
    that set is chosen from the actual robot action schema during composition.
    """

    control_hz: float = 200.0
    action_hz: float = 25.0
    execution_horizon: int = 20
    max_queued_actions: int = 20
    max_sample_age_s: float = 0.5
    startup_timeout_s: float = 10.0
    observation_timeout_s: float = 10.0
    max_prediction_age_s: float = 2.0
    shutdown_timeout_s: float = 2.0
    max_control_lateness_s: float = 0.01

    def to_options(self, *, interpolate_keys: Iterable[str] = ()) -> DeploymentOptions:
        """Validate loop settings and bind interpolation to the caller's fields."""
        return DeploymentOptions(
            **asdict(self), interpolate_keys=frozenset(interpolate_keys)
        )
