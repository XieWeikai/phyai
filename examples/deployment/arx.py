"""ARX X5 2025 / AC One client. Use --print-config to inspect without robot I/O."""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from .client import RemoteConfig, ControlConfig, ScheduleConfig, main
from .robots.arx import ArxConfig, build

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "arx.yaml"


@dataclass
class ArxClientConfig:
    max_steps: int | None = None
    robot: ArxConfig = field(default_factory=ArxConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    schedule: ScheduleConfig = field(
        default_factory=lambda: ScheduleConfig(action_hz=30.0, execution_horizon=25)
    )


if __name__ == "__main__":
    main(ArxClientConfig, build, default_config=DEFAULT_CONFIG)
