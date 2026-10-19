"""Resident DM0.5 service using the existing PhyAI PolicyServer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from phyai_robot.remote import PolicyServer

from examples.deployment.configuration import parse_config
from examples.deployment.dm05 import DM05Policy, DM05PolicyConfig, DM05ProcessorConfig


@dataclass
class DM05ServerConfig:
    task: str = ""
    endpoint: str = "tcp://127.0.0.1:5555"
    lease_timeout_s: float = 60.0
    policy: DM05PolicyConfig = field(default_factory=DM05PolicyConfig)
    processor: DM05ProcessorConfig = field(default_factory=DM05ProcessorConfig)

    def validate(self):
        if (
            not self.task.strip()
            or not self.endpoint.strip()
            or self.lease_timeout_s <= 0
        ):
            raise ValueError("Task, endpoint and lease timeout must be valid")
        self.policy.validate()
        self.processor.validate()


def main(argv=None):
    config = parse_config(
        DM05ServerConfig,
        default_config=Path(__file__).with_name("configs") / "dm05_tianji.yaml",
        argv=argv,
        validate=DM05ServerConfig.validate,
    )
    policy = DM05Policy(config.policy, config.processor, task=config.task)
    try:
        PolicyServer(
            policy=policy,
            spec=policy.spec(),
            reset_run=policy.reset_run,
            endpoint=config.endpoint,
            lease_timeout_s=config.lease_timeout_s,
        ).serve()
    finally:
        policy.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
