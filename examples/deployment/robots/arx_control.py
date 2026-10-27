"""Explicit ARX preparation for a workbench service; no model or Enactive dependency."""

from __future__ import annotations

import time
import numpy as np

from phyai_robot import CompositeRobot
from phyai_robot.remote import Embodiment, PolicySpec
from phyai_robot.robot import wait_for_observation

from .arx import ACTION_SCHEMA, build


class ControlledRobot(CompositeRobot):
    def __init__(self, owner, **kwargs):
        self.owner = owner
        super().__init__(**kwargs)

    def close(self):
        try:
            super().close()
        except BaseException as error:
            self.owner.cleanup_error = str(error)
            raise


class ArxControl:
    def __init__(self, client_config, preparation):
        self.config = client_config
        self.cleanup_error = ""
        self.ready = np.asarray(preparation["ready_position"], dtype=np.float64)
        self.policy = np.asarray(preparation["policy_position"], dtype=np.float64)
        self.duration = float(preparation.get("transition_s", 3.75))
        self.tolerance = float(preparation.get("tolerance", 0.08))
        self.bounds = None
        if "joint_position_min" in preparation or "joint_position_max" in preparation:
            lower = np.asarray(preparation["joint_position_min"], dtype=np.float64)
            upper = np.asarray(preparation["joint_position_max"], dtype=np.float64)
            if (
                lower.shape != (6,)
                or upper.shape != (6,)
                or not np.isfinite([lower, upper]).all()
                or not (lower < upper).all()
            ):
                raise ValueError("ARX joint bounds require six finite ordered ranges")
            self.bounds = (lower, upper)
        if any(
            p.shape != (14,) or not np.isfinite(p).all()
            for p in (self.ready, self.policy)
        ):
            raise ValueError("ARX preparation requires two finite 14D poses")
        if (
            not np.isfinite(self.duration)
            or self.duration <= 0
            or not np.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            raise ValueError("ARX preparation duration and tolerance must be positive")

    def observe(self, robot):
        return wait_for_observation(
            robot,
            timeout_s=self.config.control.observation_timeout_s,
            max_age_s=self.config.control.max_sample_age_s,
        )

    @staticmethod
    def positions(observation):
        return np.concatenate(
            [
                observation.samples[key].value
                for key in (
                    "joint_position_left",
                    "gripper_left",
                    "joint_position_right",
                    "gripper_right",
                )
            ]
        )

    @staticmethod
    def action(values):
        return {
            "joint_position_left": values[:6].copy(),
            "gripper_left": values[6:7].copy(),
            "joint_position_right": values[7:13].copy(),
            "gripper_right": values[13:14].copy(),
        }

    def guard(self, action):
        if self.bounds is not None:
            lower, upper = self.bounds
            for side in ("left", "right"):
                values = action[f"joint_position_{side}"]
                if not ((values >= lower) & (values <= upper)).all():
                    raise ValueError(
                        f"ARX {side} target exceeds configured driver joint ranges"
                    )

    def require_ready(self, robot):
        values = self.positions(self.observe(robot))
        indices = [i for i in range(14) if i not in (6, 13)]
        error = float(np.max(np.abs(values[indices] - self.ready[indices])))
        if error > self.tolerance:
            raise RuntimeError(
                f"机械臂尚未到达预备位（最大偏差 {error:.4f} rad），请先预备"
            )

    def move(self, robot, target, cancel):
        current = self.positions(self.observe(robot))
        started = time.monotonic()
        interval = 1 / self.config.control.control_hz
        while not cancel.is_set():
            elapsed = time.monotonic() - started
            t = min(1.0, elapsed / self.duration)
            eased = 10 * t**3 - 15 * t**4 + 6 * t**5
            robot.send_action(self.action(current + eased * (target - current)))
            if t >= 1:
                return
            cancel.wait(interval)
        raise RuntimeError("机械臂操作已取消")

    def prepare(self, cancel):
        # Build the same native command backend but open no camera subscriptions.
        embodiment = build(self.config.robot, PolicySpec({}, {}, {"image_size": 224}))
        joints = embodiment.robot.backends[-1]
        robot = ControlledRobot(
            self,
            observation_schema=ACTION_SCHEMA,
            action_schema=ACTION_SCHEMA,
            backends=[joints],
            action_guard=self.guard,
        )
        try:
            robot.connect()
            self.observe(robot)
            if cancel.is_set():
                return
            if embodiment.activate:
                embodiment.activate()
            self.move(robot, self.ready, cancel)
            self.require_ready(robot)
        finally:
            robot.close()

    def build_run(self, spec, cancel):
        embodiment = build(self.config.robot, spec)
        robot = ControlledRobot(
            self,
            observation_schema=embodiment.robot.observation_schema,
            action_schema=embodiment.robot.action_schema,
            backends=embodiment.robot.backends,
            action_guard=self.guard,
        )

        def activate():
            if cancel.is_set():
                return
            self.require_ready(robot)
            if embodiment.activate:
                embodiment.activate()
            self.move(robot, self.policy, cancel)

        return Embodiment(robot, embodiment.adapter, activate)
