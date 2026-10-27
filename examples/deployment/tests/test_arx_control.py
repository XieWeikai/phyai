"""Preparation trajectories use measured joints and do not require model weights."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from phyai_robot import CompositeRobot
from phyai_robot.backends.mock import MockBackend
from phyai_robot.remote import Embodiment

from examples.deployment.arx import ArxClientConfig
from examples.deployment.robots.arx import ACTION_SCHEMA
from examples.deployment.robots.arx_control import ArxControl


def setup():
    cfg = ArxClientConfig()
    cfg.control.control_hz = 500
    control = ArxControl(
        cfg,
        {
            "ready_position": [0.02] * 14,
            "policy_position": [0.03] * 14,
            "transition_s": 0.01,
            "tolerance": 0.001,
        },
    )
    backend = MockBackend(
        initial_values={
            k: np.zeros(s.shape, dtype=s.dtype) for k, s in ACTION_SCHEMA.items()
        },
        action_to_observation={k: k for k in ACTION_SCHEMA},
    )
    robot = CompositeRobot(
        observation_schema=ACTION_SCHEMA,
        action_schema=ACTION_SCHEMA,
        backends=[backend],
    )
    return control, robot, backend


def test_prepare_reaches_measured_target_without_model_or_camera_backend(monkeypatch):
    control, robot, backend = setup()
    calls = []
    monkeypatch.setattr(
        "examples.deployment.robots.arx_control.build",
        lambda *args: Embodiment(robot, None, lambda: calls.append("activate")),
    )
    control.prepare(threading.Event())
    assert calls == ["activate"] and backend.write_count > 1
    np.testing.assert_allclose(backend.last_action["joint_position_left"], [0.02] * 6)
    assert backend.stop_count == 1


def test_preparation_rejects_unprepared_pose_and_cancellation_sends_no_target():
    control, robot, backend = setup()
    robot.connect()
    try:
        with pytest.raises(RuntimeError, match="尚未到达预备位"):
            control.require_ready(robot)
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(RuntimeError, match="取消"):
            control.move(robot, control.ready, cancel)
        assert backend.write_count == 0
    finally:
        robot.close()


def test_ready_check_ignores_raw_gripper_calibration():
    control, robot, backend = setup()
    robot.connect()
    try:
        values = control.ready.copy()
        values[[6, 13]] = [-4.78, -0.08]
        robot.send_action(control.action(values))
        control.require_ready(robot)
    finally:
        robot.close()


def test_cleanup_failure_is_retained_for_workbench_handoff(monkeypatch):
    from examples.deployment.robots.arx_control import ControlledRobot

    control, robot, backend = setup()
    tracked = ControlledRobot(
        control,
        observation_schema=ACTION_SCHEMA,
        action_schema=ACTION_SCHEMA,
        backends=[backend],
    )
    tracked.connect()
    monkeypatch.setattr(
        backend, "stop", lambda: (_ for _ in ()).throw(RuntimeError("stop failed"))
    )
    with pytest.raises(BaseExceptionGroup):
        tracked.close()
    assert control.cleanup_error
