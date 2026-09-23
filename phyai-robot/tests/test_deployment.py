"""Use real mock feedback and controlled policy delays, without GPU or hardware."""

import time
import threading
from typing import Any
from collections.abc import Mapping, Callable

import numpy as np
import pytest
from phyai_robot import (
    Robot,
    Action,
    Policy,
    Sample,
    ActionChunk,
    FeatureSpec,
    Observation,
    ActionSchema,
    EnginePolicy,
    CompositeRobot,
    RobotDeployment,
    DeploymentOptions,
)
from numpy.typing import NDArray
from phyai_robot.deployment import _resample
from phyai_robot.backends.mock import MockBackend


class StepAdapter:
    def to_request(self, observation: Observation) -> NDArray[np.float32]:
        return observation.samples["q"].value.copy()

    def to_actions(
        self, result: NDArray[np.float32], observation: Observation
    ) -> ActionChunk:
        assert result.shape == observation.samples["q"].value.shape
        return ActionChunk(tuple({"target": result.copy()} for _ in range(4)), 0.05)


class StepPolicy:
    def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
        return request + np.float32(0.1)


def options(**overrides: Any) -> DeploymentOptions:
    # Tests override heterogeneous fields, including deliberately invalid values.
    values: dict[str, Any] = {
        "control_hz": 20,
        "max_sample_age_s": 1,
        "max_prediction_age_s": 2,
        "max_queued_actions": 8,
        "queue_low_watermark": 2,
    }
    values.update(overrides)
    return DeploymentOptions(**values)


def deploy(
    robot: Robot,
    policy: Policy[NDArray[np.float32], NDArray[np.float32]] | None = None,
    **overrides: Any,
) -> RobotDeployment[NDArray[np.float32], NDArray[np.float32]]:
    return RobotDeployment(
        robot=robot,
        policy=policy or StepPolicy(),
        adapter=StepAdapter(),
        options=options(**overrides),
    )


def test_mock_closed_loop_refills_and_exits(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(robot)
    loop.run(max_steps=12)
    assert backend.write_count == 12
    assert backend.last_action is not None
    assert backend.last_action["target"][0] > 0.1
    assert backend.stop_count == 1
    with pytest.raises(RuntimeError, match="single-use"):
        loop.run()


def test_engine_policy_preserves_request_and_does_not_own_lifecycle() -> None:
    class Engine:
        def step(self, request: object) -> object:
            return request

    request: object = object()
    assert EnginePolicy[object, object](Engine()).predict(request) is request
    with pytest.raises(TypeError):
        EnginePolicy(object())


def test_stale_sensor_stops_before_any_command(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    original: Callable[[], Mapping[str, Sample]] = backend.read

    def stale_read() -> Mapping[str, Sample]:
        return {key: Sample(sample.value, 0) for key, sample in original().items()}

    monkeypatch.setattr(backend, "read", stale_read)
    with pytest.raises(TimeoutError, match="stale observation"):
        deploy(robot, max_sample_age_s=0.1).run(max_steps=1)
    assert backend.write_count == 0
    assert backend.stop_count == 1


def test_policy_error_is_propagated_and_robot_stops(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend

    class Broken:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            raise ValueError("prediction failed")

    with pytest.raises(ValueError, match="prediction failed"):
        deploy(robot, Broken()).run()
    assert backend.stop_count == 1
    assert backend.write_count == 0


def test_external_stop_discards_in_flight_prediction(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    entered, release = threading.Event(), threading.Event()

    class Waiting:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            entered.set()
            release.wait(2)
            return request

    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(
        robot, Waiting()
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            loop.run()
        except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
            errors.append(error)

    actor: threading.Thread = threading.Thread(target=run)
    actor.start()
    assert entered.wait(2)
    loop.stop()
    release.set()
    actor.join(3)
    assert not actor.is_alive()
    assert not errors
    assert backend.write_count == 0
    assert backend.stop_count == 1


def test_inference_stall_depletes_queue_without_replay(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend

    class SlowRefill:
        def __init__(self) -> None:
            self.calls: int = 0

        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            self.calls += 1
            if self.calls > 1:
                time.sleep(0.4)
            return request

    with pytest.raises(RuntimeError, match="queue exhausted"):
        deploy(robot, SlowRefill()).run()
    assert backend.write_count == 4
    assert backend.stop_count == 1


def test_expired_first_predictions_never_start_motion(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend

    class Slow:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            time.sleep(0.04)
            return request

    with pytest.raises(TimeoutError, match="first observation/action"):
        deploy(robot, Slow(), max_prediction_age_s=0.01, startup_timeout_s=0.15).run()
    assert backend.write_count == 0


def test_queued_commands_expire_while_waiting(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    with pytest.raises(TimeoutError, match="expired before execution"):
        deploy(robot, max_prediction_age_s=0.08).run()
    assert 0 < backend.write_count < 4
    assert backend.stop_count == 1


def test_resampling_continuous_and_discrete_fields() -> None:
    schema: ActionSchema = {
        "q": FeatureSpec((1,), "float32"),
        "mode": FeatureSpec((), "int32"),
    }
    chunk: ActionChunk = ActionChunk(
        (
            {"q": np.array([0], dtype=np.float32), "mode": np.array(0, dtype=np.int32)},
            {"q": np.array([2], dtype=np.float32), "mode": np.array(1, dtype=np.int32)},
        ),
        0.1,
    )
    values: tuple[Action, ...] = _resample(
        chunk, schema, options(control_hz=20, interpolate_keys={"q"})
    )
    np.testing.assert_allclose([a["q"][0] for a in values], [0, 1, 2, 2])
    np.testing.assert_array_equal([a["mode"] for a in values], [0, 0, 1, 1])
    chunk.actions[0]["q"][:] = 100
    assert values[0]["q"][0] == 0


def test_resampling_caps_prefix_and_validates_discarded_tail() -> None:
    schema: ActionSchema = {"q": FeatureSpec((1,), "float32")}
    good: Action = {"q": np.zeros(1, dtype=np.float32)}
    cfg: DeploymentOptions = options(max_queued_actions=3, queue_low_watermark=1)
    assert len(_resample(ActionChunk((good,) * 100, 0.1), schema, cfg)) == 3
    with pytest.raises(ValueError, match="NaN"):
        _resample(
            ActionChunk(
                (good,) * 100 + ({"q": np.array([np.nan], dtype=np.float32)},), 0.1
            ),
            schema,
            cfg,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control_hz": float("nan")},
        {"control_hz": 0},
        {"max_sample_age_s": -1},
        {"startup_timeout_s": float("inf")},
        {"queue_low_watermark": 8},
        {"shutdown_timeout_s": 0},
        {"max_queued_actions": True},
    ],
)
def test_invalid_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        options(**kwargs)


def test_slow_write_stops_instead_of_bursting_missed_ticks(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    original: Callable[[Action], None] = backend.write

    def slow_write(action: Action) -> None:
        original(action)
        time.sleep(0.12)

    monkeypatch.setattr(backend, "write", slow_write)
    with pytest.raises(TimeoutError, match="missed a complete control period"):
        deploy(robot).run(max_steps=3)
    assert backend.write_count == 1
    assert backend.stop_count == 1


def test_stop_serializes_with_send_and_hardware_stays_on_actor(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    entered, release = threading.Event(), threading.Event()
    stopped = threading.Event()
    threads: list[int] = []
    errors: list[BaseException] = []
    original_write, original_stop = backend.write, backend.stop

    def blocking_write(action: Action) -> None:
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        original_write(action)

    def record_stop() -> None:
        threads.append(threading.get_ident())
        original_stop()

    monkeypatch.setattr(backend, "write", blocking_write)
    monkeypatch.setattr(backend, "stop", record_stop)
    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(robot)

    def run() -> None:
        try:
            loop.run()
        except BaseException as error:  # noqa: BLE001 - forward the actor's failure to pytest
            errors.append(error)

    actor: threading.Thread = threading.Thread(target=run)
    actor.start()
    assert entered.wait(2)

    def request_stop() -> None:
        loop.stop()
        stopped.set()

    stopper: threading.Thread = threading.Thread(target=request_stop)
    stopper.start()
    assert not stopped.wait(0.01)
    release.set()
    stopper.join(2)
    actor.join(2)
    assert stopped.is_set() and not actor.is_alive()
    assert not errors
    assert backend.write_count == 1
    assert set(threads) == {actor.ident}


def test_shutdown_timeout_stops_hardware_before_reporting_blocked_policy(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
) -> None:
    robot, backend = robot_and_backend
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    class Blocked:
        def predict(self, request: NDArray[np.float32]) -> NDArray[np.float32]:
            entered.set()
            try:
                release.wait(2)
                return request
            finally:
                returned.set()

    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(
        robot, Blocked(), shutdown_timeout_s=0.03
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            loop.run()
        except BaseException as error:  # noqa: BLE001 - forward the actor's failure to pytest
            errors.append(error)

    actor: threading.Thread = threading.Thread(target=run)
    actor.start()
    try:
        assert entered.wait(2)
        loop.stop()
        actor.join(1)
        assert not actor.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
        assert "policy is still running" in str(errors[0])
        assert backend.stop_count == 1
        assert backend.write_count == 0
    finally:
        release.set()
        assert returned.wait(2)


def test_stop_before_run_still_reports_cleanup_failure(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, _ = robot_and_backend
    loop: RobotDeployment[NDArray[np.float32], NDArray[np.float32]] = deploy(robot)
    loop.stop()

    def bad_close() -> None:
        raise OSError("cleanup failed")

    monkeypatch.setattr(robot, "close", bad_close)
    with pytest.raises(OSError, match="cleanup failed"):
        loop.run()
