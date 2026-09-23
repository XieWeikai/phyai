"""A bounded producer/actor loop for single-step robot control.

The actor owns all Robot I/O. One background producer owns preprocessing,
prediction, and postprocessing. New chunks append to the existing queue; this
is intentionally not RTC or a hard real-time scheduler.
"""

from __future__ import annotations

import math
import time
import threading
from typing import Any
from _thread import LockType
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .robot import Robot
from .types import Action, ActionChunk, Observation, ActionSchema, ObservationNotReady
from .policy import Policy, PolicyAdapter
from .validation import (
    freeze_values,
    freeze_samples,
    validate_values,
    validate_observation,
)


@dataclass(frozen=True)
class DeploymentOptions:
    control_hz: float  # Requested command frequency, independent of the model rate.
    max_sample_age_s: float  # Maximum local receive age of every required sensor.
    startup_timeout_s: float = 5.0  # Budget after connect for first sensors/actions.
    queue_low_watermark: int = 10  # Start prediction when this many commands remain.
    max_queued_actions: int = 50  # Hard limit on retained commands, including refill.
    max_prediction_age_s: float = 0.5  # Snapshot-to-send limit, including queue time.
    shutdown_timeout_s: float = 2.0  # Maximum wait for a still-running prediction.
    interpolate_keys: frozenset[str] = frozenset()  # Explicit continuous target fields.

    def __post_init__(self) -> None:
        for name in (
            "control_hz",
            "max_sample_age_s",
            "startup_timeout_s",
            "max_prediction_age_s",
            "shutdown_timeout_s",
        ):
            value: float = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if 1e9 / self.control_hz < 1:
            raise ValueError("control period must be at least one nanosecond")
        if type(self.max_queued_actions) is not int or self.max_queued_actions < 1:
            raise ValueError("max_queued_actions must be a positive integer")
        if (
            type(self.queue_low_watermark) is not int
            or not 0 <= self.queue_low_watermark < self.max_queued_actions
        ):
            raise ValueError("queue_low_watermark must be below queue capacity")
        object.__setattr__(self, "interpolate_keys", frozenset(self.interpolate_keys))


@dataclass(frozen=True)
class _QueuedAction:
    action: Action  # Detached single-tick values, already structurally validated.
    observed_at_ns: int  # Request snapshot time; expiry includes time in the queue.


def _resample(
    chunk: ActionChunk,
    schema: ActionSchema,
    options: DeploymentOptions,
) -> tuple[Action, ...]:
    """Resample a source interval [0, N * dt) and hold its final target.

    Return at most the queue capacity, retaining the earliest prefix. Validate
    every source target, even if the capacity later truncates the sequence.
    Arrays are copied so model buffers cannot change queued commands.
    """
    if not isinstance(chunk, ActionChunk) or not chunk.actions:
        raise ValueError("adapter must return a nonempty ActionChunk")
    if not math.isfinite(chunk.step_period_s) or chunk.step_period_s <= 0:
        raise ValueError("chunk period must be finite and positive")
    for action in chunk.actions:
        validate_values(action, schema)
    ratio: float = chunk.step_period_s * options.control_hz
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("source/control period ratio cannot be represented")
    # Cap before ceil to avoid allocating an unbounded model horizon.
    count: int = math.ceil(min(len(chunk.actions) * ratio, options.max_queued_actions))
    result: list[Action] = []
    for tick in range(count):
        position: float = tick / ratio
        # Correct only rounding near an exact source boundary, not real delays.
        nearest: int = round(position)
        if abs(position - nearest) < 1e-10:
            position = float(nearest)
        lower: int = min(int(position), len(chunk.actions) - 1)
        upper: int = min(lower + 1, len(chunk.actions) - 1)
        weight: float = min(position - lower, 1.0)
        values: dict[str, NDArray[Any]] = {}
        for key, spec in schema.items():
            first: NDArray[Any] = chunk.actions[lower][key]
            if key in options.interpolate_keys and upper != lower:
                second: NDArray[Any] = chunk.actions[upper][key]
                # Wider arithmetic avoids overflow when subtracting float32 targets.
                values[key] = np.asarray(
                    (1 - weight) * first.astype(np.float64)
                    + weight * second.astype(np.float64),
                    dtype=spec.dtype,
                )
            else:
                values[key] = first
        validate_values(values, schema)
        result.append(freeze_values(values))
    return tuple(result)


class RobotDeployment[RequestT, ResultT]:
    """Run a robot with one in-flight prediction and a bounded append-only queue.

    stop() requests termination; it never invokes hardware from another thread.
    It serializes with an in-progress send, so no new send starts after stop()
    returns. run() performs the actual stop/close before returning. Backend I/O
    must have finite timeouts. A blocked Policy cannot be forcibly cancelled;
    shutdown reports a timeout if the producer cannot finish within its budget.
    """

    def __init__(
        self,
        *,
        robot: Robot,  # Actor-owned hardware boundary.
        policy: Policy[RequestT, ResultT],  # Producer-owned synchronous predictor.
        adapter: PolicyAdapter[
            RequestT, ResultT
        ],  # Conversion for this schema/checkpoint pair.
        options: DeploymentOptions,  # Fixed timing and queue limits for this run.
    ) -> None:
        if not options.interpolate_keys <= set(robot.action_schema):
            raise ValueError("interpolate_keys contains unknown command fields")
        for key in options.interpolate_keys:
            if np.dtype(robot.action_schema[key].dtype).kind != "f":
                raise ValueError(f"{key}: interpolation requires a floating dtype")
        self.robot: Robot = robot
        self.policy: Policy[RequestT, ResultT] = policy
        self.adapter: PolicyAdapter[RequestT, ResultT] = adapter
        self.options: DeploymentOptions = options
        self._condition: threading.Condition = (
            threading.Condition()
        )  # Protect queue, latest snapshot, and errors.
        self._send_lock: LockType = (
            threading.Lock()
        )  # Linearize user stop requests against sends.
        self._stop_event: threading.Event = (
            threading.Event()
        )  # Wakes actor sleep and cancels late results.
        self._queue: deque[_QueuedAction] = deque()
        self._latest: tuple[Observation, int, int] | None = (
            None  # Snapshot, time, sequence.
        )
        self._sequence: int = 0  # Prevent repeated predictions from the same snapshot.
        self._worker_error: BaseException | None = (
            None  # Forward producer failures to run.
        )
        self._started: bool = False  # A Deployment instance may run exactly once.

    def stop(self) -> None:
        """Request shutdown from any thread without concurrent hardware access."""
        with self._send_lock:
            self._stop_event.set()
        with self._condition:
            self._condition.notify_all()

    def _observe(self) -> None:
        raw: Observation = self.robot.get_observation()
        # Custom Robot implementations receive the same validation as CompositeRobot.
        observation: Observation = Observation(freeze_samples(raw.samples))
        validate_observation(observation, self.robot.observation_schema)
        now_ns: int = time.monotonic_ns()
        for key, sample in observation.samples.items():
            if now_ns - sample.received_at_ns > self.options.max_sample_age_s * 1e9:
                raise TimeoutError(f"stale observation: {key}")
        with self._condition:
            self._sequence += 1
            self._latest = observation, now_ns, self._sequence
            self._condition.notify_all()

    def _produce(self) -> None:
        last_sequence: int = -1
        observation: Observation
        observed_at_ns: int
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda previous=last_sequence: (
                            self._stop_event.is_set()
                            or (
                                self._latest is not None
                                and self._latest[2] != previous
                                and len(self._queue) <= self.options.queue_low_watermark
                            )
                        )
                    )
                    if self._stop_event.is_set():
                        return
                    # wait_for guarantees a snapshot unless shutdown was requested.
                    assert self._latest is not None
                    observation, observed_at_ns, last_sequence = self._latest
                # No shared lock is held during model execution or conversion.
                request: RequestT = self.adapter.to_request(observation)
                result: ResultT = self.policy.predict(request)
                if self._stop_event.is_set():
                    return
                chunk: ActionChunk = self.adapter.to_actions(result, observation)
                actions: tuple[Action, ...] = _resample(
                    chunk, self.robot.action_schema, self.options
                )
                with self._condition:
                    if self._stop_event.is_set():
                        return
                    if (
                        time.monotonic_ns() - observed_at_ns
                        > self.options.max_prediction_age_s * 1e9
                    ):
                        continue
                    free: int = self.options.max_queued_actions - len(self._queue)
                    # Existing actions retain their order. A new chunk starts after
                    # them, not necessarily at the next control tick. Drop its tail.
                    self._queue.extend(
                        _QueuedAction(a, observed_at_ns) for a in actions[:free]
                    )
                    self._condition.notify_all()
        except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()
            self.stop()

    def _raise_worker_error(self) -> None:
        with self._condition:
            if self._worker_error is not None:
                raise self._worker_error

    def run(self, *, max_steps: int | None = None) -> None:
        """Connect, run, and always stop/close; optionally finish after N sends.

        max_steps is useful for bounded examples and tests. It is not a control
        duration guarantee. A late actor never bursts through missed commands:
        missing a whole control period terminates the run instead.
        """
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        if self._started:
            raise RuntimeError("deployment instances are single-use")
        self._started = True
        worker: threading.Thread | None = None
        errors: list[BaseException] = []
        period_ns: int = round(1e9 / self.options.control_hz)
        try:
            if not self._stop_event.is_set():
                self.robot.connect()
                worker = threading.Thread(
                    target=self._produce, name="phyai-policy", daemon=True
                )
                worker.start()
            startup_deadline: float = time.monotonic() + self.options.startup_timeout_s
            # Keep receiving while the first prediction runs, but do not send
            # anything until a complete, timely chunk has entered the queue.
            while not self._stop_event.is_set():
                self._raise_worker_error()
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("timed out waiting for first observation/action")
                try:
                    self._observe()
                except ObservationNotReady:
                    pass
                with self._condition:
                    if self._queue:
                        break
                self._stop_event.wait(min(period_ns / 1e9, 0.01))

            next_deadline: int = time.monotonic_ns()
            sent: int = 0
            while not self._stop_event.is_set():
                delay_s: float = (next_deadline - time.monotonic_ns()) / 1e9
                if delay_s > 0 and self._stop_event.wait(delay_s):
                    break
                self._raise_worker_error()
                self._observe()
                with self._condition:
                    if not self._queue:
                        raise RuntimeError("action queue exhausted")
                    pending: _QueuedAction = self._queue.popleft()
                    self._condition.notify_all()
                with self._send_lock:
                    if self._stop_event.is_set():
                        break
                    now_ns: int = time.monotonic_ns()
                    if now_ns - next_deadline >= period_ns:
                        raise TimeoutError("actor missed a complete control period")
                    if (
                        now_ns - pending.observed_at_ns
                        > self.options.max_prediction_age_s * 1e9
                    ):
                        raise TimeoutError("queued prediction expired before execution")
                    self.robot.send_action(pending.action)
                sent += 1
                if max_steps is not None and sent >= max_steps:
                    break
                next_deadline += period_ns
            self._raise_worker_error()
        except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
            errors.append(error)
        finally:
            self._stop_event.set()
            with self._condition:
                self._condition.notify_all()
            # Hardware stops before waiting for a potentially blocked predictor.
            for cleanup in (self.robot.stop, self.robot.close):
                try:
                    cleanup()
                except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
                    errors.append(error)
            if worker is not None:
                worker.join(self.options.shutdown_timeout_s)
                if worker.is_alive():
                    errors.append(
                        TimeoutError(
                            "policy is still running after robot shutdown; "
                            "do not destroy its resources until predict() returns"
                        )
                    )
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("deployment and/or cleanup failed", errors)
