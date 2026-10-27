"""Time local pi0.5 prediction on a saved observation, without ROS or robot RPC."""

from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
from omegaconf import MISSING

from .pi05 import Pi05Policy
from .configuration import parse_config, repository_path
from .serve_pi05 import Pi05ServerConfig
from .processor_loader import build_processing


@dataclass
class BenchmarkConfig(Pi05ServerConfig):
    observation: str = MISSING
    output: str = MISSING
    warmup: int = 3
    iterations: int = 20

    def validate(self) -> None:
        super().validate()
        if not repository_path(self.observation).is_file():
            raise ValueError("observation must name a saved NPZ file")
        if self.warmup < 1 or self.iterations < 1:
            raise ValueError("warmup and iterations must be positive")
        if repository_path(self.output).exists():
            raise ValueError("output already exists; use a new report filename")


def read_observation(path: Path, cameras: list[str], action_dim: int) -> dict:
    """Read raw RGB images and state; no live-sensor fallback or object unpickling."""
    with np.load(path, allow_pickle=False) as data:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (action_dim,) or not np.isfinite(state).all():
            raise ValueError("saved state has the wrong width or non-finite values")
        result = {"state": state.copy()}
        for camera in cameras:
            image = data[camera]
            if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
                or min(image.shape[:2]) < 1
            ):
                raise ValueError(f"{camera} must be nonempty RGB uint8 HWC")
            result["image." + camera] = image.copy()
        return result


def measure(policy, inputs: Mapping, *, warmup: int, iterations: int) -> dict:
    """Measure synchronous predict, including CPU processing and GPU-to-CPU output."""
    if warmup < 1 or iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    policy.reset_run()
    samples = []
    shape = None
    for _ in range(warmup + iterations):
        started = time.perf_counter()
        actions = policy.predict(inputs)["actions"]
        elapsed_ms = (time.perf_counter() - started) * 1000
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError("policy returned invalid actions")
        if shape is not None and list(actions.shape) != shape:
            raise ValueError("policy action shape changed during benchmark")
        shape = list(actions.shape)
        samples.append(elapsed_ms)
    measured = samples[warmup:]
    return {
        "measurement": "local predict: preprocessing + inference + postprocessing; no ROS or RPC",
        "warmup_ms": samples[:warmup],
        "prediction_ms": measured,
        "median_ms": float(np.median(measured)),
        "p95_ms": float(np.percentile(measured, 95)),
        "min_ms": min(measured),
        "max_ms": max(measured),
        "action_shape": shape,
    }


def main(argv=None) -> None:
    config = parse_config(
        BenchmarkConfig,
        default_config=Path(__file__).with_name("configs") / "pi05_arx.yaml",
        argv=argv,
        validate=BenchmarkConfig.validate,
    )
    if not config.processor.camera_names or not config.processor.action_dim:
        raise ValueError(
            "offline benchmark requires explicit processor camera_names and action_dim"
        )
    inputs = read_observation(
        repository_path(config.observation),
        config.processor.camera_names,
        config.processor.action_dim,
    )
    processing = build_processing(
        repository_path(config.policy.checkpoint),
        task=config.task,
        config=config.processor,
    )
    policy = Pi05Policy(config.policy, processing)
    try:
        result = measure(
            policy, inputs, warmup=config.warmup, iterations=config.iterations
        )
        result.update(
            task=config.task,
            camera_names=processing.camera_names,
            num_inference_steps=policy.num_inference_steps,
            seed=config.policy.seed,
            use_cuda_graph=config.policy.use_cuda_graph,
        )
        output = repository_path(config.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(
            f"Median {result['median_ms']:.2f} ms; p95 {result['p95_ms']:.2f} ms; report: {output}"
        )
    finally:
        policy.close()


if __name__ == "__main__":
    main()
