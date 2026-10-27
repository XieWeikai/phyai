"""Benchmark orchestration with a fake predictor, without importing Torch or ROS."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.deployment.benchmark_pi05 import read_observation, measure, main


def test_saved_observation_preserves_raw_state_and_rgb(tmp_path):
    path = tmp_path / "sample.npz"
    state = np.arange(14, dtype=np.float32)
    state[[6, 13]] = [-4.78, -0.08]
    image = np.zeros((8, 10, 3), np.uint8)
    np.savez(path, base=image, left_wrist=image, right_wrist=image, state=state)
    result = read_observation(path, ["base", "left_wrist", "right_wrist"], 14)
    np.testing.assert_array_equal(result["state"], state)
    assert result["image.base"].shape == (8, 10, 3)
    with pytest.raises(KeyError):
        read_observation(path, ["missing"], 14)
    with pytest.raises(ValueError, match="width"):
        read_observation(path, ["base"], 16)


def test_warmups_excluded_and_seed_reset_once():
    calls = []
    policy = SimpleNamespace(
        reset_run=lambda: calls.append("reset"),
        predict=lambda inputs: (
            calls.append("predict") or {"actions": np.zeros((50, 14), np.float32)}
        ),
    )
    result = measure(policy, {}, warmup=2, iterations=4)
    assert calls == ["reset"] + ["predict"] * 6
    assert len(result["warmup_ms"]) == 2 and len(result["prediction_ms"]) == 4
    assert result["action_shape"] == [50, 14]
    assert "no ROS or RPC" in result["measurement"]


def test_bad_outputs_are_not_reported_as_successful_timings():
    policy = SimpleNamespace(
        reset_run=lambda: None,
        predict=lambda inputs: {"actions": np.full((50, 14), np.nan)},
    )
    with pytest.raises(ValueError, match="invalid actions"):
        measure(policy, {}, warmup=1, iterations=1)


def test_print_config_does_not_import_engine_or_ros(capsys):
    before = set(sys.modules)
    with pytest.raises(SystemExit) as result:
        main(["--print-config"])
    assert result.value.code == 0
    assert "iterations: 20" in capsys.readouterr().out
    assert not {"torch", "phyai", "rclpy"} & (set(sys.modules) - before)
