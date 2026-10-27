"""ARX raw units, lazy construction and controller handoff, using fake messages."""

from io import BytesIO
from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np
import pytest
from PIL import Image
from phyai_robot import Sample, Observation, InferenceContext
from phyai_robot.remote import PolicySpec

from examples.deployment import arx as entry
from examples.deployment.client import validate_client
from examples.deployment.configuration import parse_config
from examples.deployment.robots import arx
from examples.deployment.robots.arx_codecs import (
    decode_arm,
    encode_arm,
    decode_compressed_rgb,
)


def test_robot_status_round_trip_preserves_raw_gripper_and_negative_joints():
    original = [-0.2, 0.948, 0.858, -0.573, 0.01, -0.04, -4.78]
    fields = decode_arm(SimpleNamespace(joint_pos=original), "left")
    encoded = encode_arm(fields, "left", SimpleNamespace)
    assert encoded.joint_pos == original
    assert encoded.joint_vel == encoded.joint_cur == [0.0] * 7
    assert arx.ACTION_SCHEMA["gripper_left"].unit == "rad"
    original[0] = 2.0
    assert fields["joint_position_left"][0] == -0.2


@pytest.mark.parametrize(
    "values", [[0.0] * 6, [0.0] * 8, [float("nan")] * 7, [float("inf")] * 7]
)
def test_bad_feedback_is_not_replaced_by_zeros(values):
    with pytest.raises(ValueError, match="seven finite"):
        decode_arm(SimpleNamespace(joint_pos=values), "left")


def test_command_rejects_wrong_shape_and_nonfinite_values():
    for joints, grip in [
        ([0.0] * 5, [0.0]),
        ([0.0] * 6, [0.0, 1.0]),
        ([float("inf")] * 6, [0.0]),
    ]:
        with pytest.raises(ValueError):
            encode_arm(
                {"joint_position_right": joints, "gripper_right": grip},
                "right",
                SimpleNamespace,
            )


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_compressed_rgb_color_order(format):
    image = np.full((8, 10, 3), [250, 5, 20], dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format=format)
    decoded = decode_compressed_rgb(SimpleNamespace(data=buffer.getvalue()))
    assert decoded.shape == image.shape and decoded.dtype == np.uint8
    assert decoded[0, 0, 0] > 240 and decoded[0, 0, 2] < 30


def test_depth_and_corrupt_images_are_rejected():
    buffer = BytesIO()
    Image.fromarray(np.ones((8, 10), np.uint16)).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="color image"):
        decode_compressed_rgb(SimpleNamespace(data=buffer.getvalue()))
    with pytest.raises(Exception):
        decode_compressed_rgb(SimpleNamespace(data=b"not an image"))


def test_adapter_orders_both_arms_without_gripper_normalization_or_delta_addition():
    config = arx.ArxConfig()
    schema = arx.observation_schema(config, 4)
    adapter = arx.make_adapter(config, schema)
    now = time.monotonic_ns()
    samples = {
        key: Sample(np.full(spec.shape, i, dtype=spec.dtype), now)
        for i, (key, spec) in enumerate(schema.items())
    }
    samples["gripper_left"] = Sample(np.array([-4.78]), now)
    samples["gripper_right"] = Sample(np.array([-0.08]), now)
    observation = Observation(samples)
    request = adapter.to_request(observation, InferenceContext())
    assert set(request) == {
        "state",
        "image.base",
        "image.left_wrist",
        "image.right_wrist",
    }
    assert request["state"].shape == (14,)
    np.testing.assert_allclose(request["state"][[6, 13]], [-4.78, -0.08])
    row = np.linspace(-0.6, 0.6, 14, dtype=np.float32)
    row[[6, 13]] = [-2.8, -0.08]
    action = adapter.to_actions({"actions": row[None]}, observation).actions[0]
    np.testing.assert_allclose(action["joint_position_right"], row[7:13])
    np.testing.assert_allclose(action["gripper_left"], [-2.8])
    assert action["gripper_left"].dtype == np.float64


def test_bundled_config_and_overrides_are_checked_without_ros():
    config = parse_config(
        entry.ArxClientConfig,
        default_config=entry.DEFAULT_CONFIG,
        argv=["robot.controller_interface=workbench"],
        validate=validate_client,
    )
    assert config.schedule.action_hz == 30 and config.schedule.execution_horizon == 25
    assert config.robot.controller_interface == "workbench"
    assert config.control.control_hz == 200


@pytest.mark.parametrize(
    "changes",
    [
        {"camera_transport": "h264"},
        {"controller_interface": "guess"},
        {"state_topics": {"left": "/a"}},
        {"controller_nodes": {"left": "/same", "right": "/same"}},
        {"model_cameras": {"base": "missing"}},
        {"camera_topics": {"joint_position_left": "/image"}},
        {"io_timeout_s": float("nan")},
        {"service_timeout_s": -1},
        {"node_name": "3invalid"},
        {"command_topics": {"left": "/arm_slave_l_status", "right": "/other"}},
    ],
)
def test_invalid_robot_config_fails_before_connections(changes):
    with pytest.raises(ValueError):
        replace(arx.ArxConfig(), **changes).validate()


@pytest.mark.parametrize("transport", ["raw", "compressed"])
def test_factory_constructs_lazy_stamped_backends_without_service_calls(
    ros_messages, transport
):
    config = arx.ArxConfig(camera_transport=transport)
    embodiment = arx.build(config, PolicySpec({}, {}, {"image_size": 4}))
    assert embodiment.activate is None
    assert len(embodiment.robot.backends) == 4
    joints = embodiment.robot.backends[-1]
    assert joints._node is None
    assert all(binding.stamp for binding in joints._actions.values())
    assert all(
        binding.message_type is ros_messages["RobotStatus"]
        for binding in joints._actions.values()
    )


def test_workbench_warmup_failure_does_not_touch_controller(ros_messages):
    embodiment = arx.build(
        arx.ArxConfig(controller_interface="workbench"),
        PolicySpec({}, {}, {"image_size": 4}),
    )
    joints = embodiment.robot.backends[-1]
    joints.call_service = lambda *args, **kwargs: pytest.fail(
        "no services before activation"
    )
    joints._on_stop(joints)


def test_workbench_activation_and_stop_never_enable_home_or_switch_to_capture(
    ros_messages,
):
    embodiment = arx.build(
        arx.ArxConfig(controller_interface="workbench"),
        PolicySpec({}, {}, {"image_size": 4}),
    )
    joints = embodiment.robot.backends[-1]
    calls = []
    joints.call_service = lambda kind, name, request, **kw: calls.append(name)
    embodiment.activate()
    assert calls == [
        f"/arm_slave_{side}/{action}"
        for action in ["health_check", "select_inference", "remember_fold"]
        for side in ["l", "r"]
    ]
    joints._on_stop(joints)
    assert calls[-2:] == [
        "/arm_slave_l/select_inference",
        "/arm_slave_r/select_inference",
    ]
    before = len(calls)
    joints._on_stop(joints)
    assert len(calls) == before


def test_partial_activation_and_stop_failure_still_attempt_both_sides(ros_messages):
    embodiment = arx.build(
        arx.ArxConfig(controller_interface="workbench"),
        PolicySpec({}, {}, {"image_size": 4}),
    )
    joints = embodiment.robot.backends[-1]
    calls = []
    fail = {"name": "/arm_slave_r/remember_fold"}

    def call(kind, name, request, **kw):
        calls.append(name)
        if name == fail["name"]:
            raise RuntimeError("service failed")

    joints.call_service = call
    with pytest.raises(RuntimeError):
        embodiment.activate()
    fail["name"] = "/arm_slave_l/select_inference"
    with pytest.raises(ExceptionGroup):
        joints._on_stop(joints)
    assert calls[-2:] == [
        "/arm_slave_l/select_inference",
        "/arm_slave_r/select_inference",
    ]
