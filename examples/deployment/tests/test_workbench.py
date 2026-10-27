"""Workbench lifecycle tests: fake robot operations, no ROS nodes or GPU loads."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from examples.deployment.workbench import Runtime, ControlError, serve


@pytest.fixture
def runtime(tmp_path):
    config = {"robot_kind": "arx", "client_config": str(tmp_path / "client.yaml")}
    (tmp_path / "client.yaml").write_text("{}")
    events = []
    completed = threading.Event()

    def runner(policy, **kwargs):
        events.append(kwargs["run_options"])
        deployment = SimpleNamespace(stop=lambda: completed.set())
        kwargs["on_deployment"](deployment)
        kwargs["report"]("Robot commands enabled; Ctrl-C stops")
        completed.wait(3)
        policy.close()

    control = SimpleNamespace(
        prepare=lambda cancel: cancel.wait(3), build_run=lambda *args: None
    )
    service = Runtime(config, robot_control=control, runner=runner)
    service.process = SimpleNamespace(poll=lambda: None)
    service.checkpoint = "/model/one"
    service.phase = "ready"
    service.remote = lambda *args: SimpleNamespace(close=lambda: None)
    yield service, events
    service.stop()


def wait_running(service):
    for _ in range(100):
        if service.status()["phase"] == "running":
            return
        time.sleep(0.001)
    pytest.fail("worker did not enter running phase")


def test_construct_and_status_do_not_touch_robot(runtime):
    service, events = runtime
    assert service.status()["quiescent"] and service.status()["model_ready"]
    assert events == [] and service.worker is None


def test_task_is_forwarded_once_and_stop_acknowledges_thread_exit(runtime):
    service, events = runtime
    body = {
        "checkpoint_path": "/model/one",
        "prompt": "Pour water.",
        "request_id": "request-123",
    }
    service.start(body)
    wait_running(service)
    service.start(body)
    assert events == [{"task": "Pour water."}]
    with pytest.raises(ControlError):
        service.start({**body, "prompt": "A different task"})
    with pytest.raises(ControlError):
        service.start({**body, "request_id": "request-456", "prompt": "Another task"})
    assert service.status()["prompt"] == "Pour water."
    with pytest.raises(ControlError):
        service.prepare()
    stopped = service.stop()
    assert stopped["quiescent"] and not stopped["running"]
    assert not service.worker.is_alive()
    service.start(body)  # A stale retry must not restart a stopped run.
    assert not service.status()["running"]


def test_prepare_is_explicit_cancellable_and_independent_of_model(runtime):
    service, events = runtime
    service.process = None
    service.checkpoint = ""
    result = service.prepare()
    assert result["arm_operation"] == "prepare"
    assert service.stop()["quiescent"]
    assert events == []


def test_raw_training_checkpoint_is_rejected_before_process_start(runtime, tmp_path):
    service, _ = runtime
    service.config["checkpoint_roots"] = [str(tmp_path)]
    with pytest.raises(ControlError, match="PhyAI 导出"):
        service.load(str(tmp_path))
    with pytest.raises(ControlError, match="已配置目录"):
        service.load("/outside")


def test_http_requires_token_before_any_operation(runtime):
    import urllib.request
    import urllib.error

    service, events = runtime
    token = "a" * 32
    server = serve(service, token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/state")
        assert error.value.code == 401
        request = urllib.request.Request(
            base + "/state", headers={"Authorization": "Bearer " + token}
        )
        assert json.load(urllib.request.urlopen(request))["quiescent"]
        request = urllib.request.Request(base + "/robot/prepare", data=b"{}")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 401 and events == [] and service.worker is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_cleanup_failure_does_not_acknowledge_handoff(runtime):
    service, _ = runtime
    service.robot_control.cleanup_error = "controller did not acknowledge stop"
    assert service.status()["recovery_required"]
    assert not service.status()["quiescent"]
    with pytest.raises(ControlError, match="交接"):
        service.stop()
    with pytest.raises(ControlError, match="现场人员"):
        service.prepare()
    service.robot_control.cleanup_error = ""
