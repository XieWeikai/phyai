"""Authenticated loopback control service for a resident model and robot sessions.

Starting this service opens no ROS connections and loads no weights. Only an
explicit prepare/start request constructs robot transports.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from omegaconf import OmegaConf
from phyai_robot.remote import RemotePolicy, run_remote

from .arx import ArxClientConfig
from .client import validate_client
from .robots.arx_control import ArxControl


class ControlError(RuntimeError):
    def __init__(self, message, code=409):
        super().__init__(message)
        self.code = code


class Runtime:
    def __init__(self, config, *, robot_control=None, runner=run_remote):
        self.config = config
        self.lock = threading.RLock()
        self.operations = threading.Lock()
        if config.get("robot_kind") != "arx":
            raise ValueError("An explicit supported robot_kind is required")
        client = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ArxClientConfig),
                OmegaConf.load(config["client_config"]),
            )
        )
        validate_client(client)
        self.client = client
        self.robot_control = robot_control or ArxControl(client, config["preparation"])
        self.runner = runner
        self.process = self.model_log = self.worker = self.deployment = None
        self.checkpoint = self.prompt = self.error = ""
        self.phase = "idle"
        self.cancel = threading.Event()
        self.receipts = {}

    def status(self):
        with self.lock:
            model_ready = (
                self.process is not None
                and self.process.poll() is None
                and bool(self.checkpoint)
            )
            active = self.worker is not None and self.worker.is_alive()
            cleanup_error = getattr(self.robot_control, "cleanup_error", "")
            return {
                "ok": True,
                "phase": self.phase,
                "model_ready": model_ready,
                "checkpoint_path": self.checkpoint if model_ready else None,
                "prompt": self.prompt,
                "running": active and self.phase in {"starting", "running", "stopping"},
                "busy": self.phase == "loading" or (active and self.phase != "running"),
                "arm_operation": "prepare"
                if active and self.phase == "preparing"
                else None,
                "error": cleanup_error or self.error,
                "recovery_required": bool(cleanup_error),
                "quiescent": not active and not cleanup_error,
            }

    def remote(self, timeout=None):
        return RemotePolicy(
            self.client.remote.endpoint,
            timeout_s=timeout or self.client.remote.timeout_s,
            warmup_timeout_s=self.client.remote.warmup_timeout_s,
        )

    def info(self):
        remote = self.remote(0.5)
        try:
            return remote.info()
        finally:
            remote.close()

    def load(self, checkpoint):
        path = Path(checkpoint).expanduser().resolve()
        roots = [Path(root).resolve() for root in self.config["checkpoint_roots"]]
        if not any(path.is_relative_to(root) for root in roots):
            raise ControlError("请选择已配置目录中的 PhyAI 权重。", 400)
        required = [
            "config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "tokenizer.json",
        ]
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise ControlError(
                "这不是完整的 PhyAI 导出权重，缺少：" + ", ".join(missing), 400
            )
        with self.operations:
            if self.status()["busy"] or self.status()["running"]:
                raise ControlError("请先结束当前推理或机械臂操作。")
            if self.status()["model_ready"]:
                if path != Path(self.checkpoint):
                    raise ControlError("请先关闭当前策略，再更换权重。")
                return self.status()
            try:
                self.info()
            except Exception:
                pass
            else:
                raise ControlError("PhyAI 模型端口已由其他服务占用，未启动重复进程。")
            with self.lock:
                self.phase, self.error = "loading", ""
            log_path = Path(self.config["model_log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.model_log:
                self.model_log.close()
            self.model_log = log_path.open("ab")
            command = [
                self.config["model_python"],
                "-m",
                "examples.deployment.serve_pi05",
                "--config",
                self.config["model_config"],
                f"policy.checkpoint={json.dumps(str(path))}",
                "task=Awaiting task.",
                f"endpoint={self.client.remote.endpoint}",
            ]
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=self.config["repository"],
                    stdout=self.model_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": ""},
                )
                deadline = time.monotonic() + float(
                    self.config.get("model_start_timeout_s", 150)
                )
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        failure = log_path.read_text(errors="replace")
                        failure = failure.rsplit(
                            "Traceback (most recent call last):", 1
                        )[-1]
                        failure = failure.split("\nnanobind:", 1)[0]
                        raise RuntimeError("模型进程退出：" + failure[-1500:])
                    try:
                        spec = self.info()
                    except Exception:
                        time.sleep(0.25)
                        continue
                    if Path(spec.metadata.get("checkpoint", "")).resolve() != path:
                        raise RuntimeError("模型服务报告的权重与所选目录不同")
                    with self.lock:
                        self.checkpoint, self.phase = str(path), "ready"
                    return self.status()
                raise TimeoutError("等待 PhyAI 模型启动超时")
            except Exception as error:
                self._terminate_model()
                with self.lock:
                    self.phase, self.error = "error", str(error)
                raise ControlError(str(error), 503) from error

    def _terminate_model(self):
        process = self.process
        if process is not None and process.poll() is None:
            if os.getpgid(process.pid) != process.pid:
                raise ControlError("模型进程组不属于本服务，未执行关闭。")
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        self.process = None
        self.checkpoint = ""
        if self.model_log:
            self.model_log.close()
            self.model_log = None

    def _launch(self, phase, function):
        with self.lock:
            if getattr(self.robot_control, "cleanup_error", ""):
                raise ControlError("机械臂停止确认失败，请先由现场人员检查执行服务。")
            if self.worker and self.worker.is_alive():
                raise ControlError("已有推理或机械臂操作正在执行。")
            self.cancel = threading.Event()
            self.phase, self.error, self.deployment = phase, "", None

            def work():
                try:
                    function()
                except Exception as error:
                    with self.lock:
                        self.error = str(error)
                finally:
                    with self.lock:
                        self.phase = (
                            "error"
                            if self.error
                            else "ready"
                            if self.checkpoint
                            else "idle"
                        )
                        self.deployment = None

            self.worker = threading.Thread(
                target=work, name="phyai-workbench-run", daemon=True
            )
            self.worker.start()

    def start(self, body):
        with self.operations:
            task = str(body.get("prompt", "")).strip()
            request_id = body.get("request_id")
            if (
                not task
                or len(task) > 4000
                or not isinstance(request_id, str)
                or not 8 <= len(request_id) <= 128
            ):
                raise ControlError("需要有效的任务指令和 request_id。", 400)
            signature = (str(Path(body.get("checkpoint_path", "")).resolve()), task)
            if request_id in self.receipts:
                if self.receipts[request_id] != signature:
                    raise ControlError("request_id 已用于不同操作。")
                return self.status()
            if self.status()["busy"] or self.status()["running"]:
                raise ControlError("已有推理或机械臂操作正在执行。")
            if not self.status()["model_ready"] or signature[0] != self.checkpoint:
                raise ControlError("请先加载所选 PhyAI 权重。")
            self.prompt = task

            def run():
                def ready(deployment):
                    with self.lock:
                        self.deployment = deployment
                    if self.cancel.is_set():
                        deployment.stop()

                def report(message):
                    if message.startswith("Robot commands enabled"):
                        with self.lock:
                            self.phase = "running"

                self.runner(
                    self.remote(),
                    build=lambda spec: self.robot_control.build_run(spec, self.cancel),
                    scheduler=self.client.schedule.create(),
                    options=self.client.control.to_options(),
                    run_options={"task": task},
                    on_deployment=ready,
                    report=report,
                )

            self._launch("starting", run)
            self.receipts[request_id] = signature
            if len(self.receipts) > 256:
                self.receipts.pop(next(iter(self.receipts)))
            return self.status()

    def prepare(self):
        with self.operations:
            if self.status()["busy"] or self.status()["running"]:
                raise ControlError("请先结束当前推理或操作。")
            self._launch("preparing", lambda: self.robot_control.prepare(self.cancel))
            return self.status()

    def stop(self):
        with self.operations:
            with self.lock:
                self.cancel.set()
                worker, deployment = self.worker, self.deployment
                if worker and worker.is_alive():
                    self.phase = "stopping"
            if deployment is not None:
                deployment.stop()
            if worker:
                worker.join(timeout=35)
                if worker.is_alive():
                    raise ControlError(
                        "已请求停止，仍在等待执行端清理；暂不交接控制权。", 504
                    )
            state = self.status()
            if state["recovery_required"]:
                raise ControlError("机械臂停止确认失败，暂不交接控制权。", 503)
            return state

    def unload(self):
        self.stop()
        with self.operations:
            self._terminate_model()
            with self.lock:
                self.phase = "idle"
                self.error = ""
            return self.status()


def serve(runtime, token, host="127.0.0.1", port=10868):
    if host not in {"127.0.0.1", "::1"} or len(token) < 24:
        raise ValueError("Workbench requires loopback binding and an access token")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            return hmac.compare_digest(
                self.headers.get("Authorization", "").encode(),
                ("Bearer " + token).encode(),
            )

        def do_GET(self):
            if not self.authorized():
                return self.reply(401, {"error": "Unauthorized"})
            if self.path != "/state":
                return self.reply(404, {"error": "Not found"})
            self.reply(200, runtime.status())

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {"error": "Unauthorized"})
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 16384:
                    raise ControlError("Invalid request size", 400)
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ControlError("Request must be an object", 400)
                actions = {
                    "/model/load": lambda: runtime.load(body["checkpoint_path"]),
                    "/model/unload": runtime.unload,
                    "/run/start": lambda: runtime.start(body),
                    "/run/stop": runtime.stop,
                    "/robot/prepare": runtime.prepare,
                }
                if self.path not in actions:
                    raise ControlError("Not found", 404)
                self.reply(200, actions[self.path]())
            except ControlError as error:
                self.reply(error.code, {"ok": False, "error": str(error)})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"ok": False, "error": str(error)})
            except Exception as error:
                self.reply(503, {"ok": False, "error": str(error)})

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    runtime = Runtime(config)
    server = serve(runtime, config["token"], port=int(config.get("port", 10868)))
    try:
        server.serve_forever()
    finally:
        runtime.unload()
        server.server_close()


if __name__ == "__main__":
    main()
