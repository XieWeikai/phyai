"""DM0.5 resident policy using EnginePolicy and the existing RPC contract."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING
from phyai_robot import EnginePolicy, FeatureSpec
from phyai_robot.adapters import ACTIONS_KEY, IMAGE_PREFIX, STATE_KEY
from phyai_robot.remote import PolicySpec

from examples.deployment.configuration import repository_path


@dataclass
class DM05PolicyConfig:
    checkpoint: str = MISSING
    seed: int = 42
    num_inference_steps: int = 10
    num_threads: int = 8

    def validate(self):
        if not repository_path(self.checkpoint).is_dir():
            raise ValueError("policy.checkpoint must be an existing OpenDM checkpoint")
        if (
            self.num_inference_steps < 1
            or self.num_threads < 1
            or not 0 <= self.seed < 2**64
        ):
            raise ValueError("Invalid policy steps, threads or seed")


@dataclass
class DM05ProcessorConfig:
    norm_stats: str = MISSING
    camera_names: list[str] = field(
        default_factory=lambda: ["head_left", "left_wrist", "right_wrist"]
    )
    image_prompts: list[str] = field(
        default_factory=lambda: ["Head left", "Left wrist", "Right wrist"]
    )
    robot_type: str = "Marvin Pro"
    control_mode: str = "joint"
    speed: str = "0.5"
    delta_mask: list[bool] = field(
        default_factory=lambda: [True] * 7 + [False] + [True] * 7 + [False]
    )
    wire_image_size: int = 960
    history_camera: str = "head_left"
    history_slots: int = 32
    history_interval_s: float = 1.0
    history_tolerance_s: float = 0.25

    def validate(self):
        if not repository_path(self.norm_stats).is_file():
            raise ValueError("processor.norm_stats must name the SFT norm_stats.json")
        if (
            len(self.camera_names) != len(self.image_prompts)
            or not self.camera_names
            or len(set(self.camera_names)) != len(self.camera_names)
        ):
            raise ValueError("Camera names and prompts must match in training order")
        if (
            self.history_camera not in self.camera_names
            or not 0 <= self.history_slots <= 32
        ):
            raise ValueError("Invalid history camera or history slot count")
        if (
            self.history_interval_s <= 0
            or not 0 <= self.history_tolerance_s < self.history_interval_s / 2
        ):
            raise ValueError("Invalid history interval or tolerance")
        if self.wire_image_size < 1 or not 1 <= len(self.delta_mask) <= 32:
            raise ValueError("Invalid image or state/action dimensions")
        if self.control_mode != "joint":
            raise ValueError("This deployment example supports joint-space checkpoints")


class DM05History:
    """Sample only previously received frames; never manufacture missing history."""

    def __init__(self, slots, interval_s, tolerance_s):
        self.slots, self.interval_s, self.tolerance_s = slots, interval_s, tolerance_s
        self.frames = deque()

    def reset(self):
        self.frames.clear()

    def select(self, now):
        selected = []
        for slot in range(self.slots, 0, -1):
            target = now - slot * self.interval_s
            if not self.frames:
                break
            timestamp, frame = min(self.frames, key=lambda item: abs(item[0] - target))
            if timestamp < now and abs(timestamp - target) <= self.tolerance_s:
                selected.append(frame)
        return selected

    def append(self, now, image):
        if not self.slots:
            return
        if not self.frames or now - self.frames[-1][0] >= 0.1:
            self.frames.append((now, np.array(image, copy=True)))
        oldest = now - self.slots * self.interval_s - self.tolerance_s
        while self.frames and self.frames[0][0] < oldest:
            self.frames.popleft()


class DM05Policy:
    def __init__(
        self, config: DM05PolicyConfig, processor_config: DM05ProcessorConfig, *, task
    ):
        import torch
        from phyai import Engine, EngineArgs
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.models.dm05.configuration_dm05 import DM05Config
        from phyai.models.dm05.main_dm05 import DM05Args
        from phyai.models.dm05.scheduler_dm05 import DM05Request
        from phyai_utils_tools.models.dm05 import DM05Processor
        from phyai_utils_tools.models.dm05.processor_dm05 import square_image

        config.validate()
        processor_config.validate()
        checkpoint = repository_path(config.checkpoint)
        model_config = DM05Config.from_checkpoint(checkpoint)
        self.config, self.processor_config, self.task = config, processor_config, task
        self.torch, self.request_type = torch, DM05Request
        self.history_image = lambda image: np.asarray(
            square_image(image, model_config.vision.image_size)
        )
        self.horizon, self.padded_dim = model_config.chunk_size, model_config.action_dim
        self.action_dim = len(processor_config.delta_mask)
        if self.action_dim > self.padded_dim:
            raise ValueError("Processor action dimensions exceed the model output")
        self.processor = DM05Processor(
            checkpoint,
            norm_stats=repository_path(processor_config.norm_stats),
            image_prompts=processor_config.image_prompts,
            delta_mask=processor_config.delta_mask,
            robot_type=processor_config.robot_type,
            control_mode=processor_config.control_mode,
            speed=processor_config.speed,
            history_slots=processor_config.history_slots,
            image_size=model_config.vision.image_size,
            max_length=model_config.text.sliding_window - 1,
        )
        self.history = DM05History(
            processor_config.history_slots,
            processor_config.history_interval_s,
            processor_config.history_tolerance_s,
        )
        self.generator = torch.Generator().manual_seed(config.seed)
        self.engine = Engine(
            EngineArgs(
                plugin="dm05",
                plugin_args=DM05Args(checkpoint_dir=checkpoint),
                config=EngineConfig(
                    device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                    runtime=RuntimeConfig(
                        use_cuda_graph=False, num_threads=config.num_threads
                    ),
                ),
            )
        )
        self.policy = EnginePolicy(self.engine)

    def spec(self):
        c = self.processor_config
        return PolicySpec(
            inputs={
                **{
                    IMAGE_PREFIX + name: FeatureSpec(
                        (c.wire_image_size, c.wire_image_size, 3), "uint8"
                    )
                    for name in c.camera_names
                },
                STATE_KEY: FeatureSpec((self.action_dim,), "float32"),
            },
            outputs={
                ACTIONS_KEY: FeatureSpec((self.horizon, self.action_dim), "float64")
            },
            metadata={
                "model": "dm05",
                "task": self.task,
                "image_size": c.wire_image_size,
                "camera_names": c.camera_names,
                "action_dim": self.action_dim,
                "horizon": self.horizon,
                "num_inference_steps": self.config.num_inference_steps,
                "history_slots": c.history_slots,
            },
        )

    def predict(self, inputs):
        c, torch = self.processor_config, self.torch
        now = time.monotonic()
        images = [
            np.array(inputs[IMAGE_PREFIX + name], copy=True) for name in c.camera_names
        ]
        history = self.history.select(now)
        raw_state = np.array(inputs[STATE_KEY], dtype=np.float32, copy=True)
        processed = self.processor.preprocess(
            {
                "images": images,
                "state": raw_state,
                "task": self.task,
                "history_images": history,
            }
        )
        self.history.append(
            now, self.history_image(images[c.camera_names.index(c.history_camera)])
        )
        noise = torch.randn(
            (1, self.horizon, self.padded_dim),
            generator=self.generator,
            dtype=torch.bfloat16,
        )
        with torch.inference_mode():
            normalized = self.policy.predict(
                self.request_type(processed, noise, self.config.num_inference_steps)
            )
            actions = self.processor.postprocess(
                {"action": normalized.float().cpu(), "state": raw_state}
            )
        result = actions[0].numpy().astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError("DM0.5 produced non-finite actions")
        return {ACTIONS_KEY: result}

    def reset_run(self):
        self.generator.manual_seed(self.config.seed)
        self.history.reset()

    def close(self):
        self.history.reset()
        self.engine.close()
