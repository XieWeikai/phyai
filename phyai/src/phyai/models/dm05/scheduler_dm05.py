"""OpenDM's noise-to-action Euler schedule, through the model runner."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.models.dm05.model_runner_dm05 import DM05ModelRunner
from phyai.runtime.schedule import Scheduler


@dataclass
class DM05Request:
    inputs: dict[str, torch.Tensor | None]
    noise: torch.Tensor
    num_inference_steps: int = 10
    action_mask: torch.Tensor | None = None


class DM05Scheduler(Scheduler):
    def __init__(self, runner: DM05ModelRunner):
        self.runner = runner
        self.config = runner.model.config
        self.device = runner.model.action_in_proj.weight.device
        self.dtype = runner.model.time_mlp_in.weight.dtype

    def setup(self):
        self.runner.setup()

    @torch.inference_mode()
    def step(self, request: DM05Request):
        if request.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        shape = (1, self.config.chunk_size, self.config.action_dim)
        if tuple(request.noise.shape) != shape:
            raise ValueError(f"noise must have shape {shape}")
        x = request.noise.to(device=self.device, dtype=self.dtype)
        mask = request.action_mask
        if mask is not None:
            if tuple(mask.shape) not in (
                (self.config.action_dim,),
                shape,
                (1, 1, self.config.action_dim),
            ):
                raise ValueError("action_mask has incompatible dimensions")
            mask = mask.to(self.device)
        try:
            self.runner.prefill(request.inputs)
            time = 1.0
            dt = -1.0 / request.num_inference_steps
            for _ in range(request.num_inference_steps):
                if mask is not None:
                    x = x * mask
                times = torch.full((1,), time, dtype=self.dtype, device=self.device)
                velocity = self.runner.forward((x, times))
                x = x + dt * velocity
                time += dt
            return x
        finally:
            self.runner.reset()

    def close(self):
        self.runner.close()
