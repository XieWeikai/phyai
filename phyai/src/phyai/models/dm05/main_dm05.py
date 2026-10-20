"""DM0.5 Engine plugin; import this module to register ``dm05``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.dm05.configuration_dm05 import DM05Config
from phyai.models.dm05.modeling_dm05 import DM05Model
from phyai.models.dm05.model_runner_dm05 import DM05ModelRunner
from phyai.models.dm05.scheduler_dm05 import DM05Request, DM05Scheduler
from phyai.utils import get_logger
from phyai.weights import load_pretrained

logger = get_logger(__name__)


def dm05_weight_remap(name: str) -> str | None:
    # Language logits and the expert's unused affine final norm are not evaluated.
    if name in {
        "model.vlm.lm_head.weight",
        "lm_head.weight",
        "model.action_expert.norm.weight",
    }:
        return None
    return name


@dataclass
class DM05Args(EntryArgs):
    checkpoint_dir: str | Path = ""
    config: DM05Config | None = None


@Engine.register
class DM05Entry(Entry):
    name: ClassVar[str] = "dm05"
    args_cls: ClassVar[type[EntryArgs]] = DM05Args

    def __init__(self):
        self.model = None
        self.scheduler = None

    def setup(self, args: DM05Args):
        eng = get_engine_config()
        config = args.config or DM05Config.from_checkpoint(args.checkpoint_dir)
        self.model = DM05Model(
            config, params_dtype=eng.device.params_dtype, device=eng.device.target
        )
        report = load_pretrained(
            self.model, args.checkpoint_dir, remap=dm05_weight_remap, strict=True
        )
        logger.info_rank0("Loaded DM0.5: %d checkpoint tensors", len(report.loaded))
        self.scheduler = DM05Scheduler(DM05ModelRunner(self.model))
        self.scheduler.setup()

    def step(self, request: DM05Request):
        if self.scheduler is None:
            raise RuntimeError("DM0.5 entry is not initialized")
        return self.scheduler.step(request)

    def close(self):
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.model = None
