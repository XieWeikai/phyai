"""OpenDM prompt, image, state and action processing on existing pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline, ProcessorStep
from phyai_utils_tools.processing.steps.action_steps import (
    DeltaActionStep,
    SliceActionStep,
)
from phyai_utils_tools.processing.steps.normalize_steps import (
    NormalizerStep,
    UnnormalizerStep,
)


class DM05QuantilesStep(ProcessorStep):
    """OpenDM adds clipping and zero-stat handling to quantile normalization."""

    def __init__(self, stats, key, *, inverse=False):
        self.key, self.inverse = key, inverse
        self.lo = torch.tensor(stats["q01"], dtype=torch.float32)
        self.hi = torch.tensor(stats["q99"], dtype=torch.float32)
        if (
            self.lo.ndim != 1
            or self.lo.shape != self.hi.shape
            or not torch.isfinite(self.lo).all()
            or not torch.isfinite(self.hi).all()
            or (self.hi < self.lo).any()
        ):
            raise ValueError(f"Invalid {key} quantile statistics")
        cls = UnnormalizerStep if inverse else NormalizerStep
        self.step = cls(
            features={key: {"type": key.upper(), "shape": [len(self.lo)]}},
            norm_map={key.upper(): "QUANTILES"},
            stats={key: stats},
            eps=1e-6,
        )

    def __call__(self, transition):
        values = torch.as_tensor(transition[self.key], dtype=torch.float32)
        if values.shape[-1] != len(self.lo) or not torch.isfinite(values).all():
            raise ValueError(f"{self.key} must be finite and match norm stats")
        if not self.inverse:
            values = torch.clip(values, self.lo, self.hi)
        result = self.step({**transition, self.key: values})
        result[self.key] = torch.where(
            (self.lo == 0) & (self.hi == 0), 0.0, result[self.key]
        )
        return result


def square_image(value, size):
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("DM0.5 images must be RGB uint8 HWC arrays")
    h, w = array.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    array = cv2.copyMakeBorder(
        array, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=0
    )
    return Image.fromarray(
        cv2.resize(array, (size, size), interpolation=cv2.INTER_LINEAR)
    )


class DM05ChatStep(ProcessorStep):
    def __init__(
        self,
        processor,
        *,
        image_prompts,
        robot_type,
        control_mode,
        speed,
        history_slots,
        image_size,
        max_length,
    ):
        self.processor = processor
        self.image_prompts = tuple(image_prompts)
        self.robot_type, self.control_mode, self.speed = robot_type, control_mode, speed
        self.history_slots, self.image_size, self.max_length = (
            history_slots,
            image_size,
            max_length,
        )

    def __call__(self, transition):
        images = transition["images"]
        if len(images) != len(self.image_prompts):
            raise ValueError("Camera count must match image_prompts in training order")
        task = transition["task"]
        text = ""
        if self.robot_type is not None:
            text += f"Robot: {self.robot_type}\n"
        if self.control_mode is not None:
            mode = (
                "end effector"
                if self.control_mode in {"ee", "eef", "end effector"}
                else self.control_mode
            )
            text += f"Control mode: {mode}\n"
        text += f"Overall speed: {self.speed}\nTask: {task}.\n"
        history = [
            image for image in transition.get("history_images", []) if image is not None
        ]
        if len(history) > self.history_slots:
            history = history[-self.history_slots :] if self.history_slots else []
        if self.history_slots:
            text += (
                "History images: "
                + "<unused1>" * (16 * (self.history_slots - len(history)))
                + ("<unused0>" * 16 + "\n") * len(history)
            )
        content = [{"type": "text", "text": text}]
        for name, image in zip(self.image_prompts, images, strict=True):
            if content[-1]["type"] == "text":
                content[-1]["text"] += f"{name} image: "
            else:
                content.append({"type": "text", "text": f"{name} image: "})
            content.append(
                {"type": "image", "image": square_image(image, self.image_size)}
            )
        state = transition["state"].numpy()
        bins = np.floor((np.clip(state, -1, 1) + 1) / 2 * 255).astype(int)
        content.append(
            {
                "type": "text",
                "text": "States: "
                + " ".join(str(b) for b in np.clip(bins, 0, 255).tolist()),
            }
        )
        values = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if values["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                "Prompt exceeds max_length; shorten the task instead of truncating image/state tokens"
            )
        values = dict(values)
        values["history_mask"] = values[
            "input_ids"
        ] == self.processor.tokenizer.convert_tokens_to_ids("<unused0>")
        values["token_type_ids"] = values["token_type_ids"].clone()
        values["token_type_ids"][values["history_mask"]] = 1
        values["history_pixel_values"] = (
            self.processor.image_processor(
                images=[square_image(image, self.image_size) for image in history],
                return_tensors="pt",
            )["pixel_values"]
            if history
            else None
        )
        return values


class DM05Processor(BaseModelProcessor):
    """Single-observation joint-space processor. Stats and layouts must match SFT."""

    def __init__(
        self,
        checkpoint,
        *,
        norm_stats,
        image_prompts,
        delta_mask,
        robot_type=None,
        control_mode="joint",
        speed="0.5",
        history_slots=32,
        image_size=448,
        max_length=4095,
        processor=None,
    ):
        if not image_prompts or len(delta_mask) == 0 or history_slots < 0:
            raise ValueError("Cameras, action layout and history_slots must be valid")
        if not isinstance(speed, str):
            raise ValueError("speed must be the string recorded in the training prompt")
        self.action_dim = len(delta_mask)
        self.delta_mask = list(delta_mask)
        if self.action_dim > 32 or not all(isinstance(v, bool) for v in delta_mask):
            raise ValueError("delta_mask must contain up to 32 booleans")
        if isinstance(norm_stats, (str, Path)):
            norm_stats = json.loads(Path(norm_stats).read_text())
        if "norm_stats_by_robot" in norm_stats:
            key = robot_type or norm_stats.get("default_robot_type")
            stats = norm_stats["norm_stats_by_robot"][key]
        else:
            stats = norm_stats.get("norm_stats", norm_stats)
        modes = norm_stats.get("control_modes")
        if modes:
            key = "eef" if control_mode in {"ee", "end effector"} else control_mode
            stats = modes[key]
        if any(
            len(stats[key]["q01"]) != self.action_dim for key in ("state", "action")
        ):
            raise ValueError("State and action stats must match delta_mask width")
        self.stats = stats
        self.processor = processor or AutoProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.chat = DM05ChatStep(
            self.processor,
            image_prompts=image_prompts,
            robot_type=robot_type,
            control_mode=control_mode,
            speed=speed,
            history_slots=history_slots,
            image_size=image_size,
            max_length=max_length,
        )
        super().__init__()

    def build_preprocessor(self):
        return ProcessorPipeline(
            [DM05QuantilesStep(self.stats["state"], "state"), self.chat]
        )

    def build_postprocessor(self):
        return ProcessorPipeline(
            [
                SliceActionStep(action_dim=self.action_dim),
                DM05QuantilesStep(self.stats["action"], "action", inverse=True),
                DeltaActionStep(delta_mask=self.delta_mask),
            ],
            to_output=lambda data: data["action"],
        )
