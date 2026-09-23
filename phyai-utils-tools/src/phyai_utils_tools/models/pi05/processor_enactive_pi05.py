"""Checkpoint-owned input and action transforms for Enactive's three-view Pi0.5."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline, ProcessorStep


@dataclass
class EnactivePI05ProcessedInputs:
    """Engine inputs plus normalized state retained for reference comparisons."""

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    lang_lens: torch.Tensor
    state: torch.Tensor


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _state_batch(value: Any) -> np.ndarray:
    state = np.asarray(_numpy(value), dtype=np.float32)
    if state.ndim == 1:
        state = state[None, :]
    if state.ndim != 2 or state.shape[0] == 0 or state.shape[1] != 14:
        raise ValueError("state must have shape [14] or [batch, 14]")
    if not np.isfinite(state).all():
        raise ValueError("state must contain only finite values")
    return state


def _image_pixels(value: Any) -> np.ndarray:
    image = _numpy(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            "images must be RGB uint8 arrays with shape [height, width, 3]"
        )
    height, width = image.shape[:2]
    if min(height, width) < 1:
        raise ValueError("images must have positive height and width")
    if (height, width) != (224, 224):
        ratio = max(width / 224, height / 224)
        resized_height, resized_width = int(height / ratio), int(width / ratio)
        if min(resized_height, resized_width) < 1:
            raise ValueError("image aspect ratio is too extreme for a 224-pixel canvas")
        resized = Image.fromarray(image).resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
        canvas = Image.new(resized.mode, (224, 224), 0)
        canvas.paste(resized, ((224 - resized_width) // 2, (224 - resized_height) // 2))
        image = np.asarray(canvas, dtype=np.uint8)
    pixels = image.astype(np.float32) / np.float32(255.0) * np.float32(
        2.0
    ) - np.float32(1.0)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def _quantiles(stats: Mapping[str, Any], name: str) -> tuple[np.ndarray, np.ndarray]:
    if stats.get("valid_dim") != 14:
        raise ValueError(f"{name}.valid_dim must be 14")
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    if low.shape != (32,) or high.shape != (32,):
        raise ValueError(f"{name} quantiles must have model width 32")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high < low):
        raise ValueError(f"{name} quantiles must be finite and ordered")
    if np.any(high[14:] != low[14:]) or not np.any(high[:14] > low[:14]):
        raise ValueError(
            f"{name} quantiles must have 14 valid dimensions and constant padding"
        )
    return low[:14].copy(), high[:14].copy()


class _EnactiveInputStep(ProcessorStep):
    def __init__(self, processor: EnactivePI05Processor) -> None:
        self.processor = processor

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        processor = self.processor
        states = _state_batch(transition["state"])
        batch_size = len(states)
        tasks = transition["task"]
        if isinstance(tasks, str):
            tasks = [tasks]
        if (
            not isinstance(tasks, Sequence)
            or len(tasks) != batch_size
            or any(not isinstance(task, str) for task in tasks)
        ):
            raise ValueError("task must contain one instruction string per state")
        views = transition["images"]
        if not isinstance(views, Mapping) or set(views) != {"front", "left", "right"}:
            raise ValueError(
                "images must contain exactly the front, left and right RGB views"
            )
        cameras = []
        for name in ("front", "left", "right"):
            camera = views[name]
            if isinstance(camera, (np.ndarray, torch.Tensor)) and camera.ndim == 3:
                camera = [camera]
            if (
                not isinstance(camera, (Sequence, np.ndarray, torch.Tensor))
                or len(camera) != batch_size
            ):
                raise ValueError(f"images.{name} must contain one RGB image per state")
            cameras.append(np.stack([_image_pixels(value) for value in camera]))
        pixels = np.stack(cameras, axis=1)
        token_ids = np.zeros((batch_size, 200), dtype=np.int64)
        token_lengths = np.zeros(batch_size, dtype=np.int64)
        for index, task in enumerate(tasks):
            cleaned = task.strip().replace("_", " ").replace("\n", " ")
            tokens = processor.tokenizer.encode(cleaned, add_bos=True)
            tokens += processor.tokenizer.encode("\n")
            tokens = tokens[:200]
            token_ids[index, : len(tokens)] = tokens
            token_lengths[index] = len(tokens)
        low, high = processor.state_quantiles
        normalized = (states - low) / (high - low + 1e-6) * 2.0 - 1.0
        normalized = np.pad(normalized, ((0, 0), (0, 18))).astype(np.float32)
        return {
            "pixel_values": torch.from_numpy(pixels).to(processor.device),
            "input_ids": torch.from_numpy(token_ids).to(processor.device),
            "lang_lens": torch.from_numpy(token_lengths).to(processor.device),
            "state": torch.from_numpy(normalized).to(processor.device),
        }


class _EnactiveActionStep(ProcessorStep):
    def __init__(self, processor: EnactivePI05Processor) -> None:
        self.processor = processor

    def __call__(self, transition: dict[str, Any]) -> dict[str, Any]:
        values = transition["action"]
        if isinstance(values, torch.Tensor):
            values = values.detach().float().cpu().numpy()
        actions = np.asarray(values, dtype=np.float32)
        unbatched = actions.ndim == 2
        if unbatched:
            actions = actions[None, ...]
        if actions.ndim != 3 or actions.shape[1:] != (50, 32):
            raise ValueError("action must have shape [50, 32] or [batch, 50, 32]")
        states = _state_batch(transition["state"])
        if len(states) != len(actions):
            raise ValueError("action and state batch sizes must match")
        if not np.isfinite(actions).all():
            raise ValueError("action must contain only finite values")
        low, high = self.processor.action_quantiles
        physical = (actions[..., :14] + 1.0) / 2.0 * (high - low + 1e-6) + low
        mask = self.processor.delta_mask
        physical[..., mask] += states[:, None, mask]
        if not np.isfinite(physical).all():
            raise ValueError("decoded actions must contain only finite values")
        if unbatched:
            physical = physical[0]
        return {
            "action": torch.from_numpy(np.ascontiguousarray(physical, dtype=np.float32))
        }


class EnactivePI05Processor(BaseModelProcessor):
    """Three RGB views and instruction in; absolute bimanual joint targets out.

    ``preprocess`` accepts ``images={front, left, right}``, ``task`` and raw
    14-dimensional ``state``. Images are uint8 HWC, or NHWC for batches.
    ``postprocess`` accepts ``action`` and the corresponding raw ``state``;
    it returns a CPU float32 tensor and never keeps a previous observation.
    """

    def __init__(
        self,
        *,
        tokenizer_path: str | Path,
        normalization: Mapping[str, Any],
        delta_mask: Sequence[bool],
        device: torch.device | str = "cpu",
    ) -> None:
        try:
            import sentencepiece
        except ImportError as exc:
            raise ImportError(
                "EnactivePI05Processor requires sentencepiece; install the enactive optional extra"
            ) from exc
        self.state_quantiles = _quantiles(normalization["state"], "state")
        self.action_quantiles = _quantiles(normalization["action"], "action")
        expected_mask = [True] * 6 + [False] + [True] * 6 + [False]
        if list(delta_mask) != expected_mask or any(
            type(value) is not bool for value in delta_mask
        ):
            raise ValueError(
                "delta_mask must describe six joints and one absolute gripper per arm"
            )
        self.delta_mask = np.asarray(delta_mask, dtype=np.bool_)
        self.device = torch.device(device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=Path(tokenizer_path).read_bytes()
        )
        super().__init__()

    def build_preprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[_EnactiveInputStep(self)],
            name="enactive_pi05_preprocessor",
            to_output=lambda value: EnactivePI05ProcessedInputs(**value),
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[_EnactiveActionStep(self)],
            name="enactive_pi05_postprocessor",
            to_output=lambda value: value["action"],
        )

    @classmethod
    def from_pretrained(
        cls, checkpoint: str | Path, *, device: torch.device | str = "cpu"
    ) -> EnactivePI05Processor:
        """Load only the supported instruction-only, non-RTC manipulation contract."""
        root = Path(checkpoint)

        def read(name: str) -> dict[str, Any]:
            value = json.loads((root / name).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"{name} must contain a JSON object")
            return value

        config = read("enactive_config.json")
        manifest = read("checkpoint_manifest.json")
        metadata = read("vla_metadata.json")
        normalization = read("normalization.json")
        generator = config.get("dit_config", {}).get("generator_cfg", {})
        runtime = manifest.get("runtime", {})
        if (
            config.get("model_type") not in {"pi", "layerwise"}
            or config.get("understander_backend") != "paligemma"
        ):
            raise ValueError("Expected an Enactive Pi0.5 PaliGemma checkpoint")
        if runtime.get("online_policy_contract") != "manipulation":
            raise ValueError("Expected a manipulation policy contract")
        if generator.get("action_dim") != 32 or generator.get("action_horizon") != 50:
            raise ValueError("Expected model action shape [50, 32]")
        if runtime.get("valid_action_dim") != 14 or runtime.get("action_horizon") != 50:
            raise ValueError("Expected physical action shape [50, 14]")
        if (
            config.get("state_input_mode") != "und"
            or generator.get("state_input_mode") != "und"
        ):
            raise ValueError(
                "Expected the instruction-only understander state contract"
            )
        if (
            Path(config.get("prompt_path", "")).name
            != "openpi_pi05_instruction_3view.py"
        ):
            raise ValueError("Expected the three-view instruction-only prompt")
        if (
            config.get("max_token_len") != 200
            or config.get("visual_memory", {}).get("type") != "native"
        ):
            raise ValueError("Expected native visual inputs and 200 text tokens")
        if any(
            config.get(name, {}).get("enabled", False)
            for name in ("geometry_memory", "cognitive_map")
        ):
            raise ValueError("Memory-conditioned checkpoints are not supported")
        if (
            config.get("enable_summary", False)
            or config.get("fast_tokenizer_type") is not None
        ):
            raise ValueError("Summary and action-tokenizer inputs are not supported")
        if any(
            scope.get(name, False)
            for scope in (config, generator, runtime)
            for name in (
                "pi0_use_rtc",
                "pi0_use_training_rtc",
                "use_rtc",
                "rtc_training_max_delay_steps",
            )
        ):
            raise ValueError("RTC checkpoints are not supported")
        if runtime.get("policy_tokenizer_file") != "policy_tokenizer.model":
            raise ValueError("Expected a checkpoint-local policy_tokenizer.model")
        modalities = metadata.get("joint_position", {}).get("modalities", {})
        action = modalities.get("action", {}).get("joint_position", {})
        if (
            action.get("shape") != [14]
            or action.get("representation") != "chunk_anchor"
            or action.get("absolute") is not False
            or action.get("anchor") != "observation.state"
            or runtime.get("trajectory_representation") != "chunk_anchor"
        ):
            raise ValueError("Expected chunk-anchor bimanual joint-position actions")
        expected_views = {
            "base": "observation.images.front",
            "left_wrist": "observation.images.left",
            "right_wrist": "observation.images.right",
        }
        if {
            key: value.get("source_key")
            for key, value in modalities.get("video", {}).items()
        } != expected_views:
            raise ValueError(
                "Expected front, left-wrist and right-wrist camera metadata"
            )
        return cls(
            tokenizer_path=root / "policy_tokenizer.model",
            normalization=normalization,
            delta_mask=action.get("delta_mask", []),
            device=device,
        )


__all__ = ["EnactivePI05ProcessedInputs", "EnactivePI05Processor"]
