"""Per-request prefix cache and model input preparation."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from phyai.layers.attention.mask import AttnMask
from phyai.models.dm05.modeling_dm05 import DM05Model, DM05PrefixMask
from phyai.runtime.model_runner import ModelRunner


class DM05ModelRunner(ModelRunner):
    def __init__(self, model: DM05Model):
        self.model = model
        self.reset()

    def setup(self):
        self.model.eval()

    def reset(self):
        self.cache = None
        self.positions = None
        self.mask = None

    def prefill(self, inputs):
        self.reset()
        device = self.model.action_in_proj.weight.device
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        ids = inputs["input_ids"]
        c = self.model.config
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("DM0.5 serving currently supports batch size one")
        if ids.shape[1] >= c.text.sliding_window:
            raise ValueError("Prefix must be shorter than the Gemma3 sliding window")
        if (
            ids.dtype != torch.long
            or ids.numel() == 0
            or bool(((ids < 0) | (ids >= c.text.vocab_size)).any())
        ):
            raise ValueError("input_ids must contain valid int64 vocabulary IDs")
        for key in ("attention_mask", "token_type_ids", "history_mask"):
            if inputs[key].shape != ids.shape:
                raise ValueError(f"{key} must have the same shape as input_ids")
        pixels = inputs["pixel_values"]
        image_shape = (c.vision.num_channels, c.vision.image_size, c.vision.image_size)
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != image_shape:
            raise ValueError(
                "pixel_values must have shape (images, channels, height, width)"
            )
        if (
            int((ids == c.image_token_index).sum())
            != pixels.shape[0] * c.mm_tokens_per_image
        ):
            raise ValueError("Current image token and pixel counts do not match")
        history = inputs.get("history_pixel_values")
        if history is not None and (
            history.ndim != 4 or tuple(history.shape[1:]) != image_shape
        ):
            raise ValueError("history_pixel_values has incompatible image dimensions")
        expected_history = (
            0 if history is None else history.shape[0] * c.history_pool_size**2
        )
        if int(inputs["history_mask"].sum()) != expected_history:
            raise ValueError("History token and pixel counts do not match")
        visible = inputs["attention_mask"].bool() & (ids != c.history_pad_token_id)
        positions = (visible.long().cumsum(-1) - 1).clamp_min(0)
        image = inputs["token_type_ids"] == 1
        starts = image & ~F.pad(image, (1, 0), value=False)[:, :-1]
        groups = starts.long().cumsum(-1) - 1
        groups = groups.masked_fill(~image, -1)
        masks = {
            name: DM05PrefixMask(
                key_mask=visible,
                image_groups=groups,
                window=c.text.sliding_window if name == "sliding_attention" else None,
            )
            for name in set(c.text.layer_types)
        }
        self.cache, hidden = self.model.encode_prefix(inputs, masks, positions)
        valid_suffix_prefix = (ids != c.pad_token_id) & (ids != c.history_pad_token_id)
        self.positions = (
            valid_suffix_prefix.sum(-1)[:, None]
            + torch.arange(c.chunk_size, device=device)[None]
        )
        self.mask = AttnMask.from_key_mask(
            torch.cat(
                (
                    valid_suffix_prefix,
                    torch.ones((1, c.chunk_size), dtype=torch.bool, device=device),
                ),
                dim=1,
            )
        )
        return hidden

    def forward(self, batch):
        if self.cache is None:
            raise RuntimeError("prefill must run before an action step")
        actions, times = batch
        return self.model(
            actions, times, cache=self.cache, positions=self.positions, mask=self.mask
        )

    def close(self):
        self.reset()
