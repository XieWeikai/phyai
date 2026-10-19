"""Inference geometry read from an OpenDM DM0.5 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Mapping

from phyai.models.configuration import PretrainedConfig
from phyai.utils.checkpoint import load_config


@dataclass(frozen=True)
class DM05TextConfig(PretrainedConfig):
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    vocab_size: int = 262208
    rms_norm_eps: float = 1e-6
    query_pre_attn_scalar: float = 256
    sliding_window: int = 4096
    max_position_embeddings: int = 131072
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attn_logit_softcapping: float | None = None
    use_bidirectional_attention: bool = False
    layer_types: tuple[str, ...] = ()
    rope_parameters: dict = field(
        default_factory=lambda: {
            "full_attention": {
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 8.0,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
    )

    def __post_init__(self):
        if any(
            value <= 0
            for value in (
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.query_pre_attn_scalar,
                self.sliding_window,
            )
        ):
            raise ValueError("Text and attention geometry must be positive")
        if (
            self.attention_bias
            or self.attn_logit_softcapping is not None
            or self.use_bidirectional_attention
        ):
            raise ValueError("Unsupported Gemma3 text attention variant")
        if not self.layer_types:
            object.__setattr__(
                self,
                "layer_types",
                tuple(
                    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                    for i in range(self.num_hidden_layers)
                ),
            )
        else:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("Invalid GQA / rotary geometry")
        if set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError("Unsupported Gemma3 attention type")
        for name in self.layer_types:
            rope = self.rope_parameters.get(name, {})
            if (
                rope.get("rope_type") not in {"default", "linear"}
                or rope.get("rope_theta", 0) <= 0
            ):
                raise ValueError("Unsupported rotary parameters")
            if rope["rope_type"] == "linear" and rope.get("factor", 0) <= 0:
                raise ValueError("Linear rotary scaling must be positive")


@dataclass(frozen=True)
class DM05VisionConfig(PretrainedConfig):
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 448
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"

    def __post_init__(self):
        if any(
            value <= 0
            for value in (
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.image_size,
                self.patch_size,
                self.num_channels,
            )
        ):
            raise ValueError("Vision geometry must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                "Vision hidden_size must be divisible by num_attention_heads"
            )


@dataclass(frozen=True)
class DM05Config(PretrainedConfig):
    nested_sources: ClassVar[Mapping[str, str | tuple[str, ...]]] = {
        "text": "vlm_config.text_config",
        "expert": "action_config",
        "vision": "vlm_config.vision_config",
        "image_token_index": "vlm_config.image_token_index",
        "mm_tokens_per_image": "vlm_config.mm_tokens_per_image",
    }

    model_type: str = "dm05"
    text: DM05TextConfig = field(default_factory=DM05TextConfig)
    expert: DM05TextConfig = field(
        default_factory=lambda: DM05TextConfig(hidden_size=1024, intermediate_size=4096)
    )
    vision: DM05VisionConfig = field(default_factory=DM05VisionConfig)
    action_dim: int = 32
    chunk_size: int = 50
    image_token_index: int = 262144
    pad_token_id: int = 0
    history_pad_token_id: int = 7
    history_pool_size: int = 4
    mm_tokens_per_image: int = 256

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> DM05Config:
        path = Path(path)
        return cls.from_json(path) if path.is_file() else load_config(path, cls)

    def __post_init__(self):
        if self.model_type != "dm05":
            raise ValueError("Expected an OpenDM DM0.5 config.json")
        geometry = (
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
        if any(getattr(self.text, k) != getattr(self.expert, k) for k in geometry):
            raise ValueError("VLM and action expert must share attention geometry")
        if self.text.layer_types != self.expert.layer_types:
            raise ValueError("VLM and expert layer types must match")
        if self.text.rope_parameters != self.expert.rope_parameters:
            raise ValueError("VLM and expert rotary parameters must match")
        if self.history_pool_size != 4 or self.mm_tokens_per_image <= 0:
            raise ValueError(
                "DM0.5 MEM requires 4x4 history tokens and positive image tokens"
            )
        if self.chunk_size <= 0 or self.action_dim <= 0:
            raise ValueError("Action geometry must be positive")
        patches = self.vision.image_size // self.vision.patch_size
        side = int(self.mm_tokens_per_image**0.5)
        if (
            self.vision.image_size % self.vision.patch_size
            or side**2 != self.mm_tokens_per_image
            or patches % side
        ):
            raise ValueError("Invalid vision pooling geometry")
