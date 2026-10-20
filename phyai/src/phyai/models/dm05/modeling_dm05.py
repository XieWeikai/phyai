"""Stateless DM0.5 Gemma3/SigLIP and AdaRMS action expert.

The runner owns prefix K/V. Attention, norms, rotary embeddings, MLPs and
projections use the shared PhyAI layers. No Transformers modeling code runs here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from phyai.layers.activation import SiLU
from phyai.layers.attention import Attention, AttnMask
from phyai.layers.conv import Conv2d
from phyai.layers.layer_norm import AdaRMSNorm, GemmaRMSNorm, LayerNorm
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.layers.transformer_block import TransformerBlock
from phyai.layers.vocab_embedding import VocabParallelEmbedding
from phyai.models.dm05.configuration_dm05 import DM05Config, DM05TextConfig
from phyai.weights.shards import replicated


def source_parameter(shape, name, *, dtype, device):
    parameter = nn.Parameter(
        torch.empty(shape, dtype=dtype, device=device), requires_grad=False
    )
    parameter.hf_keys = [(name, None)]
    parameter.weight_loader = replicated()
    return parameter


@dataclass(frozen=True)
class DM05PrefixMask(AttnMask):
    """Gemma3 causal text / bidirectional image blocks, including local windows."""

    image_groups: torch.Tensor | None = None
    window: int | None = None

    @property
    def kind(self):
        return "segments"

    def dense(self, S_q, S_kv, batch_size, device, *, causal, sliding_window):
        q = torch.arange(S_q, device=device)[:, None]
        k = torch.arange(S_kv, device=device)[None, :]
        allowed = q >= k
        if self.window is not None:
            allowed = allowed & (q - k < self.window)
        if self.image_groups is not None:
            groups = self.image_groups
            image_block = (groups[:, :, None] == groups[:, None, :]) & (
                groups[:, :, None] >= 0
            )
            allowed = allowed[None, :, :] | image_block
        else:
            allowed = allowed[None, :, :]
        return (allowed & self.key_mask[:, None, :])[:, None, :, :]


class DM05Vision(nn.Module):
    def __init__(self, config: DM05Config, *, dtype, device):
        super().__init__()
        c = config.vision
        p = "model.vlm.model.vision_tower.vision_model"
        self.patch_embedding = Conv2d(
            c.num_channels,
            c.hidden_size,
            c.patch_size,
            stride=c.patch_size,
            dtype=dtype,
            device=device,
            prefix=p + ".embeddings.patch_embedding",
        )
        patches = (c.image_size // c.patch_size) ** 2
        self.position_embedding = source_parameter(
            (patches, c.hidden_size),
            p + ".embeddings.position_embedding.weight",
            dtype=dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    c.hidden_size,
                    c.num_attention_heads,
                    c.intermediate_size,
                    attn_causal=False,
                    attn_backend="sdpa",
                    attn_bias=True,
                    attn_out_hf_name="out_proj",
                    norm_type="layernorm",
                    norm_bias=True,
                    norm_hf_names={
                        "input_layernorm": "layer_norm1",
                        "post_attention_layernorm": "layer_norm2",
                    },
                    norm_eps=c.layer_norm_eps,
                    mlp_gated=False,
                    mlp_activation=c.hidden_act,
                    mlp_bias=True,
                    params_dtype=dtype,
                    device=device,
                    prefix=f"{p}.encoder.layers.{i}",
                )
                for i in range(c.num_hidden_layers)
            ]
        )
        self.post_layernorm = LayerNorm(
            c.hidden_size,
            eps=c.layer_norm_eps,
            bias=True,
            dtype=dtype,
            device=device,
            prefix=p + ".post_layernorm",
        )
        p = "model.vlm.model.multi_modal_projector"
        self.projector_norm = GemmaRMSNorm(
            c.hidden_size,
            eps=c.layer_norm_eps,
            dtype=dtype,
            device=device,
            prefix=p + ".mm_soft_emb_norm",
        )
        self.projection = source_parameter(
            (c.hidden_size, config.text.hidden_size),
            p + ".mm_input_projection_weight",
            dtype=dtype,
            device=device,
        )
        self.grid_size = c.image_size // c.patch_size
        self.pool_stride = self.grid_size // int(config.mm_tokens_per_image**0.5)

    def forward(self, pixels):
        x = (
            self.patch_embedding(pixels.to(self.patch_embedding.weight.dtype))
            .flatten(2)
            .transpose(1, 2)
        )
        x = x + self.position_embedding[None]
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        grid = (
            x.transpose(1, 2)
            .reshape(x.shape[0], x.shape[-1], self.grid_size, self.grid_size)
            .contiguous()
        )
        x = (
            F.avg_pool2d(grid, self.pool_stride, self.pool_stride)
            .flatten(2)
            .transpose(1, 2)
        )
        return (self.projector_norm(x) @ self.projection).to(x.dtype)


class DM05DecoderLayer(nn.Module):
    def __init__(self, c: DM05TextConfig, index, prefix, *, expert, dtype, device):
        super().__init__()
        self.block = TransformerBlock(
            c.hidden_size,
            c.num_attention_heads,
            c.intermediate_size,
            num_kv_heads=c.num_key_value_heads,
            head_dim=c.head_dim,
            sandwich_norm=True,
            attn_qk_norm=True,
            attn_causal=False,
            attn_backend="sdpa",
            attn_scale=c.query_pre_attn_scalar**-0.5,
            norm_type="gemma_rmsnorm",
            norm_eps=c.rms_norm_eps,
            mlp_activation=c.hidden_activation,
            mlp_cast_before_multiply=True,
            params_dtype=dtype,
            device=device,
            prefix=prefix,
        )
        self.attention = Attention(
            c.num_attention_heads,
            c.head_dim,
            num_kv_heads=c.num_key_value_heads,
            scale=c.query_pre_attn_scalar**-0.5,
            causal=False,
            backend="sdpa",
        )
        self.expert = expert
        if expert:
            base = "model.action_expert"
            self.input_adarms = AdaRMSNorm(
                c.hidden_size, c.hidden_size, c.rms_norm_eps, dtype=dtype, device=device
            )
            self.mlp_adarms = AdaRMSNorm(
                c.hidden_size, c.hidden_size, c.rms_norm_eps, dtype=dtype, device=device
            )
            for norm, name in (
                (self.input_adarms, "input_time_modulators"),
                (self.mlp_adarms, "mlp_time_modulators"),
            ):
                for suffix in ("weight", "bias"):
                    parameter = getattr(norm.dense, suffix)
                    parameter.hf_keys = [(f"{base}.{name}.{index}.{suffix}", None)]
                    parameter.weight_loader = replicated()

    def forward(self, x, positions, rope, mask, *, prefix_kv=None, cond=None):
        b = self.block
        if self.expert:
            h, attn_gate = self.input_adarms(x, cond)
        else:
            h, attn_gate = b.input_norm(x), 1
        fused, _ = b.qkv_proj(h)
        q_dim = b.q_heads_local * b.head_dim
        kv_dim = b.kv_heads_local * b.head_dim
        q, k, v = fused.split((q_dim, kv_dim, kv_dim), dim=-1)
        leading = fused.shape[:-1]
        q = b.q_norm(q.reshape(*leading, b.q_heads_local, b.head_dim))
        k = b.k_norm(k.reshape(*leading, b.kv_heads_local, b.head_dim))
        v = v.reshape(*leading, b.kv_heads_local, b.head_dim)
        q, k = rope(positions, q, k)
        current_kv = k, v
        if prefix_kv is not None:
            k = torch.cat((prefix_kv[0], k), dim=1)
            v = torch.cat((prefix_kv[1], v), dim=1)
        out = self.attention(q, k, v, mask=mask).flatten(-2)
        out, _ = b.o_proj(out)
        x = x + b.post_attn_norm(out) * attn_gate
        if self.expert:
            h, mlp_gate = self.mlp_adarms(x, cond)
        else:
            h, mlp_gate = b.pre_ff_norm(x), 1
        return x + b.post_ff_norm(b.mlp(h)) * mlp_gate, current_kv


class DM05Model(nn.Module):
    def __init__(self, config: DM05Config, *, params_dtype=torch.bfloat16, device=None):
        super().__init__()
        self.config = config
        c = config.text
        self.vision = DM05Vision(config, dtype=params_dtype, device=device)
        self.embed_tokens = VocabParallelEmbedding(
            c.vocab_size,
            c.hidden_size,
            embed_scale=c.hidden_size**0.5,
            params_dtype=params_dtype,
            device=device,
            prefix="model.vlm.model.language_model.embed_tokens",
        )
        self.text_layers = nn.ModuleList(
            [
                DM05DecoderLayer(
                    c,
                    i,
                    f"model.vlm.model.language_model.layers.{i}",
                    expert=False,
                    dtype=params_dtype,
                    device=device,
                )
                for i in range(c.num_hidden_layers)
            ]
        )
        self.text_norm = GemmaRMSNorm(
            c.hidden_size,
            eps=c.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix="model.vlm.model.language_model.norm",
        )
        e = config.expert
        self.expert_layers = nn.ModuleList(
            [
                DM05DecoderLayer(
                    e,
                    i,
                    f"model.action_expert.layers.{i}",
                    expert=True,
                    dtype=params_dtype,
                    device=device,
                )
                for i in range(e.num_hidden_layers)
            ]
        )
        self.final_adarms = AdaRMSNorm(
            e.hidden_size,
            e.hidden_size,
            e.rms_norm_eps,
            dtype=params_dtype,
            device=device,
        )
        for suffix in ("weight", "bias"):
            p = getattr(self.final_adarms.dense, suffix)
            p.hf_keys = [(f"model.action_expert.final_time_modulator.{suffix}", None)]
            p.weight_loader = replicated()
        self.action_in_proj = ReplicatedLinear(
            config.action_dim,
            e.hidden_size,
            params_dtype=torch.float32,
            device=device,
            prefix="model.action_in_proj",
        )
        self.action_out_proj = ReplicatedLinear(
            e.hidden_size,
            config.action_dim,
            params_dtype=torch.float32,
            device=device,
            prefix="model.action_out_proj",
        )
        self.time_mlp_in = ReplicatedLinear(
            e.hidden_size,
            e.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix="model.time_mlp_in",
        )
        self.time_mlp_out = ReplicatedLinear(
            e.hidden_size,
            e.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix="model.time_mlp_out",
        )
        self.silu = SiLU()
        self.ropes = nn.ModuleDict()
        for name in set(c.layer_types):
            values = c.rope_parameters[name]
            self.ropes[name] = RotaryEmbedding(
                c.head_dim,
                c.max_position_embeddings,
                rope_theta=values["rope_theta"],
                rope_type=values["rope_type"],
                rope_scaling={"factor": values["factor"]}
                if values["rope_type"] == "linear"
                else None,
                device=device,
            )
        fraction = torch.linspace(
            0, 1, e.hidden_size // 2, dtype=torch.float32, device=device
        )
        self.register_buffer(
            "time_frequencies",
            1 / (4e-3 * (4.0 / 4e-3) ** fraction) * (2 * torch.pi),
            persistent=False,
        )

    def encode_prefix(self, inputs, masks, positions):
        c = self.config
        x = self.embed_tokens(inputs["input_ids"])
        x = x.masked_fill((inputs["input_ids"] == c.history_pad_token_id)[..., None], 0)
        images = self.vision(inputs["pixel_values"])
        x = x.masked_scatter(
            (inputs["input_ids"] == c.image_token_index)[..., None].expand_as(x),
            images.to(x.dtype),
        )
        history_pixels = inputs.get("history_pixel_values")
        if history_pixels is not None and history_pixels.shape[0]:
            history = self.vision(history_pixels)
            side = int(history.shape[1] ** 0.5)
            grid = history.transpose(1, 2).reshape(-1, history.shape[-1], side, side)
            history = (
                F.adaptive_avg_pool2d(grid, (c.history_pool_size, c.history_pool_size))
                .flatten(2)
                .transpose(1, 2)
            )
            x = x.masked_scatter(
                inputs["history_mask"][..., None].expand_as(x), history.to(x.dtype)
            )
        cache = []
        for layer, name in zip(self.text_layers, c.text.layer_types, strict=True):
            x, kv = layer(x, positions, self.ropes[name], masks[name])
            cache.append(kv)
        return tuple(cache), self.text_norm(x)

    def forward(self, actions, times, *, cache, positions, mask):
        angles = times.float()[:, None] * self.time_frequencies[None]
        cond = torch.cat((angles.sin(), angles.cos()), dim=-1).to(
            self.time_mlp_in.weight.dtype
        )
        cond, _ = self.time_mlp_in(cond)
        cond, _ = self.time_mlp_out(self.silu(cond))
        cond = self.silu(cond)
        x, _ = self.action_in_proj(actions.float())
        x = x.to(cond.dtype)
        for layer, kv, name in zip(
            self.expert_layers, cache, self.config.expert.layer_types, strict=True
        ):
            x, _ = layer(x, positions, self.ropes[name], mask, prefix_kv=kv, cond=cond)
        x, _ = self.final_adarms(x, cond)
        output, _ = self.action_out_proj(x.float())
        return output
