"""Convert a complete Enactive pi0.5 export without changing tensor bytes."""

from __future__ import annotations

import re
import json
import math
import shutil
import struct
import hashlib
import argparse
import tempfile
from typing import Any, BinaryIO
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

GENERATOR_PREFIX = "generator.flow_model."
EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
PALIGEMMA_PREFIX = "paligemma_with_expert.paligemma."
DROP_REASONS = {
    "state_proj.weight": "state_input_mode=und does not use the state projection",
    "state_proj.bias": "state_input_mode=und does not use the state projection",
    "_abs_scale_step_buf": "training auxiliary loss step counter",
    "_abs_loss_scale_buf": "training auxiliary loss scale",
    "action_min": "training auxiliary loss statistics; inference uses JSON statistics",
    "action_max": "training auxiliary loss statistics; inference uses JSON statistics",
    "traj_min": "training auxiliary loss statistics; inference uses JSON statistics",
    "traj_max": "training auxiliary loss statistics; inference uses JSON statistics",
}
SIDECARS = (
    "normalization.json",
    "vla_metadata.json",
    "checkpoint_manifest.json",
    "policy_tokenizer.model",
)
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}
COPY_BLOCK_BYTES = 8 * 1024 * 1024


def enactive_pi05_weight_remap(name: str) -> str | None:
    """Map an Enactive name to the existing pi0.5 loader's HF key."""
    if name == "understander.model.language_model.embed_tokens.weight":
        return PALIGEMMA_PREFIX + "lm_head.weight"
    if name.startswith("understander.model."):
        return PALIGEMMA_PREFIX + name.removeprefix("understander.")
    if not name.startswith(GENERATOR_PREFIX):
        raise ValueError(f"Unsupported Enactive weight: {name}")
    suffix = name.removeprefix(GENERATOR_PREFIX)
    if suffix in DROP_REASONS:
        return None
    if re.fullmatch(r"action_(in|out)_proj\.(weight|bias)", suffix):
        return suffix
    if re.fullmatch(r"action_time_mlp_(in|out)\.(weight|bias)", suffix):
        return suffix.removeprefix("action_")
    match = re.fullmatch(r"adarms_(input|post)_proj\.(\d+)\.(weight|bias)", suffix)
    if match:
        site, layer, parameter = match.groups()
        norm = "input_layernorm" if site == "input" else "post_attention_layernorm"
        return f"{EXPERT_PREFIX}layers.{layer}.{norm}.dense.{parameter}"
    if re.fullmatch(r"adarms_final_proj\.(weight|bias)", suffix):
        return EXPERT_PREFIX + "norm.dense." + suffix.rsplit(".", 1)[1]
    if suffix.startswith("generator_model.layers."):
        return EXPERT_PREFIX + suffix.removeprefix("generator_model.")
    raise ValueError(f"Unsupported Enactive weight: {name}")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path.name}")
    return value


def require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"Unsupported {label}: expected {expected!r}, got {value!r}")


def build_phyai_config(source: dict[str, Any]) -> dict[str, Any]:
    """Extract explicit geometry and reject contracts this adapter cannot serve."""
    require_equal(
        source.get("understander_backend"), "paligemma", "understander_backend"
    )
    if source.get("model_type") not in {"pi", "layerwise"}:
        raise ValueError("Expected an Enactive pi/layerwise export")
    require_equal(
        source.get("generator_core", "action_flow"), "action_flow", "generator_core"
    )
    require_equal(source.get("state_input_mode"), "und", "state_input_mode")
    require_equal(
        source.get("trajectory_representation"),
        "chunk_anchor",
        "trajectory_representation",
    )
    require_equal(
        source.get("visual_memory", {}).get("type", "native"),
        "native",
        "visual_memory.type",
    )
    require_equal(
        source.get("visual_memory", {}).get("output_mode", "current_only"),
        "current_only",
        "visual_memory.output_mode",
    )
    for key in ("geometry_memory", "cognitive_map", "discrete_action_classifier"):
        require_equal(
            source.get(key, {}).get("enabled", False), False, f"{key}.enabled"
        )
    for key in (
        "enable_summary",
        "inject_rel_pose_generator",
        "enable_knowledge_insulation",
    ):
        require_equal(source.get(key, False), False, key)
    for key in (
        "peft_config",
        "policy_input",
        "inference_context_mix",
        "fast_tokenizer_type",
    ):
        require_equal(source.get(key), None, key)
    generator = source["dit_config"]["generator_cfg"]
    for section in (source, generator):
        for key in ("pi0_use_cfg", "pi0_use_rtc", "pi0_use_training_rtc"):
            require_equal(section.get(key, False), False, key)
        require_equal(
            section.get("rtc_training_max_delay_steps", 0),
            0,
            "rtc_training_max_delay_steps",
        )
    for key, expected in {
        "adarms_layout": "split_per_norm",
        "state_input_mode": "und",
        "generator_attention_layout": "match_understander",
        "inference_init_mode": "gaussian",
        "vlm_und_causal": False,
        "add_pos_embed": False,
        "add_cross_attention": False,
        "enable_knowledge_insulation": False,
        "use_vlln": False,
    }.items():
        require_equal(generator.get(key, expected), expected, f"generator.{key}")
    vlm = source["vlm_config"]["vl_config"]
    vision_source = vlm["vision_config"]
    text_source = vlm["text_config"]
    diffusion = generator["diffusion_model_cfg"]
    require_equal(vlm.get("model_type"), "paligemma", "vl_config.model_type")
    require_equal(text_source.get("model_type"), "gemma", "text_config.model_type")
    require_equal(text_source.get("attention_bias", False), False, "attention_bias")
    require_equal(text_source.get("use_adarms", False), False, "text_config.use_adarms")
    require_equal(
        diffusion.get("interleave_self_attention", False),
        False,
        "interleave_self_attention",
    )
    if source.get("mlp_activation_implementation") not in {
        None,
        "fused_tanh",
        "reference_bf16_tanh",
    }:
        raise ValueError("Unsupported mlp_activation_implementation")
    for label, config in (("vision", vision_source), ("text", text_source)):
        if config.get("hidden_act") not in {
            "gelu_pytorch_tanh",
            "gelu_new",
            "gelu_tanh",
        }:
            raise ValueError(
                f"Unsupported {label}.hidden_act: {config.get('hidden_act')!r}"
            )
    vision_fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "image_size",
        "patch_size",
        "num_channels",
        "layer_norm_eps",
    )
    text_fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
        "vocab_size",
        "rms_norm_eps",
        "rope_theta",
        "max_position_embeddings",
    )
    vision = {name: vision_source[name] for name in vision_fields}
    vision["projection_dim"] = vlm["projection_dim"]
    text = {name: text_source[name] for name in text_fields}
    expert = {
        "hidden_size": generator["generator_hidden_size"],
        "num_hidden_layers": diffusion["num_layers"],
        "num_attention_heads": diffusion["num_attention_heads"],
        "num_key_value_heads": diffusion["num_key_value_heads"],
        "head_dim": diffusion["attention_head_dim"],
        "intermediate_size": generator["generator_intermediate_size"],
        "rms_norm_eps": text["rms_norm_eps"],
        "rope_theta": text["rope_theta"],
        "max_position_embeddings": text["max_position_embeddings"],
        "use_adarms": True,
        "adarms_cond_dim": generator["generator_hidden_size"],
    }
    for name in (
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    ):
        require_equal(expert[name], text[name], f"expert.{name}")
    require_equal(
        vision["projection_dim"], text["hidden_size"], "vision.projection_dim"
    )
    require_equal(
        vision_source.get("projection_dim", vision["projection_dim"]),
        vision["projection_dim"],
        "vision_config.projection_dim",
    )
    require_equal(
        generator.get("hidden_size", expert["hidden_size"]),
        expert["hidden_size"],
        "generator.hidden_size",
    )
    require_equal(
        diffusion["output_dim"], expert["hidden_size"], "diffusion.output_dim"
    )
    require_equal(
        diffusion["cross_attention_dim"],
        text["hidden_size"],
        "diffusion.cross_attention_dim",
    )
    require_equal(
        generator["action_horizon"],
        source.get("action_horizon", generator["action_horizon"]),
        "action_horizon",
    )
    require_equal(
        generator["action_dim"], generator["max_action_dim"], "max_action_dim"
    )
    require_equal(
        vision["hidden_size"] % vision["num_attention_heads"],
        0,
        "vision head divisibility",
    )
    require_equal(
        vision["image_size"] % vision["patch_size"], 0, "image patch divisibility"
    )
    require_equal(
        text["num_attention_heads"] % text["num_key_value_heads"],
        0,
        "text head divisibility",
    )
    require_equal(text["head_dim"] % 2, 0, "even RoPE head dimension")
    result = {
        "vision": vision,
        "text": text,
        "expert": expert,
        "chunk_size": generator["action_horizon"],
        "max_action_dim": generator["action_dim"],
        "num_inference_steps": generator["num_inference_timesteps"],
        "min_period": 4e-3,
        "max_period": 4.0,
        "tokenizer_max_length": source["max_token_len"],
    }
    for section in (vision, text, expert, result):
        for key, value in section.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value <= 0
            ):
                raise ValueError(f"Expected positive {key}, got {value}")
    return result


def expected_weight_shapes(config: dict[str, Any]) -> dict[str, list[int]]:
    """Describe the loader's complete tensor contract from explicit geometry."""
    result: dict[str, list[int]] = {}

    def linear(prefix: str, output: int, input_: int, bias: bool = False) -> None:
        result[prefix + ".weight"] = [output, input_]
        if bias:
            result[prefix + ".bias"] = [output]

    vision, text, expert = (config[name] for name in ("vision", "text", "expert"))
    width = vision["hidden_size"]
    vp = PALIGEMMA_PREFIX + "model.vision_tower.vision_model."
    result[vp + "embeddings.patch_embedding.weight"] = [
        width,
        vision["num_channels"],
        vision["patch_size"],
        vision["patch_size"],
    ]
    result[vp + "embeddings.patch_embedding.bias"] = [width]
    result[vp + "embeddings.position_embedding.weight"] = [
        (vision["image_size"] // vision["patch_size"]) ** 2,
        width,
    ]
    for index in range(vision["num_hidden_layers"]):
        prefix = f"{vp}encoder.layers.{index}."
        for norm in ("layer_norm1", "layer_norm2"):
            for parameter in ("weight", "bias"):
                result[prefix + norm + "." + parameter] = [width]
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(prefix + "self_attn." + projection, width, width, True)
        linear(prefix + "mlp.fc1", vision["intermediate_size"], width, True)
        linear(prefix + "mlp.fc2", width, vision["intermediate_size"], True)
    result[vp + "post_layernorm.weight"] = [width]
    result[vp + "post_layernorm.bias"] = [width]
    linear(
        PALIGEMMA_PREFIX + "model.multi_modal_projector.linear",
        vision["projection_dim"],
        width,
        True,
    )
    result[PALIGEMMA_PREFIX + "lm_head.weight"] = [
        text["vocab_size"],
        text["hidden_size"],
    ]
    for tower, prefix in (
        (text, PALIGEMMA_PREFIX + "model.language_model."),
        (expert, EXPERT_PREFIX),
    ):
        width = tower["hidden_size"]
        adaptive = tower is expert
        for index in range(tower["num_hidden_layers"]):
            layer = f"{prefix}layers.{index}."
            for norm in ("input_layernorm", "post_attention_layernorm"):
                if adaptive:
                    linear(layer + norm + ".dense", 3 * width, width, True)
                else:
                    result[layer + norm + ".weight"] = [width]
            for projection, heads in (
                ("q_proj", tower["num_attention_heads"]),
                ("k_proj", tower["num_key_value_heads"]),
                ("v_proj", tower["num_key_value_heads"]),
            ):
                linear(
                    layer + "self_attn." + projection, heads * tower["head_dim"], width
                )
            linear(
                layer + "self_attn.o_proj",
                width,
                tower["num_attention_heads"] * tower["head_dim"],
            )
            for projection in ("gate_proj", "up_proj"):
                linear(layer + "mlp." + projection, tower["intermediate_size"], width)
            linear(layer + "mlp.down_proj", width, tower["intermediate_size"])
        if adaptive:
            linear(prefix + "norm.dense", 3 * width, width, True)
        else:
            result[prefix + "norm.weight"] = [width]
    linear("action_in_proj", expert["hidden_size"], config["max_action_dim"], True)
    linear("action_out_proj", config["max_action_dim"], expert["hidden_size"], True)
    for projection in ("time_mlp_in", "time_mlp_out"):
        linear(projection, expert["hidden_size"], expert["hidden_size"], True)
    return result


@dataclass(frozen=True)
class TensorRecord:
    source_file: Path
    source_key: str
    target_key: str | None
    dtype: str
    shape: list[int]
    offset: int
    size: int


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        length = stream.read(8)
        if len(length) != 8:
            raise ValueError(f"Truncated safetensors header: {path.name}")
        size = struct.unpack("<Q", length)[0]
        if size > 100_000_000 or size > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header length: {path.name}")
        header = json.loads(stream.read(size), object_pairs_hook=unique_json_object)
    if not isinstance(header, dict):
        raise TypeError(f"Invalid safetensors header: {path.name}")
    return header, size + 8


def collect_tensors(
    source: Path, config: dict[str, Any]
) -> tuple[list[TensorRecord], list[Path]]:
    index_path = source / "model.safetensors.index.json"
    weight_map: dict[str, str] | None = None
    if index_path.is_file():
        weight_map = read_json(index_path)["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("The safetensors index has no weight map")
        names = sorted(set(weight_map.values()))
        if any(
            Path(name).name != name or not name.endswith(".safetensors")
            for name in names
        ):
            raise ValueError("Shard filenames must be local safetensors basenames")
        paths = [source / name for name in names]
    elif (source / "model.safetensors").is_file():
        paths = [source / "model.safetensors"]
    else:
        raise ValueError(
            "Expected a complete HF safetensors export; export FSDP training shards first"
        )
    expected = expected_weight_shapes(config)
    seen_source: set[str] = set()
    seen_target: set[str] = set()
    records: list[TensorRecord] = []
    for path in paths:
        header, data_start = read_safetensors_header(path)
        spans: list[tuple[int, int]] = []
        for key, metadata in header.items():
            if key == "__metadata__":
                continue
            if key in seen_source:
                raise ValueError(f"Duplicate source tensor: {key}")
            seen_source.add(key)
            if weight_map is not None and weight_map.get(key) != path.name:
                raise ValueError(f"Index/shard mismatch for {key}")
            dtype, shape = metadata["dtype"], metadata["shape"]
            if dtype not in DTYPE_BYTES or any(
                type(dim) is not int or dim < 0 for dim in shape
            ):
                raise ValueError(f"Unsupported tensor metadata: {key}")
            start, end = metadata["data_offsets"]
            size = math.prod(shape) * DTYPE_BYTES[dtype]
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end - start != size
            ):
                raise ValueError(f"Invalid tensor data offsets: {key}")
            spans.append((start, end))
            target = enactive_pi05_weight_remap(key)
            if target is not None:
                if target in seen_target:
                    raise ValueError(f"Duplicate destination tensor: {target}")
                if expected.get(target) != shape:
                    raise ValueError(
                        f"Unexpected tensor or shape for {key}: {shape}, expected {expected.get(target)}"
                    )
                if dtype not in {"BF16", "F16", "F32"}:
                    raise ValueError(f"Non-floating inference weight: {key}")
                seen_target.add(target)
            records.append(
                TensorRecord(path, key, target, dtype, shape, data_start + start, size)
            )
        offset = 0
        for start, end in sorted(spans):
            if start != offset:
                raise ValueError(f"Overlapping tensors or data holes: {path.name}")
            offset = end
        if offset + data_start != path.stat().st_size:
            raise ValueError(f"Truncated or trailing safetensors data: {path.name}")
    if weight_map is not None and seen_source != set(weight_map):
        raise ValueError("The safetensors index lists missing tensors")
    missing = sorted(set(expected) - seen_target)
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required pi0.5 weights: {missing[:5]}"
        )
    return records, paths


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_tensor(record: TensorRecord, output: BinaryIO) -> str:
    digest = hashlib.sha256()
    remaining = record.size
    with record.source_file.open("rb") as stream:
        stream.seek(record.offset)
        while remaining:
            data = stream.read(min(remaining, COPY_BLOCK_BYTES))
            if not data:
                raise ValueError(f"Truncated tensor: {record.source_key}")
            digest.update(data)
            output.write(data)
            remaining -= len(data)
    return digest.hexdigest()


def config_differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            field = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                result.append(
                    {
                        "field": field,
                        "source_present": key in left,
                        "reference_present": key in right,
                        "source": left.get(key),
                        "reference": right.get(key),
                    }
                )
            else:
                result.extend(config_differences(left[key], right[key], field))
        return result
    return (
        [] if left == right else [{"field": path, "source": left, "reference": right}]
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def convert_enactive_pi05(
    source: str | Path,
    output: str | Path,
    *,
    reference_config: str | Path | None = None,
    max_shard_bytes: int = 4_000_000_000,
) -> dict[str, Any]:
    """Write a new checkpoint and audit manifest, preserving each tensor exactly.

    This converts storage and geometry. The report records numerical contracts;
    loading successfully alone does not establish Enactive inference parity.
    Existing destinations are rejected, including empty directories.
    """
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Destination already exists: {output}")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    original = read_json(source / "config.json")
    chosen_path = (
        Path(reference_config).resolve()
        if reference_config is not None
        else source / "config.json"
    )
    chosen = read_json(chosen_path)
    config = build_phyai_config(chosen)
    if build_phyai_config(original) != config:
        raise ValueError("Reference config changes checkpoint geometry")
    for name in SIDECARS:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Required Enactive sidecar is missing: {name}")
    manifest = read_json(source / "checkpoint_manifest.json")
    runtime = manifest.get("runtime", {})
    if "action_horizon" in runtime:
        require_equal(
            runtime["action_horizon"], config["chunk_size"], "manifest action_horizon"
        )
    records, source_shards = collect_tensors(source, config)
    active = [record for record in records if record.target_key is not None]
    groups: list[list[TensorRecord]] = []
    current: list[TensorRecord] = []
    size = 0
    for record in active:
        if current and size + record.size > max_shard_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(record)
        size += record.size
    if current:
        groups.append(current)
    differences = config_differences(original, chosen)
    notes = [
        "Tensor bytes are preserved; source storage dtype does not specify module compute precision.",
        "The gzq/cobot-pi05 branch uses reference_bf16_tanh; later Enactive versions honor mlp_activation_implementation.",
        "PhyAI's default exact GELU and head precision must be aligned separately before claiming inference parity.",
    ]
    if original.get("mlp_activation_implementation") == "fused_tanh":
        notes.append(
            "The export explicitly selects fused_tanh, unlike the training branch's reference_bf16_tanh behavior. The original config is preserved unchanged."
        )
    report: dict[str, Any] = {
        "schema": "phyai_enactive_pi05_conversion",
        "version": 1,
        "source_config_sha256": hash_file(source / "config.json"),
        "reference_config_sha256": hash_file(chosen_path),
        "reference_config_supplied": reference_config is not None,
        "config_differences": differences,
        "source_mlp_activation_implementation": original.get(
            "mlp_activation_implementation"
        ),
        "reference_mlp_activation_implementation": chosen.get(
            "mlp_activation_implementation", "reference_bf16_tanh"
        ),
        "source_tensor_count": len(records),
        "mapped_tensor_count": len(active),
        "dropped_tensor_count": len(records) - len(active),
        "dtype_counts": dict(Counter(record.dtype for record in active)),
        "global_step": manifest.get("global_step"),
        "notes": notes,
        "source_files": {
            path.name: {"sha256": hash_file(path), "size_bytes": path.stat().st_size}
            for path in [
                source / "config.json",
                *source_shards,
                *(source / name for name in SIDECARS),
            ]
        },
        "tensors": [],
        "dropped_tensors": [],
        "output_files": {},
    }
    for record in records:
        if record.target_key is None:
            report["dropped_tensors"].append(
                {
                    "source_key": record.source_key,
                    "source_file": record.source_file.name,
                    "dtype": record.dtype,
                    "shape": record.shape,
                    "reason": DROP_REASONS[
                        record.source_key.removeprefix(GENERATOR_PREFIX)
                    ],
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        index: dict[str, str] = {}
        for shard_number, group in enumerate(groups, 1):
            name = (
                "model.safetensors"
                if len(groups) == 1
                else f"model-{shard_number:05d}-of-{len(groups):05d}.safetensors"
            )
            header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
            offset = 0
            for record in group:
                header[record.target_key] = {
                    "dtype": record.dtype,
                    "shape": record.shape,
                    "data_offsets": [offset, offset + record.size],
                }
                offset += record.size
            encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
            encoded += b" " * (-len(encoded) % 8)
            with (staging / name).open("wb") as stream:
                stream.write(struct.pack("<Q", len(encoded)))
                stream.write(encoded)
                for record in group:
                    digest = copy_tensor(record, stream)
                    index[record.target_key] = name
                    report["tensors"].append(
                        {
                            "source_key": record.source_key,
                            "target_key": record.target_key,
                            "source_file": record.source_file.name,
                            "output_file": name,
                            "dtype": record.dtype,
                            "shape": record.shape,
                            "size_bytes": record.size,
                            "sha256": digest,
                        }
                    )
            actual_header, data_start = read_safetensors_header(staging / name)
            with (staging / name).open("rb") as stream:
                for item in report["tensors"][-len(group) :]:
                    start, end = actual_header[item["target_key"]]["data_offsets"]
                    stream.seek(data_start + start)
                    remaining, digest = end - start, hashlib.sha256()
                    while remaining:
                        data = stream.read(min(remaining, COPY_BLOCK_BYTES))
                        if not data:
                            raise ValueError("Truncated converted checkpoint")
                        digest.update(data)
                        remaining -= len(data)
                    if digest.hexdigest() != item["sha256"]:
                        raise ValueError(
                            f"Tensor byte verification failed: {item['target_key']}"
                        )
            report["output_files"][name] = {
                "sha256": hash_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
        if len(groups) > 1:
            write_json(
                staging / "model.safetensors.index.json",
                {
                    "metadata": {"total_size": sum(record.size for record in active)},
                    "weight_map": index,
                },
            )
        write_json(staging / "config.json", config)
        shutil.copyfile(source / "config.json", staging / "enactive_config.json")
        if reference_config is not None:
            shutil.copyfile(chosen_path, staging / "enactive_reference_config.json")
        for name in SIDECARS:
            shutil.copyfile(source / name, staging / name)
        for path in sorted(staging.iterdir()):
            if path.name not in report["output_files"]:
                report["output_files"][path.name] = {
                    "sha256": hash_file(path),
                    "size_bytes": path.stat().st_size,
                }
        report["all_tensor_bytes_verified"] = True
        write_json(staging / "conversion_report.json", report)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Destination appeared during conversion: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Complete Enactive HF export directory"
    )
    parser.add_argument("output", type=Path, help="New phyai checkpoint directory")
    parser.add_argument(
        "--reference-config",
        type=Path,
        help="Explicit original/reference config; preserved and audited separately",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=4_000_000_000)
    args = parser.parse_args(argv)
    report = convert_enactive_pi05(
        args.source,
        args.output,
        reference_config=args.reference_config,
        max_shard_bytes=args.max_shard_bytes,
    )
    parser.exit(
        message=f"Converted {report['mapped_tensor_count']} weights; dropped {report['dropped_tensor_count']} unused tensors. All tensor bytes verified. See conversion_report.json.\n"
    )


if __name__ == "__main__":
    main()
