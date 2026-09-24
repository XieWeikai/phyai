"""Reusable structured-YAML loading with dotted command-line overrides.

Callers supply their own dataclass schema, default YAML, and optional semantic
validator. This module knows no robot, model, control mode, or action layout.
Schema field types are checked by OmegaConf before the caller's validator runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Callable, Sequence

from yaml import YAMLError
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_path(value: str) -> Path:
    """Resolve an example's runtime path against the checkout, expanding '~'.

    Runtime paths do not change meaning when a YAML file is moved to another
    directory. The --config path itself still resolves from the working directory.
    """
    return (REPOSITORY_ROOT / Path(value).expanduser()).resolve()


def parse_config[ConfigT](
    schema: type[ConfigT],
    *,
    default_config: Path,
    argv: Sequence[str] | None = None,
    validate: Callable[[ConfigT], None] | None = None,
) -> ConfigT:
    """Merge schema < default YAML < selected YAML < dotted CLI overrides.

    Validation is supplied by the entry point, keeping device/model constraints
    out of this loader. --print-config permits missing required values and skips
    semantic validation, so it can inspect configuration without creating any
    runtime resources. Normal parsing requires all mandatory fields and runs
    validation before returning the fully typed dataclass object.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="YAML configuration (may override only selected defaults)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged YAML and exit without creating runtime resources",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="Typed dotted overrides; field names come from the selected schema",
    )
    args = parser.parse_args(argv)
    try:
        if any("=" not in item or not item.split("=", 1)[0] for item in args.overrides):
            raise ValueError("overrides must use key=value syntax")
        merged = OmegaConf.merge(
            OmegaConf.structured(schema),
            OmegaConf.load(default_config),
            OmegaConf.load(args.config.expanduser()),
            OmegaConf.from_dotlist(list(args.overrides)),
        )
        if args.print_config:
            print(OmegaConf.to_yaml(merged, resolve=True), end="")
            parser.exit()
        config = OmegaConf.to_object(merged)
        assert isinstance(config, schema)
        if validate is not None:
            validate(config)
        return config
    except (OSError, ValueError, YAMLError, OmegaConfBaseException) as error:
        parser.error(str(error))
