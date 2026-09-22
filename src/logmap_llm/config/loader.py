"""
logmap_llm.config.loader — Config loading, validation, and display utilities.
Loads TOML config, applies CLI overrides, validates via Pydantic schema.
"""
from __future__ import annotations

import sys
import os
import tomllib
from pathlib import Path
from typing import Any

from logmap_llm.utils.logging import error, info, success
from logmap_llm.config.schema import validate_config, LogMapLLMConfig
from pydantic import ValidationError


def load_config(
    config_path: str | os.PathLike[str],
    *,
    reuse_align: bool = False,
    reuse_prompts: bool = False,
) -> LogMapLLMConfig:
    """Load and validate a TOML config without terminating the interpreter.

    Public library seam for experiment planners and notebooks. File, TOML, and
    Pydantic validation errors propagate to the caller; CLI-facing code may
    catch and render them separately.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")

    with path.open(mode="rb") as fp:
        config = tomllib.load(fp)

    pipeline = config.setdefault("pipeline", {})
    if reuse_align:
        pipeline["align_ontologies"] = "reuse"
    if reuse_prompts:
        pipeline["build_oracle_prompts"] = "reuse"
        pipeline["align_ontologies"] = "reuse"

    return validate_config(config)


def load_and_validate_config(
    config_path: str,
    reuse_align: bool = False,
    reuse_prompts: bool = False,
    reporting: bool = True,
) -> LogMapLLMConfig:
    """CLI-compatible loader that renders errors and exits with status 1."""
    if reuse_align:
        info("--reuse-align set, overriding config (reuse init align)")
    if reuse_prompts:
        info("--reuse-prompts set, overriding config (reuse prompts+align)")
    try:
        cfg = load_config(
            config_path,
            reuse_align=reuse_align,
            reuse_prompts=reuse_prompts,
        )
    except ValidationError as e:
        error(f"configuration file is invalid: {config_path}")
        for err in e.errors():
            field = " x ".join(str(loc) for loc in err["loc"])
            error(f" {field}: {err['msg']}")
        sys.exit(1)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError) as e:
        error(str(e))
        sys.exit(1)

    if reporting:
        success(f"configuration validated: {config_path}")

    return cfg


def inspect_and_mask_api_key(key: str) -> str:
    """Mask an API key for display, showing only first and last 4 chars"""
    if key is None: # guards against None
        return "<unset>"
    if key == 'EMPTY':
        return key
    return f"{key[:4]} ... {key[-4:]}" if len(key) > 8 else "***"


def parse_config_into_list(
    config_dict: dict, key_prefix: str = ""
) -> list[tuple[str, Any]]:
    """Recursively flatten a nested config dict into (dotted-key, value) pairs."""
    if not isinstance(config_dict, dict):
        return [(key_prefix, config_dict)]
    if not config_dict and key_prefix:
        return [(key_prefix, {})] # anticipate empty dicts
    kv_config_pairs = []
    for key, value in config_dict.items():
        extended_key = f"{key_prefix}.{key}" if key_prefix else key
        kv_config_pairs.extend(parse_config_into_list(value, extended_key))
    return kv_config_pairs


def print_config_summary(cfg: LogMapLLMConfig) -> None:
    """Print a human-readable summary of the configuration."""
    flat_config_params: list = parse_config_into_list(cfg.model_dump())
    expr_params_str: str = "Summary of Experiment Parameters:\n\n"
    for key, value in flat_config_params:
        if key == "oracle.api_key":
            value = inspect_and_mask_api_key(value)
        elif value is None:
            value = "<unset>"
        expr_params_str += f"{key}: {value}\n"
    info(expr_params_str)
