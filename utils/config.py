from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, MutableMapping

import yaml


def _parse_value(raw_value: str) -> Any:
    """Convert a CLI override string into a Python object when possible."""
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return raw_value


def _set_nested(mapping: MutableMapping[str, Any], key_path: list[str], value: Any) -> None:
    """Set a value into a nested mapping, creating intermediate dictionaries as needed."""
    current = mapping
    for key in key_path[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]
    current[key_path[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    """
    Load a YAML configuration file and apply CLI-style overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Iterable of strings in the form \"key.subkey=value\".

    Returns:
        Parsed configuration as a dictionary with applied overrides.
    """
    config_path = Path(path)
    with config_path.open(\"r\", encoding=\"utf-8\") as f:
        config = yaml.safe_load(f) or {}

    if overrides:
        for override in overrides:
            if \"=\" not in override:
                raise ValueError(f\"Invalid override '{override}'. Expected format key.subkey=value\")
            key_str, raw_value = override.split(\"=\", 1)
            keys = [k for k in key_str.split(\".\") if k]
            if not keys:
                raise ValueError(f\"Invalid override '{override}'. No keys provided.\")
            value = _parse_value(raw_value)
            _set_nested(config, keys, value)

    return config
