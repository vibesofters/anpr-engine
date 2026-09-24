"""Typed configuration loading and identity generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

REDACTION_MARKER = "<machine-path>"


class ConfigurationError(ValueError):
    """Raised when configuration cannot be loaded or validated."""


class DevicePolicy(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathSettings(StrictConfigModel):
    dataset_root: Path | None = None
    artifact_root: Path = Path("artifacts")
    model_root: Path | None = None


class RuntimeSettings(StrictConfigModel):
    device: DevicePolicy = DevicePolicy.CPU
    log_level: LogLevel = LogLevel.INFO


class EngineConfig(StrictConfigModel):
    schema_version: Literal["1.0"] = "1.0"
    paths: PathSettings = Field(default_factory=PathSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


class ResolvedConfig(StrictConfigModel):
    config: EngineConfig
    redacted_snapshot: dict[str, Any]
    config_hash: str


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigurationError("YAML mapping keys must be strings")
        if key in mapping:
            raise ConfigurationError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


_ENVIRONMENT_OVERRIDES = {
    "ANPR_DATASET_ROOT": "paths.dataset_root",
    "ANPR_ARTIFACT_ROOT": "paths.artifact_root",
    "ANPR_MODEL_ROOT": "paths.model_root",
    "ANPR_DEVICE": "runtime.device",
    "ANPR_LOG_LEVEL": "runtime.log_level",
}


def load_config(
    profile_path: Path,
    *,
    local_override_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> ResolvedConfig:
    """Load and validate one profile using the documented precedence order."""
    profile = _read_yaml(profile_path)
    if "schema_version" not in profile:
        raise ConfigurationError("The base profile must declare schema_version")

    merged = _deep_merge({}, profile)
    if local_override_path is not None:
        merged = _deep_merge(merged, _read_yaml(local_override_path))

    environment = os.environ if environ is None else environ
    environment_values = {
        path: environment[name]
        for name, path in _ENVIRONMENT_OVERRIDES.items()
        if name in environment
    }
    merged = _apply_dotted_overrides(merged, environment_values)
    merged = _apply_dotted_overrides(merged, cli_overrides or {})

    try:
        config = EngineConfig.model_validate(merged)
    except ValidationError as error:
        raise ConfigurationError(str(error)) from error

    redacted = config.model_dump(mode="json")
    redacted["paths"] = {
        name: REDACTION_MARKER if value is not None else None
        for name, value in redacted["paths"].items()
    }
    identity = config.model_dump(mode="json", exclude={"paths"})
    canonical = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return ResolvedConfig(
        config=config,
        redacted_snapshot=redacted,
        config_hash=hashlib.sha256(canonical).hexdigest(),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"Cannot read configuration profile: {path}") from error

    try:
        loaded = yaml.load(content, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML profile: {path}") from error
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"Configuration profile must be a mapping: {path}")
    return loaded


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_dotted_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    for dotted_path, value in overrides.items():
        if not isinstance(dotted_path, str) or not dotted_path:
            raise ConfigurationError("Override names must be non-empty dotted strings")
        parts = dotted_path.split(".")
        target = result
        for part in parts[:-1]:
            nested = target.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ConfigurationError(f"Override conflicts with scalar field: {dotted_path}")
            target = nested
        target[parts[-1]] = value
    return result
