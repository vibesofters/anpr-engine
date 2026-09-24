from pathlib import Path

import pytest

from anpr_engine.configuration import (
    REDACTION_MARKER,
    ConfigurationError,
    DevicePolicy,
    LogLevel,
    load_config,
)


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_precedence_is_defaults_profile_local_environment_cli(tmp_path: Path) -> None:
    profile = _write_yaml(
        tmp_path / "profile.yaml",
        'schema_version: "1.0"\nruntime:\n  log_level: WARNING\n',
    )
    local = _write_yaml(tmp_path / "local.yaml", "runtime:\n  device: cuda\n")

    resolved = load_config(
        profile,
        local_override_path=local,
        environ={"ANPR_DEVICE": "cpu", "ANPR_LOG_LEVEL": "ERROR"},
        cli_overrides={"runtime.log_level": "DEBUG"},
    )

    assert resolved.config.runtime.device is DevicePolicy.CPU
    assert resolved.config.runtime.log_level is LogLevel.DEBUG
    assert resolved.config.paths.artifact_root == Path("artifacts")


def test_machine_paths_are_redacted_and_excluded_from_identity(tmp_path: Path) -> None:
    profile = _write_yaml(tmp_path / "profile.yaml", 'schema_version: "1.0"\n')

    first = load_config(profile, environ={"ANPR_DATASET_ROOT": "/private/first"})
    second = load_config(profile, environ={"ANPR_DATASET_ROOT": "/private/second"})

    assert first.redacted_snapshot["paths"]["dataset_root"] == REDACTION_MARKER
    assert "/private/first" not in str(first.redacted_snapshot)
    assert first.config_hash == second.config_hash


def test_behavior_change_changes_config_identity(tmp_path: Path) -> None:
    profile = _write_yaml(tmp_path / "profile.yaml", 'schema_version: "1.0"\n')

    first = load_config(profile, environ={})
    second = load_config(profile, environ={}, cli_overrides={"runtime.device": "cuda"})

    assert first.config_hash != second.config_hash


@pytest.mark.parametrize(
    "content, expected",
    [
        ('schema_version: "1.0"\nunknown: true\n', "Extra inputs are not permitted"),
        ('schema_version: "1.0"\nruntime:\n  device: cpu\n  device: cuda\n', "Duplicate YAML key"),
        ("runtime:\n  device: cpu\n", "must declare schema_version"),
    ],
)
def test_invalid_profiles_fail(tmp_path: Path, content: str, expected: str) -> None:
    profile = _write_yaml(tmp_path / "invalid.yaml", content)

    with pytest.raises(ConfigurationError, match=expected):
        load_config(profile, environ={})
