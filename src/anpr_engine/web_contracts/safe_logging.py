"""Fail-closed validation for planned structured operational log records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

ALLOWED_FIELDS: Final = frozenset(
    {
        "timestamp",
        "severity",
        "request_id",
        "route_category",
        "status_code",
        "error_code",
        "stage_timings_ms",
        "capacity_state",
        "application_version",
        "api_version",
        "schema_version",
        "model_versions",
        "health_state",
        "readiness_state",
    }
)


class SafeLogContractError(ValueError):
    """A record attempted to cross the approved logging boundary."""


def validate_safe_log_record(
    record: Mapping[str, object], *, canary_values: tuple[str, ...] = ()
) -> None:
    """Reject unapproved fields or sentinel sensitive values before logging."""

    unexpected = set(record) - ALLOWED_FIELDS
    if unexpected:
        raise SafeLogContractError("unapproved_log_field")
    serialized = json.dumps(record, sort_keys=True, ensure_ascii=False)
    if any(canary and canary in serialized for canary in canary_values):
        raise SafeLogContractError("sensitive_log_canary")
