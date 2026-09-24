"""Allowlisted JSON event logging with no request-derived values."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import TextIO

from anpr_engine.web_contracts.safe_logging import validate_safe_log_record


class SafeJsonLogger:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def emit(self, **fields: object) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            **fields,
        }
        validate_safe_log_record(record)
        self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._stream.flush()
