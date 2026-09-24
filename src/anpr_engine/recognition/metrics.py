"""Deterministic recognition-only metrics (not end-to-end ANPR metrics)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from anpr_engine.domain import ResultStatus


class RecognitionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int = Field(ge=0)
    exact_normalized_plate_accuracy: float = Field(ge=0.0, le=1.0)
    character_accuracy: float = Field(ge=0.0, le=1.0)
    character_error_rate: float = Field(ge=0.0)
    plate_length_accuracy: float = Field(ge=0.0, le=1.0)
    normalized_edit_distance: float = Field(ge=0.0, le=1.0)
    valid_plate_rate: float = Field(ge=0.0, le=1.0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)


class RecognitionErrorCounts(BaseModel):
    """Edit-operation ledger retained for post-training confusion analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    substitutions: int = Field(ge=0)
    insertions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    invalid_decoded_sequences: int = Field(ge=0)
    empty_decodes: int = Field(ge=0)


def edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for row, reference_character in enumerate(reference, start=1):
        current = [row]
        for column, prediction_character in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_character != prediction_character),
                )
            )
        previous = current
    return previous[-1]


def compute_recognition_error_counts(
    predictions: tuple[str, ...],
    targets: tuple[str, ...],
    *,
    valid: tuple[bool, ...] | None = None,
) -> RecognitionErrorCounts:
    """Classify Levenshtein operations without applying speculative substitutions."""
    if not targets or len(predictions) != len(targets):
        raise ValueError("predictions and non-empty targets must have equal length")
    if valid is not None and len(valid) != len(targets):
        raise ValueError("valid flags must match the metric population")
    substitutions = insertions = deletions = 0
    for prediction, target in zip(predictions, targets, strict=True):
        operations = _edit_operations(target, prediction)
        substitutions += operations["substitutions"]
        insertions += operations["insertions"]
        deletions += operations["deletions"]
    resolved_valid = valid if valid is not None else tuple(False for _ in targets)
    return RecognitionErrorCounts(
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        invalid_decoded_sequences=sum(not item for item in resolved_valid),
        empty_decodes=sum(not prediction for prediction in predictions),
    )


def _edit_operations(reference: str, prediction: str) -> dict[str, int]:
    """Deterministically recover one minimum-cost edit path with fixed tie order."""
    rows, columns = len(reference) + 1, len(prediction) + 1
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for row in range(1, rows):
        table[row][0] = (row, 0, 0, row)
    for column in range(1, columns):
        table[0][column] = (column, 0, column, 0)
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == prediction[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue
            substitution = table[row - 1][column - 1]
            insertion = table[row][column - 1]
            deletion = table[row - 1][column]
            choices = (
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (insertion[0] + 1, insertion[1], insertion[2] + 1, insertion[3]),
                (deletion[0] + 1, deletion[1], deletion[2], deletion[3] + 1),
            )
            table[row][column] = min(choices, key=lambda value: value[0])
    _, substitutions, insertions, deletions = table[-1][-1]
    return {
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
    }


def compute_recognition_metrics(
    predictions: tuple[str, ...],
    targets: tuple[str, ...],
    *,
    valid: tuple[bool, ...] | None = None,
    statuses: tuple[ResultStatus, ...] | None = None,
    predicted_lengths: tuple[int, ...] | None = None,
) -> RecognitionMetrics:
    if not targets or len(predictions) != len(targets):
        raise ValueError("predictions and non-empty targets must have equal length")
    if valid is not None and len(valid) != len(targets):
        raise ValueError("valid flags must match the metric population")
    if statuses is not None and len(statuses) != len(targets):
        raise ValueError("statuses must match the metric population")
    if predicted_lengths is not None and len(predicted_lengths) != len(targets):
        raise ValueError("predicted lengths must match the metric population")

    distances = tuple(
        edit_distance(target, prediction)
        for prediction, target in zip(predictions, targets, strict=True)
    )
    exact = sum(
        prediction == target for prediction, target in zip(predictions, targets, strict=True)
    )
    target_characters = sum(len(target) for target in targets)
    total_distance = sum(distances)
    cer = total_distance / target_characters if target_characters else 0.0
    normalized_distances = tuple(
        distance / max(len(target), len(prediction), 1)
        for distance, target, prediction in zip(distances, targets, predictions, strict=True)
    )
    resolved_valid = valid if valid is not None else tuple(False for _ in targets)
    resolved_statuses = statuses or tuple(ResultStatus.FAILED for _ in targets)
    resolved_lengths = predicted_lengths or tuple(len(prediction) for prediction in predictions)
    accepted = sum(status is ResultStatus.ACCEPTED for status in resolved_statuses)
    failures = sum(status is ResultStatus.FAILED for status in resolved_statuses)
    return RecognitionMetrics(
        sample_count=len(targets),
        exact_normalized_plate_accuracy=exact / len(targets),
        character_accuracy=max(0.0, 1.0 - cer),
        character_error_rate=cer,
        plate_length_accuracy=sum(
            length == len(target) for length, target in zip(resolved_lengths, targets, strict=True)
        )
        / len(targets),
        normalized_edit_distance=sum(normalized_distances) / len(normalized_distances),
        valid_plate_rate=sum(resolved_valid) / len(resolved_valid),
        accepted_count=accepted,
        rejected_count=len(targets) - accepted - failures,
        failure_count=failures,
    )
