"""CTC target construction and deterministic greedy decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from anpr_engine.recognition.charset import PlateCharset


@dataclass(frozen=True)
class CtcTargets:
    values: Tensor
    lengths: Tensor


def encode_transcriptions(texts: tuple[str, ...], charset: PlateCharset) -> CtcTargets:
    if not texts:
        raise ValueError("at least one transcription is required")
    encoded = tuple(charset.encode(text) for text in texts)
    if any(not item for item in encoded):
        raise ValueError("CTC targets cannot contain empty transcriptions")
    flattened = [index for item in encoded for index in item]
    return CtcTargets(
        values=torch.tensor(flattened, dtype=torch.long),
        lengths=torch.tensor([len(item) for item in encoded], dtype=torch.long),
    )


def collapse_ctc_indexes(indexes: tuple[int, ...], charset: PlateCharset) -> tuple[int, ...]:
    collapsed: list[int] = []
    previous: int | None = None
    for index in indexes:
        if index < 0 or index >= charset.class_count:
            raise ValueError(f"invalid CTC class index: {index}")
        if index != previous and index != charset.blank_index:
            collapsed.append(index)
        previous = index
    return tuple(collapsed)


def greedy_decode_indexes(
    indexes: Tensor,
    charset: PlateCharset,
    *,
    lengths: Tensor | None = None,
) -> tuple[str, ...]:
    """Decode a time-major integer tensor with shape ``[time, batch]``."""
    if indexes.ndim != 2:
        raise ValueError("CTC indexes must have shape [time, batch]")
    if indexes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("CTC indexes must use an integer dtype")
    time_steps, batch_size = indexes.shape
    if time_steps < 1 or batch_size < 1:
        raise ValueError("CTC index tensor must be non-empty")
    if lengths is None:
        resolved_lengths = (time_steps,) * batch_size
    else:
        if lengths.ndim != 1 or lengths.numel() != batch_size:
            raise ValueError("CTC lengths must have shape [batch]")
        resolved_lengths = tuple(int(value) for value in lengths.detach().cpu().tolist())
        if any(length < 0 or length > time_steps for length in resolved_lengths):
            raise ValueError("CTC lengths must remain within the time axis")

    cpu_indexes = indexes.detach().cpu()
    decoded: list[str] = []
    for batch_index, length in enumerate(resolved_lengths):
        raw = tuple(int(value) for value in cpu_indexes[:length, batch_index].tolist())
        decoded.append(charset.decode_indices(collapse_ctc_indexes(raw, charset)))
    return tuple(decoded)


def greedy_decode_logits(
    logits: Tensor,
    charset: PlateCharset,
    *,
    lengths: Tensor | None = None,
) -> tuple[str, ...]:
    if logits.ndim != 3:
        raise ValueError("CTC logits must have shape [time, batch, classes]")
    if logits.shape[2] != charset.class_count:
        raise ValueError(f"expected {charset.class_count} CTC classes, got {logits.shape[2]}")
    if not logits.is_floating_point():
        raise ValueError("CTC logits must use a floating-point dtype")
    return greedy_decode_indexes(logits.argmax(dim=2), charset, lengths=lengths)
