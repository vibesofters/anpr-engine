"""Model-independent recognizer interface and PyTorch boundary adapter."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import torch
from torch import nn

from anpr_engine.domain import ArtifactIdentity, RecognitionResult, ResultStatus
from anpr_engine.recognition.adapters import adapt_ctc_logits
from anpr_engine.recognition.charset import PlateCharset
from anpr_engine.recognition.crop import PlateCrop
from anpr_engine.recognition.ctc import greedy_decode_logits
from anpr_engine.recognition.decoder import ProbabilityAwareTrDecoder, ProfiledRecognitionResult
from anpr_engine.recognition.enhanced_model import EnhancedRecognizerOutput
from anpr_engine.recognition.evidence import CommonRecognitionOutput
from anpr_engine.recognition.registry import M0_CAPABILITY

GREEDY_DECODER_VERSION = "ctc-greedy-v1"


class Recognizer(Protocol):
    def recognize(self, crop: PlateCrop) -> RecognitionResult: ...


class PyTorchRecognizer:
    """Adapter for a trained compatible model; foundation models remain untrained."""

    def __init__(
        self,
        model: nn.Module,
        *,
        charset: PlateCharset,
        identity: ArtifactIdentity,
        device: torch.device,
    ) -> None:
        self._model = model.to(device)
        self._charset = charset
        self._identity = identity
        self._device = device

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        image = crop.prepared_rgb
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("recognizer crop must be prepared RGB uint8 HWC")
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
            .to(device=self._device, dtype=torch.float32)
            .div(255.0)
            .unsqueeze(0)
        )
        self._model.eval()
        with torch.inference_mode():
            logits = cast(torch.Tensor, self._model(tensor))
        raw_text = greedy_decode_logits(logits, self._charset)[0]
        return RecognitionResult(
            raw_text=raw_text,
            recognition_confidence=None,
            recognizer=self._identity,
            decoder_version=GREEDY_DECODER_VERSION,
            charset_version=self._charset.version,
            preprocessing_signature=crop.reference.preprocessing_signature,
        )


class EnhancedPyTorchRecognizer:
    """Adapter preserving raw and probability-aware profile-decoded evidence."""

    def __init__(
        self,
        model: nn.Module,
        *,
        charset: PlateCharset,
        identity: ArtifactIdentity,
        device: torch.device,
        decoder: ProbabilityAwareTrDecoder,
    ) -> None:
        self._model = model.to(device)
        self._charset = charset
        self._identity = identity
        self._device = device
        self._decoder = decoder

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        common = self.recognize_common_output(crop)
        decoded = self._decoder.decode_common(common)
        return recognition_result_from_profiled(
            decoded,
            identity=self._identity,
            charset_version=self._charset.version,
            preprocessing_signature=crop.reference.preprocessing_signature,
        )

    def recognize_common_output(self, crop: PlateCrop) -> CommonRecognitionOutput:
        """Expose M0's immutable common evidence without changing public result JSON."""
        image = crop.prepared_rgb
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("recognizer crop must be prepared RGB uint8 HWC")
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
            .to(device=self._device, dtype=torch.float32)
            .div(255.0)
            .unsqueeze(0)
        )
        self._model.eval()
        with torch.inference_mode():
            output = cast(EnhancedRecognizerOutput, self._model(tensor))
        return adapt_ctc_logits(
            output.character_logits[:, 0, :],
            output.type_logits[:, 0, :],
            capability=M0_CAPABILITY,
            character_charset=self._charset,
            decoder_config=self._decoder.config,
        )


def recognition_result_from_profiled(
    decoded: ProfiledRecognitionResult,
    *,
    identity: ArtifactIdentity,
    charset_version: str,
    preprocessing_signature: str,
) -> RecognitionResult:
    """Keep the established public RecognitionResult boundary additive-free in Phase 0C."""
    return RecognitionResult(
        raw_text=decoded.raw_text,
        recognition_confidence=decoded.raw_confidence,
        decoded_text=decoded.decoded_text,
        decoder_confidence=decoded.decoder_confidence,
        profile_id=decoded.profile_id,
        decoding_changed_raw=decoded.decoding_changed_raw,
        structural_validation=(
            decoded.structural_validation.model_dump(mode="json")
            if decoded.structural_validation is not None
            else None
        ),
        decoder_status=(
            "ACCEPTED" if decoded.status is ResultStatus.ACCEPTED else "LOW_CONFIDENCE"
        ),
        decoder_reason=decoded.reason,
        recognizer=identity,
        decoder_version=decoded.decoder_version,
        charset_version=charset_version,
        preprocessing_signature=preprocessing_signature,
    )


class StaticRecognizer:
    """Deterministic test/integration stub; never an OCR implementation."""

    def __init__(
        self,
        raw_text: str,
        *,
        confidence: float | None,
        identity: ArtifactIdentity,
        charset_version: str,
    ) -> None:
        self._raw_text = raw_text
        self._confidence = confidence
        self._identity = identity
        self._charset_version = charset_version

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        return RecognitionResult(
            raw_text=self._raw_text,
            recognition_confidence=self._confidence,
            recognizer=self._identity,
            decoder_version="deterministic-static-stub-v1",
            charset_version=self._charset_version,
            preprocessing_signature=crop.reference.preprocessing_signature,
        )
