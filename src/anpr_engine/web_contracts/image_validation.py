"""Bounded, in-memory validation for the Phase 2A raw-image contract.

This module is transport- and framework-independent. It does not persist,
process, infer from, or log image content.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Final

from PIL import Image, UnidentifiedImageError

MAX_COMPRESSED_BYTES: Final = 10 * 1024 * 1024
MAX_DECODED_PIXELS: Final = 20_000_000
MAX_DIMENSION: Final = 8_192
ALLOWED_MEDIA_TYPES: Final = frozenset({"image/jpeg", "image/png"})


class ImageContractError(ValueError):
    """Safe validation failure carrying only a stable public error code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ValidatedImageInfo:
    """Non-identifying facts allowed to cross the validation boundary."""

    media_type: str
    width: int
    height: int
    compressed_size_bytes: int


def _signature_media_type(payload: bytes) -> str | None:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return None


def _validate_png_boundary(payload: bytes) -> None:
    offset = 8
    while offset + 12 <= len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        offset += 12 + length
        if offset > len(payload):
            raise ImageContractError("IMAGE_TRUNCATED")
        if chunk_type == b"IEND":
            if length != 0:
                raise ImageContractError("IMAGE_DECODE_FAILED")
            if offset != len(payload):
                raise ImageContractError("IMAGE_MULTIFRAME_REJECTED")
            return
    raise ImageContractError("IMAGE_TRUNCATED")


def _validate_jpeg_boundary(payload: bytes) -> None:
    offset = 2
    in_scan = False
    while offset < len(payload):
        if payload[offset] != 0xFF:
            if not in_scan:
                raise ImageContractError("IMAGE_DECODE_FAILED")
            offset += 1
            continue
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            raise ImageContractError("IMAGE_TRUNCATED")
        marker = payload[offset]
        offset += 1
        if in_scan and (marker == 0x00 or 0xD0 <= marker <= 0xD7):
            continue
        if marker == 0xD9:
            if offset != len(payload):
                raise ImageContractError("IMAGE_MULTIFRAME_REJECTED")
            return
        if marker == 0xD8 or marker == 0x00:
            raise ImageContractError("IMAGE_DECODE_FAILED")
        if marker == 0x01:
            continue
        if offset + 2 > len(payload):
            raise ImageContractError("IMAGE_TRUNCATED")
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(payload):
            raise ImageContractError("IMAGE_TRUNCATED")
        offset += segment_length
        in_scan = marker == 0xDA
    raise ImageContractError("IMAGE_TRUNCATED")


def validate_image_dimensions(width: int, height: int, frame_count: int = 1) -> None:
    """Apply decoded size and single-frame limits without allocating an image."""

    if frame_count != 1:
        raise ImageContractError("IMAGE_MULTIFRAME_REJECTED")
    if width < 1 or height < 1:
        raise ImageContractError("IMAGE_DECODE_FAILED")
    if width > MAX_DIMENSION or height > MAX_DIMENSION or width * height > MAX_DECODED_PIXELS:
        raise ImageContractError("IMAGE_DIMENSIONS_EXCEEDED")


def validate_encoded_image(payload: bytes, declared_media_type: str) -> ValidatedImageInfo:
    """Validate exactly one JPEG/PNG held in caller-controlled memory."""

    if declared_media_type not in ALLOWED_MEDIA_TYPES:
        raise ImageContractError("UNSUPPORTED_MEDIA_TYPE")
    if not payload:
        raise ImageContractError("INVALID_REQUEST")
    if len(payload) > MAX_COMPRESSED_BYTES:
        raise ImageContractError("IMAGE_TOO_LARGE")
    detected_media_type = _signature_media_type(payload)
    if detected_media_type is None or detected_media_type != declared_media_type:
        raise ImageContractError("UNSUPPORTED_MEDIA_TYPE")
    if declared_media_type == "image/jpeg":
        _validate_jpeg_boundary(payload)
    else:
        _validate_png_boundary(payload)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                expected_format = "JPEG" if declared_media_type == "image/jpeg" else "PNG"
                if image.format != expected_format:
                    raise ImageContractError("UNSUPPORTED_MEDIA_TYPE")
                frame_count = int(getattr(image, "n_frames", 1))
                validate_image_dimensions(image.width, image.height, frame_count)
                image.verify()
            with Image.open(BytesIO(payload)) as decoded:
                decoded.load()
    except ImageContractError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageContractError("IMAGE_DIMENSIONS_EXCEEDED") from None
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        message = str(exc).lower()
        code = (
            "IMAGE_TRUNCATED"
            if "truncated" in message or "broken data stream" in message
            else "IMAGE_DECODE_FAILED"
        )
        raise ImageContractError(code) from None
    return ValidatedImageInfo(
        media_type=detected_media_type,
        width=decoded.width,
        height=decoded.height,
        compressed_size_bytes=len(payload),
    )
