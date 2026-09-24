"""Deterministic, governed Turkish-plate synthetic QA renderer.

This module is deliberately limited to a small, inspectable QA corpus.  It is
not a training-dataset builder and cannot silently make a large release. Text
is selected before rendering and written to explicit label files.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from anpr_engine.data.recognition_layout import (
    RecognitionLayoutRecord,
    sha256_file,
    sha256_text,
)
from anpr_engine.recognition.normalization import normalize_plate_text
from anpr_engine.recognition.validation import validate_turkish_plate

PROJECT_SYNTHETIC_GENERATOR_VERSION = "project-turkish-plate-synthetic-v1.1"
PROJECT_SYNTHETIC_GENERATOR_STATUS = "READY_FOR_OWNER_BUILD_AUTHORIZATION"
GENERATOR_TECHNICAL_STATUS = "READY_FOR_OWNER_BUILD_AUTHORIZATION"
ASSET_LICENSE_STATUS = "VERIFIED_FOR_CONFIGURED_PROJECT_USE"
SYNTHETIC_DATASET_BUILD_STATUS = "NOT_BUILT_PENDING_OWNER_AUTHORIZATION"
OWNER_TRAINING_AUTHORIZATION = "NOT_GRANTED"
HERSHEY_TRAINING_GRADE = "SUPPLEMENT_ONLY"
SYNTHETIC_QA_CORPUS_MAX_COUNT = 96

BARLOW_SOURCE_REVISION = "ec626514f79f831f1ab848a82114a0ce7e2d6372"
BARLOW_LICENSE = "SIL Open Font License 1.1"
SYNTHETIC_ASSET_ID = "barlow-condensed-v1.408-vendored"
SYNTHETIC_ASSET_LICENSE = BARLOW_LICENSE
HERSHEY_ASSET_ID = "opencv-hershey-simplex-vector-strokes"
# Turkish V1's project grammar intentionally excludes Q/W/X.
_LETTERS = "ABCDEFGHIJKLMNOPRSTUVYZ"
_DIFFICULTIES: tuple[Literal["clean", "easy", "medium", "hard"], ...] = (
    "clean",
    "easy",
    "medium",
    "hard",
)
ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class SyntheticRenderConfig:
    """Versioned conservative V1 approximations, not regulatory dimensional claims."""

    canvas_width: int = 520
    canvas_height: int = 110
    outer_margin_px: int = 6
    vertical_margin_px: int | None = None
    border_width_px: int = 3
    blue_band_width_px: int = 53
    text_left_margin_px: int = 72
    text_right_margin_px: int = 15
    character_spacing_px: int = 2
    group_spacing_px: int = 14
    max_perspective_offset_px: float = 4.0
    max_rotation_degrees: float = 2.2
    min_scale: float = 0.94
    max_scale: float = 1.0
    max_translation_px: float = 3.0
    min_brightness_alpha: float = 0.84
    max_brightness_alpha: float = 1.14
    max_brightness_beta: float = 20.0
    min_gamma: float = 0.84
    max_gamma: float = 1.16
    max_noise_stddev: float = 5.5
    max_shadow_strength: float = 0.18
    min_jpeg_quality: int = 76
    max_jpeg_quality_exclusive: int = 97
    scale_font_to_height: bool = False


DEFAULT_SYNTHETIC_RENDER_CONFIG = SyntheticRenderConfig()


@dataclass(frozen=True)
class SyntheticPlateText:
    sample_id: int
    normalized_text: str
    formatted_text: str
    seed: int
    attempt: int


@dataclass(frozen=True)
class FontVariant:
    asset_id: str
    filename: str
    weight: int
    sha256: str


_FONT_VARIANTS: tuple[FontVariant, ...] = (
    FontVariant(
        asset_id="barlow-condensed-semibold-v1.408",
        filename="BarlowCondensed-SemiBold.ttf",
        weight=600,
        sha256="7b619d14bc2327509a9ef32b0890f709626f7ecc9ff61191c2a4314c5499d2d9",
    ),
    FontVariant(
        asset_id="barlow-condensed-bold-v1.408",
        filename="BarlowCondensed-Bold.ttf",
        weight=700,
        sha256="e476562ec9c1e16cf16475895b511f08c804f438cc9a9f80a44ea50a0eeb5b65",
    ),
)


def format_turkish_plate(normalized_text: str) -> str:
    """Format a validated canonical target for renderer display only."""
    validation = validate_turkish_plate(normalized_text)
    if not validation.valid:
        raise ValueError(f"cannot format invalid Turkish plate: {normalized_text}")
    letter_end = 2
    while letter_end < len(normalized_text) and normalized_text[letter_end].isalpha():
        letter_end += 1
    return f"{normalized_text[:2]} {normalized_text[2:letter_end]} {normalized_text[letter_end:]}"


class TurkishPlateTextGenerator:
    """Stable seeded grammar with bounded attempts and duplicate-target control."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def generate(self, sample_id: int, *, reserved: set[str] | None = None) -> SyntheticPlateText:
        if sample_id < 1:
            raise ValueError("sample_id must be one-based")
        occupied = reserved if reserved is not None else set()
        for attempt in range(10_000):
            target = self._candidate(sample_id, attempt)
            if target not in occupied:
                occupied.add(target)
                return SyntheticPlateText(
                    sample_id=sample_id,
                    normalized_text=target,
                    formatted_text=format_turkish_plate(target),
                    seed=self.seed,
                    attempt=attempt,
                )
        raise RuntimeError("could not generate a unique Turkish plate target")

    def generate_many(self, count: int) -> tuple[SyntheticPlateText, ...]:
        if count < 1:
            raise ValueError("count must be positive")
        reserved: set[str] = set()
        return tuple(
            self.generate(sample_id, reserved=reserved) for sample_id in range(1, count + 1)
        )

    def _candidate(self, sample_id: int, attempt: int) -> str:
        digest = hashlib.sha256(f"{self.seed}:{sample_id}:{attempt}".encode("ascii")).digest()
        province = 1 + int.from_bytes(digest[:2], "big") % 81
        letter_length = 1 + digest[2] % 3
        serial_length = 2 + digest[3] % 3
        letters = "".join(
            _LETTERS[digest[4 + index] % len(_LETTERS)] for index in range(letter_length)
        )
        serial_value = int.from_bytes(digest[8:12], "big") % (10**serial_length)
        target = f"{province:02d}{letters}{serial_value:0{serial_length}d}"
        if not validate_turkish_plate(target).valid:
            raise RuntimeError(f"generator emitted invalid target: {target}")
        return target


def synthetic_distribution(samples: tuple[SyntheticPlateText, ...]) -> dict[str, dict[str, int]]:
    """Expose measurable text coverage without claiming real-world frequencies."""
    province = Counter(sample.normalized_text[:2] for sample in samples)
    lengths = Counter(str(len(sample.normalized_text)) for sample in samples)
    characters = Counter("".join(sample.normalized_text for sample in samples))
    return {
        "province": dict(sorted(province.items())),
        "length": dict(sorted(lengths.items())),
        "character": dict(sorted(characters.items())),
    }


def build_synthetic_qa_specs(*, count: int, seed: int) -> tuple[SyntheticPlateText, ...]:
    """Create a small coverage-first QA set; never a scale-training distribution."""
    if not 1 <= count <= SYNTHETIC_QA_CORPUS_MAX_COUNT:
        raise ValueError(f"QA count must be between 1 and {SYNTHETIC_QA_CORPUS_MAX_COUNT}")
    forced_targets = _coverage_targets()
    selected = forced_targets[:count]
    reserved = set(selected)
    generator = TurkishPlateTextGenerator(seed)
    next_sample_id = len(selected) + 1
    while len(selected) < count:
        selected.append(generator.generate(next_sample_id, reserved=reserved).normalized_text)
        next_sample_id += 1
    return tuple(
        SyntheticPlateText(
            sample_id=index,
            normalized_text=target,
            formatted_text=format_turkish_plate(target),
            seed=seed,
            attempt=-1 if index <= len(forced_targets) else 0,
        )
        for index, target in enumerate(selected, start=1)
    )


def render_synthetic_plate(
    spec: SyntheticPlateText,
    *,
    config: SyntheticRenderConfig = DEFAULT_SYNTHETIC_RENDER_CONFIG,
) -> tuple[ImageArray, dict[str, object]]:
    """Render clean geometry, typography, then camera degradation as separate stages."""
    if normalize_plate_text(spec.normalized_text).normalized_text != spec.normalized_text:
        raise ValueError("renderer requires a normalized target")
    variant = _FONT_VARIANTS[(spec.sample_id - 1) % len(_FONT_VARIANTS)]
    difficulty = _DIFFICULTIES[(spec.sample_id - 1) % len(_DIFFICULTIES)]
    clean, clean_metadata = _render_clean_plate(spec, variant=variant, config=config)
    rng = np.random.default_rng(_render_seed(spec))
    rendered, camera_metadata = _apply_camera_degradation(
        clean, rng, difficulty=difficulty, config=config
    )
    return rendered, {
        "renderer_version": PROJECT_SYNTHETIC_GENERATOR_VERSION,
        "pipeline": ["clean_plate_geometry", "typography", "camera_degradation"],
        "font_asset_id": variant.asset_id,
        "font_asset_license": SYNTHETIC_ASSET_LICENSE,
        "plate_geometry": "project-created-procedural-v1",
        "clean_plate": clean_metadata,
        "typography": {
            "font_family": "Barlow Condensed",
            "font_filename": variant.filename,
            "font_weight": variant.weight,
            "font_sha256": variant.sha256,
            "character_spacing_px": config.character_spacing_px,
            "group_spacing_px": config.group_spacing_px,
        },
        "transform": camera_metadata,
    }


def build_synthetic_qa_corpus(
    *,
    output_root: Path,
    count: int = 72,
    seed: int = 20260824,
    config: SyntheticRenderConfig = DEFAULT_SYNTHETIC_RENDER_CONFIG,
) -> str:
    """Build a deterministic, small visual-QA corpus in the generic file layout."""
    specs = build_synthetic_qa_specs(count=count, seed=seed)
    images_root = output_root / "images"
    labels_root = output_root / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    records: list[RecognitionLayoutRecord] = []
    contact_inputs: list[tuple[SyntheticPlateText, ImageArray, dict[str, object]]] = []
    for spec in specs:
        image, metadata = render_synthetic_plate(spec, config=config)
        image_path = images_root / f"{spec.sample_id}.png"
        encoded, content = cv2.imencode(".png", image)
        if not encoded:
            raise RuntimeError("could not encode synthetic QA PNG")
        _write_bytes_if_missing_or_identical(image_path, content.tobytes())
        label_content = f"{spec.normalized_text}\n"
        label_path = labels_root / f"{spec.sample_id}.txt"
        _write_text_if_missing_or_identical(label_path, label_content)
        records.append(
            RecognitionLayoutRecord(
                sample_id=spec.sample_id,
                derived_image_path=f"images/{spec.sample_id}.png",
                derived_label_path=f"labels/{spec.sample_id}.txt",
                source_id="project-owned-synthetic",
                source_version=PROJECT_SYNTHETIC_GENERATOR_VERSION,
                raw_image_relative_path=f"renderer-seed/{spec.seed}/{spec.sample_id}",
                raw_filename="not-applicable-generated-before-output-filename",
                raw_transcription=spec.formatted_text,
                normalized_transcription=spec.normalized_text,
                raw_image_sha256=sha256_file(image_path),
                derived_image_sha256=sha256_file(image_path),
                label_sha256=sha256_text(label_content),
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                source_license_verification="PROJECT_OWNED_ASSET_MANIFEST_RECORDED",
                source_provenance_state="PROJECT_OWNED_QA_ONLY",
                training_authorization_state="BLOCKED_PENDING_OWNER_DATASET_RELEASE",
                group_id=f"synthetic-base-text-{spec.normalized_text}",
                transformation_metadata=metadata,
            )
        )
        contact_inputs.append((spec, image, metadata))
    manifest = "".join(
        json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    _write_text_if_missing_or_identical(output_root / "manifest.jsonl", manifest)
    _write_contact_sheets(output_root / "contact-sheets", contact_inputs)
    asset_manifest = {
        "schema_version": "project-synthetic-assets-v1.1",
        "generator_technical_status": GENERATOR_TECHNICAL_STATUS,
        "asset_license_status": ASSET_LICENSE_STATUS,
        "synthetic_dataset_build_status": SYNTHETIC_DATASET_BUILD_STATUS,
        "owner_training_authorization": OWNER_TRAINING_AUTHORIZATION,
        "assets": _asset_manifest_entries(),
        "config": asdict(config),
        "distribution": synthetic_distribution(specs),
        "qa_only": True,
        "qa_count_maximum": SYNTHETIC_QA_CORPUS_MAX_COUNT,
        "happens_before_training": "owner dataset-release authorization and release-gate approval",
    }
    _write_text_if_missing_or_identical(
        output_root / "asset-manifest.json",
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n",
    )
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def _coverage_targets() -> list[str]:
    """Known-valid targets exposing every V1 family and repeated-character cases."""
    targets = [
        "01A11",
        "02A111",
        "03A1111",
        "04AB11",
        "05AB111",
        "06AB1111",
        "07ABC11",
        "08ABC111",
        "09ABC1111",
    ]
    targets.extend(
        f"{10 + index:02d}{letter}A{100 + index:03d}" for index, letter in enumerate(_LETTERS)
    )
    targets.extend(("40AA11", "41BB22", "42CC33", "43DD44"))
    targets.extend(f"{50 + digit:02d}A{digit}{digit}" for digit in range(10))
    if len(targets) != len(set(targets)) or not all(
        validate_turkish_plate(target).valid for target in targets
    ):
        raise RuntimeError("synthetic QA coverage target is not valid and unique")
    return targets


def _render_clean_plate(
    spec: SyntheticPlateText, *, variant: FontVariant, config: SyntheticRenderConfig
) -> tuple[ImageArray, dict[str, object]]:
    image = Image.new("RGB", (config.canvas_width, config.canvas_height), (244, 244, 240))
    draw = ImageDraw.Draw(image)
    outer = config.outer_margin_px
    vertical_outer = config.vertical_margin_px
    if vertical_outer is None:
        vertical_outer = outer
    draw.rectangle(
        (
            outer,
            vertical_outer,
            config.canvas_width - outer - 1,
            config.canvas_height - vertical_outer - 1,
        ),
        outline=(20, 20, 20),
        width=config.border_width_px,
    )
    draw.rectangle(
        (
            outer + config.border_width_px,
            vertical_outer + config.border_width_px,
            outer + config.blue_band_width_px,
            config.canvas_height - vertical_outer - config.border_width_px,
        ),
        fill=(20, 54, 130),
    )
    groups = spec.formatted_text.split(" ")
    font_size = (
        74 if len(spec.normalized_text) <= 6 else 68 if len(spec.normalized_text) <= 8 else 62
    )
    if config.scale_font_to_height:
        font_size = min(font_size, max(32, int(config.canvas_height * 0.62)))
    font = ImageFont.truetype(_font_path(variant), size=font_size)
    available = config.canvas_width - config.text_left_margin_px - config.text_right_margin_px
    minimum_font_size = 16 if config.scale_font_to_height else 32
    while _groups_width(groups, font, config) > available and font_size > minimum_font_size:
        font_size -= 1
        font = ImageFont.truetype(_font_path(variant), size=font_size)
    width = _groups_width(groups, font, config)
    if width > available:
        raise ValueError(f"target does not fit plate geometry: {spec.normalized_text}")
    text_bbox = font.getbbox("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text_height = text_bbox[3] - text_bbox[1]
    x = config.text_left_margin_px + max(0, (available - width) // 2)
    y = int(max(7, (config.canvas_height - text_height) // 2 - text_bbox[1]))
    for group_index, group in enumerate(groups):
        x = _draw_spaced_text(
            draw, text=group, x=x, y=y, font=font, spacing=config.character_spacing_px
        )
        if group_index < len(groups) - 1:
            x += config.group_spacing_px
    rgb = np.asarray(image, dtype=np.uint8)
    return cast(ImageArray, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), {
        "canvas": [config.canvas_width, config.canvas_height],
        "border": "black rectangular procedural border",
        "left_band": "blue procedural country band",
        "formatted_text": spec.formatted_text,
        "font_size_px": font_size,
    }


def _groups_width(
    groups: list[str], font: ImageFont.FreeTypeFont, config: SyntheticRenderConfig
) -> int:
    character_widths = (
        int(round(sum(font.getlength(character) for character in group)))
        + config.character_spacing_px * (len(group) - 1)
        for group in groups
    )
    return sum(character_widths) + config.group_spacing_px * (len(groups) - 1)


def _draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    spacing: int,
) -> int:
    cursor = x
    for character in text:
        draw.text((cursor, y), character, font=font, fill=(18, 18, 18))
        cursor += int(round(font.getlength(character))) + spacing
    return cursor - spacing


def _apply_camera_degradation(
    image: ImageArray,
    rng: np.random.Generator,
    *,
    difficulty: Literal["clean", "easy", "medium", "hard"],
    config: SyntheticRenderConfig,
) -> tuple[ImageArray, dict[str, object]]:
    if difficulty == "clean":
        return image.copy(), {"difficulty": difficulty, "camera_degradation_applied": False}
    multiplier = {"easy": 0.35, "medium": 0.67, "hard": 1.0}[difficulty]
    height, width = image.shape[:2]
    rotation = float(
        rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees) * multiplier
    )
    scale = float(1.0 - rng.uniform(0.0, 1.0 - config.min_scale) * multiplier)
    tx = float(rng.uniform(-config.max_translation_px, config.max_translation_px) * multiplier)
    ty = float(rng.uniform(-config.max_translation_px, config.max_translation_px) * multiplier)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, scale)
    matrix[:, 2] = matrix[:, 2] + np.asarray((tx, ty), dtype=np.float64)
    transformed = cast(
        ImageArray,
        np.asarray(
            cv2.warpAffine(image, matrix, (width, height), borderValue=(238, 238, 238)),
            dtype=np.uint8,
        ),
    )
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    offset = np.asarray(
        rng.uniform(
            -config.max_perspective_offset_px, config.max_perspective_offset_px, size=(4, 2)
        )
        * multiplier,
        dtype=np.float32,
    )
    transformed = cast(
        ImageArray,
        np.asarray(
            cv2.warpPerspective(
                transformed,
                cv2.getPerspectiveTransform(corners, corners + offset),
                (width, height),
                borderValue=(238, 238, 238),
            ),
            dtype=np.uint8,
        ),
    )
    blur_kernel = (
        3 if difficulty == "hard" or (difficulty == "medium" and bool(rng.integers(0, 2))) else 1
    )
    if blur_kernel > 1:
        transformed = cast(
            ImageArray,
            np.asarray(
                cv2.GaussianBlur(transformed, (blur_kernel, blur_kernel), 0.7), dtype=np.uint8
            ),
        )
    alpha = float(
        1.0
        + rng.uniform(config.min_brightness_alpha - 1.0, config.max_brightness_alpha - 1.0)
        * multiplier
    )
    beta = float(rng.uniform(-config.max_brightness_beta, config.max_brightness_beta) * multiplier)
    transformed = cast(
        ImageArray,
        np.asarray(cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta), dtype=np.uint8),
    )
    gamma = float(1.0 + rng.uniform(config.min_gamma - 1.0, config.max_gamma - 1.0) * multiplier)
    lookup = np.array(
        [((index / 255.0) ** (1.0 / gamma)) * 255 for index in range(256)], dtype=np.uint8
    )
    transformed = cast(ImageArray, np.asarray(cv2.LUT(transformed, lookup), dtype=np.uint8))
    noise_stddev = float(rng.uniform(0.0, config.max_noise_stddev) * multiplier)
    if noise_stddev > 0:
        noise = np.asarray(rng.normal(0.0, noise_stddev, transformed.shape), dtype=np.int16)
        transformed = cast(
            ImageArray,
            np.asarray(np.clip(transformed.astype(np.int16) + noise, 0, 255), dtype=np.uint8),
        )
    shadow_strength = float(rng.uniform(0.0, config.max_shadow_strength) * multiplier)
    gradient = np.linspace(1.0 - shadow_strength, 1.0, width, dtype=np.float32)
    transformed = cast(
        ImageArray,
        np.asarray(
            np.clip(transformed.astype(np.float32) * gradient[None, :, None], 0, 255),
            dtype=np.uint8,
        ),
    )
    jpeg_quality = int(round(96 - (96 - config.min_jpeg_quality) * multiplier * rng.random()))
    encoded, compressed = cv2.imencode(
        ".jpg", transformed, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    )
    if not encoded:
        raise RuntimeError("could not apply deterministic JPEG compression")
    decoded = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("could not decode deterministic JPEG compression")
    return cast(ImageArray, np.asarray(decoded, dtype=np.uint8)), {
        "difficulty": difficulty,
        "camera_degradation_applied": True,
        "rotation_degrees": rotation,
        "scale": scale,
        "translation_px": [tx, ty],
        "perspective_offsets_px": offset.tolist(),
        "gaussian_blur_kernel": blur_kernel,
        "brightness_alpha": alpha,
        "brightness_beta": beta,
        "gamma": gamma,
        "noise_stddev": noise_stddev,
        "shadow_strength": shadow_strength,
        "jpeg_quality": jpeg_quality,
    }


def _write_contact_sheets(
    root: Path, inputs: list[tuple[SyntheticPlateText, ImageArray, dict[str, object]]]
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    columns, rows, cell_width, cell_height = 4, 4, 280, 102
    for page_start in range(0, len(inputs), columns * rows):
        page = inputs[page_start : page_start + columns * rows]
        sheet = np.full((rows * cell_height, columns * cell_width, 3), 248, dtype=np.uint8)
        for index, (spec, image, metadata) in enumerate(page):
            row, column = divmod(index, columns)
            thumbnail = cv2.resize(image, (260, 55), interpolation=cv2.INTER_AREA)
            x, y = column * cell_width + 10, row * cell_height + 7
            sheet[y : y + 55, x : x + 260] = thumbnail
            difficulty = cast(dict[str, object], metadata["transform"])["difficulty"]
            cv2.putText(
                sheet,
                f"{spec.sample_id:02d} {spec.formatted_text}",
                (x, y + 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                sheet,
                f"{metadata['font_asset_id']} | {difficulty}",
                (x, y + 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.31,
                (65, 65, 65),
                1,
                cv2.LINE_AA,
            )
        encoded, content = cv2.imencode(".png", sheet)
        if not encoded:
            raise RuntimeError("could not encode synthetic contact sheet")
        _write_bytes_if_missing_or_identical(
            root / f"sheet-{page_start // (columns * rows) + 1:02d}.png", content.tobytes()
        )


def _asset_manifest_entries() -> list[dict[str, object]]:
    return [
        {
            "asset_id": SYNTHETIC_ASSET_ID,
            "source": f"Google Fonts ofl/barlowcondensed at {BARLOW_SOURCE_REVISION}",
            "version": "1.408",
            "license": SYNTHETIC_ASSET_LICENSE,
            "use": "default synthetic typography",
            "redistribution_modification": "permitted subject to OFL 1.1 terms and notices",
            "files": [asdict(variant) for variant in _FONT_VARIANTS],
        },
        {
            "asset_id": HERSHEY_ASSET_ID,
            "source": "opencv-python-headless==4.12.0.88 built-in renderer",
            "license": "Apache-2.0",
            "use": "debug/contact-sheet annotation only; not default synthetic typography",
            "training_grade": HERSHEY_TRAINING_GRADE,
        },
        {
            "asset_id": "project-created-procedural-plate-geometry-v1",
            "source": "this repository",
            "license": "project-owned",
            "use": "procedural border, country band, and camera effects",
            "redistribution_modification": "project governance required",
        },
        {
            "asset_id": "Pillow==12.3.0",
            "source": "https://github.com/python-pillow/Pillow",
            "license": "MIT-CMU",
            "use": "deterministic vendored TrueType font rasterization",
            "transitive_risk": "runtime imaging dependency; exact lockfile pins retained",
        },
    ]


def _font_path(variant: FontVariant) -> Path:
    path = (
        Path(__file__).resolve().parents[3]
        / "assets"
        / "fonts"
        / "barlow-condensed"
        / variant.filename
    )
    if not path.is_file() or sha256_file(path) != variant.sha256:
        raise RuntimeError(f"verified synthetic typography asset unavailable or modified: {path}")
    return path


def _render_seed(spec: SyntheticPlateText) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"render:{spec.seed}:{spec.sample_id}:{spec.attempt}".encode("ascii")
        ).digest()[:8],
        "big",
    )


def _write_bytes_if_missing_or_identical(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"synthetic QA rebuild conflict: {path}")
        return
    path.write_bytes(content)


def _write_text_if_missing_or_identical(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"synthetic QA rebuild conflict: {path}")
        return
    path.write_text(content, encoding="utf-8")
