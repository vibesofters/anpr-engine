"""Single-model-pair runtime and zero-queue admission control."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from anpr_engine.inference_service.config import ServiceConfig
from anpr_engine.integration.anpr_v1 import (
    DETECTOR_SHA,
    DETECTOR_VERSION,
    RECOGNIZER_SHA,
    RECOGNIZER_VERSION,
    AnprV1Pipeline,
    decode_upload,
)
from anpr_engine.web_contracts.image_validation import validate_encoded_image

CROP_POLICY = "BLUE_BAND_GLYPH_QUAD_V1"


class Pipeline(Protocol):
    load_count: int

    def health(self) -> dict[str, Any]: ...

    def infer(
        self,
        image: NDArray[np.uint8],
        *,
        batch_id: str,
        request_id: str,
        input_sha256: str,
        filename: str,
        output_directory: None,
    ) -> dict[str, Any]: ...


PipelineFactory = Callable[[ServiceConfig], Pipeline]


def _default_pipeline_factory(config: ServiceConfig) -> Pipeline:
    assert config.model_bundle is not None
    return AnprV1Pipeline(
        config.repository,
        model_bundle=config.model_bundle,
        device_name=config.device,
    )


def _validate_decode(payload: bytes, media_type: str) -> NDArray[np.uint8]:
    validate_encoded_image(payload, media_type)
    image, _format = decode_upload(payload)
    return image


@dataclass(frozen=True, slots=True)
class Admission:
    admission_id: str


class ServiceRuntime:
    """Own exactly one pipeline and one inference executor."""

    def __init__(
        self,
        config: ServiceConfig,
        *,
        pipeline_factory: PipelineFactory = _default_pipeline_factory,
    ) -> None:
        self.config = config
        self._pipeline_factory = pipeline_factory
        self.pipeline: Pipeline | None = None
        self.readiness_reason = "not_initialized"
        self._active = False
        self._work_in_flight = False
        self._abandoned = False
        self._work_token = 0
        self._closed = False
        self._recent_request_ids: deque[str] = deque(maxlen=256)
        self._recent_request_id_set: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anpr-inference")

    @property
    def ready(self) -> bool:
        return self.pipeline is not None and not self._closed and self.readiness_reason == "ready"

    @property
    def active(self) -> bool:
        return self._active

    async def initialize(self) -> None:
        try:
            self.config.validate()
            pipeline = await asyncio.to_thread(self._pipeline_factory, self.config)
            health = pipeline.health()
            if pipeline.load_count != 1 or health.get("status") != "HEALTHY":
                raise ValueError("pipeline_health_check_failed")
            self.pipeline = pipeline
            self.readiness_reason = "ready"
        except Exception:
            self.pipeline = None
            self.readiness_reason = "initialization_failed"

    def try_admit(self, request_id: str, admission_id: str) -> tuple[Admission | None, str | None]:
        if request_id in self._recent_request_id_set:
            return None, "duplicate"
        if not self.ready or self._active:
            return None, "capacity"
        self._active = True
        self._abandoned = False
        if len(self._recent_request_ids) == self._recent_request_ids.maxlen:
            oldest = self._recent_request_ids.popleft()
            self._recent_request_id_set.remove(oldest)
        self._recent_request_ids.append(request_id)
        self._recent_request_id_set.add(request_id)
        return Admission(admission_id=admission_id), None

    async def prepare_image(self, payload: bytes, media_type: str) -> NDArray[np.uint8]:
        """Validate and decode on the one worker without releasing a cancelled slot early."""

        if not self._active:
            raise RuntimeError("invalid_admission_state")
        loop = asyncio.get_running_loop()
        self._work_token += 1
        token = self._work_token
        future = self._executor.submit(_validate_decode, payload, media_type)
        self._work_in_flight = True

        def finish(_future: Future[NDArray[np.uint8]]) -> None:
            loop.call_soon_threadsafe(self._finish_preparation, token)

        future.add_done_callback(finish)
        return await asyncio.shield(asyncio.wrap_future(future))

    def _finish_preparation(self, token: int) -> None:
        if token != self._work_token:
            return
        self._work_in_flight = False
        if self._abandoned:
            self._release()

    async def execute(
        self,
        admission: Admission,
        image: NDArray[np.uint8],
        *,
        request_id: str,
    ) -> tuple[dict[str, Any], float]:
        if not self._active or self.pipeline is None:
            raise RuntimeError("invalid_admission_state")
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        future = self._executor.submit(
            self.pipeline.infer,
            image,
            batch_id="internal-single-image",
            request_id=request_id,
            input_sha256="not-retained",
            filename="not-retained",
            output_directory=None,
        )
        self._work_token += 1
        token = self._work_token
        self._work_in_flight = True

        def release(_future: Future[dict[str, Any]]) -> None:
            loop.call_soon_threadsafe(self._release_if_current, token)

        future.add_done_callback(release)
        result = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=self.config.inference_deadline_seconds,
        )
        return result, (time.perf_counter() - started) * 1000

    def _release(self) -> None:
        self._work_in_flight = False
        self._active = False

    def _release_if_current(self, token: int) -> None:
        if token == self._work_token:
            self._release()

    def abandon(self, admission: Admission) -> None:
        """Release immediately or wait for admitted worker work to finish."""

        if self._active and admission.admission_id:
            self._abandoned = True
            if not self._work_in_flight:
                self._release()

    async def close(self) -> None:
        self._closed = True
        self.readiness_reason = "shutdown"
        await asyncio.to_thread(self._executor.shutdown, True, cancel_futures=True)
        self.pipeline = None
        self._active = False


def model_identities() -> dict[str, dict[str, str]]:
    return {
        "detection": {"model_id": DETECTOR_VERSION, "sha256": DETECTOR_SHA},
        "recognition": {"model_id": RECOGNIZER_VERSION, "sha256": RECOGNIZER_SHA},
    }
