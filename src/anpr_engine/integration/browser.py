"""Legacy loopback-only owner/development review transport.

This module intentionally retains persistent multipart/batch review behavior
for local evidence work. It is not the planned public Next.js service and is
excluded from every future public deployment or container entry point.
"""

from __future__ import annotations

import json
import mimetypes
import traceback
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from anpr_engine.integration.anpr_v1 import AnprV1Pipeline, decode_upload
from anpr_engine.integration.project_models_browser_ui import HTML as PROJECT_MODELS_ONLY_HTML
from anpr_engine.integration.project_models_release import (
    OWNER_DECISION,
    RELEASE_VERSION,
    unresolved_decision,
)
from anpr_engine.integration.review_store import BatchReviewStore

HTML = PROJECT_MODELS_ONLY_HTML


def parse_multipart(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    if not message.is_multipart():
        raise ValueError("multipart/form-data required")
    files: list[tuple[str, bytes]] = []
    for part in message.iter_parts():
        filename = part.get_filename()
        if filename:
            payload = part.get_payload(decode=True)
            if not isinstance(payload, bytes):
                raise ValueError("invalid multipart file payload")
            files.append((filename, payload))
    return files


class AnprV1Application:
    def __init__(
        self,
        repository: Path,
        *,
        model_bundle: Path,
        device: str = "auto",
    ) -> None:
        self.repository = repository.resolve()
        self.pipeline = AnprV1Pipeline(
            self.repository, model_bundle=model_bundle, device_name=device
        )
        self.store = BatchReviewStore(self.repository / "artifacts/inference/anpr-v1-browser")

    def health(self) -> dict[str, Any]:
        health = self.pipeline.health()
        health["release_version"] = RELEASE_VERSION
        health["owner_decision"] = OWNER_DECISION
        health["active_pipeline"] = (
            "PROJECT_DETECTION_BLUE_BAND_GLYPH_QUAD_V1_PROJECT_RECOGNITION_"
            "STRUCTURAL_VALIDATION_HUMAN_REVIEW"
        )
        return health

    def infer_next(self, batch_id: str) -> dict[str, Any]:
        pending = self.store.pending(batch_id)
        if pending is None:
            return {"done": True, "batch": self.store.load(batch_id)}
        directory = self.store._batch(batch_id)  # governed validated path
        result_directory = directory / "artifacts" / pending["request_id"]
        started = __import__("time").perf_counter()
        try:
            content = (directory / pending["input_reference"]).read_bytes()
            image, image_format = decode_upload(content)
            decode_ms = (__import__("time").perf_counter() - started) * 1000
            result = self.pipeline.infer(
                image,
                batch_id=batch_id,
                request_id=pending["request_id"],
                input_sha256=pending["input_sha256"],
                filename=pending["safe_original_filename"],
                output_directory=result_directory,
            )
            result["input"]["decoded_format"] = image_format
            result["input"]["artifact"] = "raw"
            result["timings_ms"]["image_decode_ms"] = decode_ms
            recognition = result.get("recognition")
            result["project_release_decision"] = unresolved_decision(
                str(result.get("pipeline_status")),
                str(recognition["raw_text"]) if recognition else None,
            )
        except ValueError as error:
            result = self._failure(batch_id, pending, "IMAGE_DECODE_ERROR", str(error))
        except Exception as error:  # keep the remaining batch recoverable
            result = self._failure(batch_id, pending, "INFERENCE_ERROR", str(error))
        self.store.save_result(batch_id, pending["request_id"], result)
        return {"done": False, "record": result, "batch": self.store.load(batch_id)}

    def _failure(
        self, batch_id: str, pending: dict[str, Any], status: str, reason: str
    ) -> dict[str, Any]:
        return {
            "schema_version": "anpr-v1-browser-result-v1",
            "batch_id": batch_id,
            "request_id": pending["request_id"],
            "input": {
                "filename": pending["safe_original_filename"],
                "sha256": pending["input_sha256"],
                "artifact": "raw",
            },
            "pipeline_status": status,
            "active_device": self.pipeline.device.type,
            "models": self.pipeline.health(),
            "detection": {"candidate_count": 0, "candidates": [], "selected": None},
            "recognition": None,
            "project_release_decision": unresolved_decision(status, None),
            "crop_reference": None,
            "annotated_image": None,
            "manual_review": {
                "detection_assessment": "UNREVIEWED",
                "recognition_assessment": "NOT_APPLICABLE",
            },
            "errors": [reason],
            "timings_ms": {},
        }


def handler_factory(application: AnprV1Application) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "ANPR-V1-Local/1.0"

        def log_message(self, format: str, *args: object) -> None:
            print(f"ANPR V1 {self.address_string()} {format % args}")

        def _json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str) -> None:
            self._json({"error": {"status": status, "message": message}}, status)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            try:
                if parsed.path == "/":
                    data = HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if parsed.path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                if parsed.path == "/api/health":
                    self._json(application.health())
                    return
                if parsed.path == "/api/batches":
                    self._json({"batches": application.store.list_batches()})
                    return
                if len(parts) == 3 and parts[:2] == ["api", "batches"]:
                    self._json(application.store.load(parts[2]))
                    return
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "records":
                    self._json({"records": application.store.load(parts[2])["records"]})
                    return
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "export":
                    kind = parse_qs(parsed.query).get("format", ["json"])[0]
                    data, mime = application.store.export(parts[2], kind)
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header(
                        "Content-Disposition", f'attachment; filename="{parts[2]}.{kind}"'
                    )
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if len(parts) == 5 and parts[:2] == ["api", "artifact"]:
                    batch_id, request_id, artifact = parts[2], parts[3], parts[4]
                    batch = application.store.load(batch_id)
                    row = next((r for r in batch["records"] if r["request_id"] == request_id), None)
                    if row is None or artifact not in {
                        "raw",
                        "annotated.jpg",
                        "crop.jpg",
                        "refined-crop.jpg",
                        "owner-adjusted-crop.jpg",
                    }:
                        raise ValueError("artifact is not allowed")
                    if artifact == "raw":
                        path = application.store._batch(batch_id) / row["input_reference"]
                    else:
                        path = (
                            application.store._batch(batch_id) / "artifacts" / request_id / artifact
                        )
                    if not path.is_file():
                        self._error(404, "artifact not found")
                        return
                    data = path.read_bytes()
                    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._error(404, "not found")
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
                self._error(400, str(error))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (
                    length
                    > application.store.max_batch * application.store.max_file_bytes + 1024 * 1024
                ):
                    self._error(413, "request body exceeds batch limit")
                    return
                body = self.rfile.read(length)
                if parsed.path == "/api/batches":
                    files = parse_multipart(self.headers.get("Content-Type", ""), body)
                    self._json(application.store.create_batch(files), HTTPStatus.CREATED)
                    return
                payload = json.loads(body or b"{}")
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "infer-next":
                    self._json(application.infer_next(parts[2]))
                    return
                if len(parts) == 4 and parts[:2] == ["api", "batches"] and parts[3] == "review":
                    request_id = str(payload.pop("request_id"))
                    self._json(application.store.save_review(parts[2], request_id, payload))
                    return
                self._error(404, "not found")
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self._error(400, str(error))
            except Exception as error:
                traceback.print_exc()
                self._error(500, str(error))

        def do_PUT(self) -> None:
            self._error(405, "method not allowed")

        def do_DELETE(self) -> None:
            self._error(405, "method not allowed")

    return Handler


def serve(application: AnprV1Application, *, port: int = 8788) -> None:
    ThreadingHTTPServer(("127.0.0.1", port), handler_factory(application)).serve_forever()
