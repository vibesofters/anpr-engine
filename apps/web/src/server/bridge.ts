import { randomBytes } from "node:crypto";
import type {
  AnprImageInferenceResponseV1,
  AnprPrivateInferenceResponseV1
} from "../../../../packages/api-contract/generated/typescript/contracts";
import { ImageError, type MediaType } from "./image";

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function privateResult(value: unknown, requestId: string, admissionId: string): AnprPrivateInferenceResponseV1 {
  if (!isRecord(value) || value.schema_version !== "anpr-private-inference-response-v1" ||
      value.request_id !== requestId || value.admission_id !== admissionId || !isRecord(value.detection) ||
      !isRecord(value.models) || !isRecord(value.timings_ms) ||
      !["prediction_available", "recognition_unresolved", "no_plate_detected", "multiple_detections", "crop_refinement_failed"].includes(String(value.status))) {
    throw new ImageError("INFERENCE_FAILED");
  }
  const models = value.models;
  if (!isRecord(models.detection) || !isRecord(models.recognition) ||
      models.detection.model_id !== "detection-v3-20260827" ||
      models.recognition.model_id !== "m3-v4-stroke-20260914" ||
      models.detection.sha256 !== "0d5e9b52fae5638e0ef1badafdeb53fe0771d056e43c4953d13cceed520e71ea" ||
      models.recognition.sha256 !== "25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7") {
    throw new ImageError("INFERENCE_FAILED");
  }
  const box = value.detection.box_xyxy_normalized;
  const timings = value.timings_ms as Record<string, unknown>;
  const confidence = (item: unknown) => item === null || (typeof item === "number" && Number.isFinite(item) && item >= 0 && item <= 1);
  if (!(box === null || (Array.isArray(box) && box.length === 4 && box.every((item) => typeof item === "number" && Number.isFinite(item) && item >= 0 && item <= 1) && box[0] <= box[2] && box[1] <= box[3])) ||
      !(value.prediction === null || (typeof value.prediction === "string" && /^[0-9A-Z ]{1,16}$/.test(value.prediction))) ||
      !confidence(value.informational_confidence) || !confidence(value.detection.informational_confidence) ||
      ["total", "validation", "detection", "crop_refinement", "recognition"].some((key) => { const item = timings[key]; return typeof item !== "number" || !Number.isFinite(item) || item < 0 || item > 120_000; }) ||
      !["valid", "invalid", "not_run"].includes(String(value.structural_validity)) ||
      value.crop_policy !== "BLUE_BAND_GLYPH_QUAD_V1") throw new ImageError("INFERENCE_FAILED");
  return value as unknown as AnprPrivateInferenceResponseV1;
}

export function publicResult(privateValue: AnprPrivateInferenceResponseV1): AnprImageInferenceResponseV1 {
  const found = privateValue.detection.box_xyxy_normalized !== null;
  const noPlate = privateValue.status === "no_plate_detected";
  const warnings: AnprImageInferenceResponseV1["warnings"][number][] = [];
  if (privateValue.structural_validity === "invalid") warnings.push("STRUCTURAL_VALIDATION_FAILED");
  if (privateValue.status === "multiple_detections") warnings.push("MULTIPLE_DETECTIONS_TOP1_SELECTED");
  if (privateValue.status === "crop_refinement_failed") warnings.push("CROP_REFINEMENT_FAILED");
  if (privateValue.status === "recognition_unresolved") warnings.push("RECOGNITION_UNRESOLVED");
  return {
    schema_version: "anpr-image-inference-response-v1",
    request_id: privateValue.request_id,
    // A model prediction is never owner-resolved until the active-page review.
    status: noPlate ? "no_plate_detected" : "unresolved",
    models: privateValue.models,
    detection: {
      state: found ? "found" : "not_found",
      box_xyxy_normalized: privateValue.detection.box_xyxy_normalized,
      confidence: privateValue.detection.informational_confidence
    },
    // The private contract names the method but does not attest whether refinement applied.
    crop_refinement: { state: noPlate ? "not_run" : "unavailable", method: found ? privateValue.crop_policy : null },
    recognition: { prediction: privateValue.prediction, confidence: privateValue.informational_confidence },
    structural_validation: privateValue.structural_validity,
    human_review_required: true,
    annotated_image: null,
    timings_ms: privateValue.timings_ms,
    warnings
  };
}

export async function callPrivate(bytes: Buffer, type: MediaType, locale: "en" | "tr", requestId: string): Promise<AnprImageInferenceResponseV1> {
  const rawUrl = process.env.ANPR_PRIVATE_INFERENCE_URL;
  const credential = process.env.ANPR_INTERNAL_SERVICE_CREDENTIAL;
  if (!rawUrl || !credential || credential.length < 32) throw new ImageError("MODEL_NOT_READY");
  let url: URL;
  try { url = new URL("/api/v1/inference", rawUrl); } catch { throw new ImageError("MODEL_NOT_READY"); }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new ImageError("MODEL_NOT_READY");
  const admissionId = randomBytes(24).toString("base64url");
  let response: Response;
  const deadline = AbortSignal.timeout(125_000);
  try {
    response = await fetch(url, {
      method: "POST",
      cache: "no-store",
      signal: deadline,
      headers: {
        "Content-Type": type,
        "Content-Length": String(bytes.length),
        "Authorization": `Bearer ${credential}`,
        "X-ANPR-Request-ID": requestId,
        "X-ANPR-Admission-ID": admissionId,
        "X-ANPR-Locale": locale,
        "X-ANPR-Privacy-Acknowledged": "true"
      },
      body: bytes as unknown as BodyInit
    });
  } catch {
    throw new ImageError(deadline.aborted ? "INFERENCE_DEADLINE_EXCEEDED" : "MODEL_NOT_READY");
  }
  if (!response.ok) {
    const mapping: Record<number, ConstructorParameters<typeof ImageError>[0]> = {
      409: "SESSION_INFERENCE_ACTIVE", 413: "IMAGE_TOO_LARGE", 415: "UNSUPPORTED_MEDIA_TYPE",
      422: "IMAGE_DECODE_FAILED", 429: "RATE_LIMITED", 503: "CAPACITY_UNAVAILABLE", 504: "INFERENCE_DEADLINE_EXCEEDED"
    };
    throw new ImageError(mapping[response.status] ?? "INFERENCE_FAILED");
  }
  const declaredLength = Number(response.headers.get("content-length") ?? 0);
  if (!Number.isFinite(declaredLength) || declaredLength < 0 || declaredLength > 64_000) throw new ImageError("INFERENCE_FAILED");
  if (!response.body) throw new ImageError("INFERENCE_FAILED");
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for await (const chunk of response.body) {
      size += chunk.byteLength;
      if (size > 64_000) {
        throw new ImageError("INFERENCE_FAILED");
      }
      chunks.push(chunk);
    }
  } catch (error) {
    if (error instanceof ImageError) throw error;
    throw new ImageError(deadline.aborted ? "INFERENCE_DEADLINE_EXCEEDED" : "MODEL_NOT_READY");
  }
  const text = Buffer.concat(chunks).toString("utf8");
  let parsed: unknown;
  try { parsed = JSON.parse(text); } catch { throw new ImageError("INFERENCE_FAILED"); }
  return publicResult(privateResult(parsed, requestId, admissionId));
}
