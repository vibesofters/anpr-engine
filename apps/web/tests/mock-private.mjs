import { createServer } from "node:http";

createServer(async (request, response) => {
  for await (const chunk of request) { void chunk; }
  const payload = {
    schema_version: "anpr-private-inference-response-v1",
    request_id: request.headers["x-anpr-request-id"],
    admission_id: request.headers["x-anpr-admission-id"],
    status: "prediction_available",
    prediction: "34ABC123",
    prediction_source: "existing_project_decoder",
    decoding_changed_raw: false,
    structural_validity: "valid",
    informational_confidence: 0.81,
    detection: { candidate_count: 1, selection_rule: "highest_confidence_then_box_coordinates", box_xyxy_normalized: [0.2, 0.25, 0.8, 0.75], informational_confidence: 0.88 },
    models: {
      detection: { model_id: "detection-v3-20260827", sha256: "0d5e9b52fae5638e0ef1badafdeb53fe0771d056e43c4953d13cceed520e71ea" },
      recognition: { model_id: "m3-v4-stroke-20260914", sha256: "25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7" }
    },
    crop_policy: "BLUE_BAND_GLYPH_QUAD_V1",
    timings_ms: { total: 8, validation: 1, detection: 4, crop_refinement: 1, recognition: 2 },
    error_code: null
  };
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify(payload));
}).listen(4311, "127.0.0.1");
