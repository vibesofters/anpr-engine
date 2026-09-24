# Real-image processing trust and privacy boundary

This Phase 2A document defines the governing engineering contract. Phase 2B
and 2C later added uncommitted local service and frontend candidates; see the
[Phase 2C web boundary](../operations/web-frontend-bff.md). None of these are
a claim of production, infrastructure, security, accessibility, or
comprehensive legal approval.

The owner reports that real-image processing through the planned public web
application was approved following consultation with a lawyer. The agent did
not independently verify that consultation and is not acting as legal counsel.
The decision is limited to the described purpose-limited, one-request lifecycle;
it does not approve every possible future image-processing feature.

## Trust boundaries

- **Browser:** untrusted. It may send one raw JPEG or PNG body only after a
  bilingual notice is displayed and explicitly acknowledged. Locale comes from
  the active English or Turkish context. The browser cannot supply a filename,
  path, URL, base64 JSON value, session/admission identity, model selection,
  batch, archive, video, or arbitrary metadata.
- **Next.js BFF candidate:** the only planned public application boundary. It
  enforces origin, rate, content-length and streaming limits; validates the file
  signature; issues the secure session cookie; generates request identity; and
  forwards only reviewed metadata and validated bytes.
- **Private inference service candidate:** internal application network only. It
  repeats size/signature checks and hardened decoding. It must have no
  public route or CORS exposure.
- **Private model mount:** read-only and outside the public repository and
  container layers. Readiness fails unless both approved SHA-256 identities
  match.
- **Logging:** generated request identity and safe operational categories only.
  Images, encodings, filenames, plates, predictions, corrections, crops,
  hashes, bodies, raw IPs, session/admission values, query strings, paths,
  stack traces, and secrets are prohibited.
- **Network:** only the BFF may be public. Remote-image fetch, public Python
  inference, camera/video/WebSocket inference, and arbitrary retrieval are not
  allowed.

## Validation and lifecycle

The provisional contract accepts only a single JPEG or PNG of at most 10 MiB
compressed, 20 megapixels decoded, and 8,192 pixels in either dimension. The
declared media type, signature, and decoder result must agree. Exactly one frame
is required; malformed, truncated, animated, multi-frame, or decompression-bomb
inputs are rejected.

The candidate request lifecycle is admission, gate acquisition, bounded read,
validation/decode, metadata removal or disregard, project Detection,
`BLUE_BAND_GLYPH_QUAD_V1` crop refinement, project Recognition, structural
validation, response, and cleanup. The browser produces a metadata-free
annotation only after an explicit download action, using the returned Detection
geometry; it does not reproduce or claim the refined crop. Image, crop, tensor,
encoded output, and request buffers and all locks must be released
in `finally` after success, validation failure, no detection, Recognition
failure, timeout, cancellation, client disconnect, exception, or shutdown.

No source, crop, annotation, prediction, correction, replay, cache, request
table, history, debug dump, object-store object, or database record may be
created. Uploaded images may not be used for training, evaluation, analytics,
dataset construction, unrelated reuse, or future processing.

## Resource boundary

The planned system permits one image per request, one active inference globally,
one per session, queue depth zero, and one CPU-first model-bearing process.
Capacity is rejected immediately. There is no automatic retry, background
inference, database, Redis, object storage, account, analytics, external
monitoring, WebSocket, polling, or persistent history. Unsafe concurrency or
worker increases require later benchmarked and reviewed configuration changes.

Results exist only for the active page. Human review is mandatory. Corrections
remain in browser memory and are not returned to the server. Explicit downloads
may produce only the current versioned JSON or current metadata-stripped
annotation; there is no CSV, original-image download, history, or persistent
result URL.

## Legacy local browser is outside this boundary

The retained Python review browser is separate loopback-only owner/development
tooling. It has multipart, batch, artifact, review, and persistent local-state
features and therefore must not be confused with or reused as the planned public
service. A machine-readable exclusion policy prevents it and its review modules
from becoming public deployment or container entry points.
