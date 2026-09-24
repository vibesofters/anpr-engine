# Phase 2C bilingual web application and BFF

Status: local implementation and container candidate. A bounded, private
loopback-only staging check was completed and stopped. No public release or
deployment is approved. The real-image Phase 2A contract supersedes earlier
synthetic-only web plans.

## Local development

From `apps/web`, run `npm ci`, then supply **local test-only** values for
`ANPR_PRIVATE_INFERENCE_URL` and `ANPR_INTERNAL_SERVICE_CREDENTIAL` and run
`npm run dev`. The private service must already be available on the local
application network. Do not put either value in a `NEXT_PUBLIC_*` variable,
browser code, source file, or log. The repository supplies no production secret.
`npm run typecheck`, `npm run build`, and `npm test` exercise the frontend and a
synthetic private-service test double. The test double must never be deployed.

`/` redirects to `/en`; `/en` and `/tr` present the processing notice and
single-image workflow. The localized privacy, responsible-use, and limitations
pages are informational. English/Turkish text is provisional pending native
Turkish review. Branding is a restrained text-first proposal, not approved
final branding. Metadata is self-canonical per language with reciprocal
alternates; indexing is disabled until release approval.

## Trust boundary

The browser sends exactly one raw JPEG/PNG body to same-origin
`POST /api/inference`. A server-issued HttpOnly, SameSite=Strict cookie is
provided on localized page responses. The BFF checks the exact configured
origin, cookie, acknowledgement, media type, streaming 10 MiB limit, image
signature/trailer, single frame, full decode, 20 megapixels, and 8,192 pixels
per side. It reads no multipart, filename, remote URL, path, base64 request,
model selection, or correction input. The BFF creates request/admission IDs
and forwards reviewed bytes and headers only to the Phase 2B private service.
Its URL and bearer credential are server-only.

One Node application process has one global active admission, no queue, and
one active request per session. Initial conservative **provisional** controls
are twelve admitted requests per session and twenty-four per IP in five minutes, a
15-second upload deadline, and a 125-second private-service call deadline.
The Phase 2B 120-second model deadline and all proxy/runtime deadlines must be
aligned and replaced with measured values after actual-host staging
benchmarks. A browser-side 140-second deadline prevents an indefinitely busy
interface. Short-window admission entries are bounded and in memory. The
separate daily limit uses a small persistent SQLite file at
`ANPR_DAILY_QUOTA_PATH`: five validated, admitted processing attempts per
trusted IP per Europe/Istanbul calendar day. It stores only a day, an HMAC
token derived from the IP using the private service credential, and a count;
old rows are removed on the next admitted request. There is no image, plate,
raw IP, result history, Redis, analytics, background task, or automatic retry
in this file. The volume must be private and writable only by Web. A shared IP
shares its quota; this is not a guaranteed person-level identity. A
multi-process/multi-instance frontend requires a new reviewed admission
design before use. The private service remains the final global capacity gate.

In production, set `ANPR_PUBLIC_ORIGIN` to the exact HTTPS origin. The
`x-real-ip` value may be used only when `ANPR_TRUSTED_PROXY_IP_HEADER=x-real-ip`
and the trusted reverse proxy **replaces**, rather than appends or forwards,
client-supplied IP headers. Otherwise the BFF rejects production requests.
IP controls are imperfect for shared networks and proxies. Coolify network,
secret injection, proxy body buffering/no-spool, log retention, and timeouts
must be verified separately before deployment.

## Result and retention

The public response is the versioned Phase 2A schema and is `no-store`. A
prediction remains `unresolved` until an explicit person confirms or corrects
it. Corrections never leave the active browser page. The exported JSON follows
`anpr-session-result-v1`; the image is rendered from the original browser
image plus the Phase 2B normalized **Detection** box into a fresh canvas only
when the user clicks download. Canvas PNG encoding removes embedded source
metadata. Neither overlay nor download is labelled as the
`BLUE_BAND_GLYPH_QUAD_V1` refined crop. The Phase 2B response does not attest
whether that refinement actually applied, so the public
`crop_refinement.state` is conservatively `unavailable` when a candidate is
present and `not_run` otherwise. There is no server download route, CSV,
history, original-image server download, persistent link, or server correction.
Phase 2C added three warning codes to the public response schema so a selected
top-one multiple Detection, crop-refinement failure, and unresolved Recognition
are not silently flattened into an ordinary prediction state.

Image bytes, Object URLs, prediction, and review values live only on the
active page and are cleared on replacement, explicit clearing, navigation, or
page closure. Uploaded source metadata is discarded by the private decoder
before model processing. The BFF and private service intentionally retain no
request image/result artifacts. Browser download managers may independently
save an explicitly requested download; that is controlled by the user.

## Remaining verification gates

This local implementation does not prove reverse-proxy and Next.js runtime
no-spool behavior, multi-host capacity, actual-host load limits, accessibility
across assistive technologies, native Turkish wording, or legal/licensing
compatibility. The locked Sharp package references LGPL-3.0-or-later libvips
platform distributions; a qualified license review and container-layer audit
are required before distribution. No deployment or public release is approved.
