# Public Coolify release candidate

Status: **WEB ROUTED WITH VERIFIED HTTPS — PUBLIC INFERENCE ACTIVE WITH A
FIVE-ATTEMPT DAILY PER-IP QUOTA** (2026-09-25).
The target origin is `https://anpr-engine.vibesofters.com`. The separately
reviewed Compose candidate is `deploy/docker/compose.public-candidate.yaml`.
Never attach that hostname to `compose.phase2e-staging.yaml`: the staging
manifest is a loopback HTTP test with different assumptions.

## Live activation and bounded smoke test (2026-09-25)

After explicit owner approval, Coolify enabled
`ANPR_TRUSTED_PROXY_IP_HEADER=x-real-ip` for Web and restarted only the ANPR
service. The proxy-proof route stayed disabled and returned HTTP 404. The
public root returned HTTP 200 with successful TLS verification. A synthetic
JPEG was sent through the public domain, Web BFF, private inference service,
and the approved project Detection and Recognition models. The response used
`anpr-image-inference-response-v1`, contained both approved model SHA-256
identities, and required human review; no prediction-accuracy claim is made
from this synthetic smoke image.

Five sequential admitted requests from the same external client received
HTTP 200. A sixth valid request, carrying a forged `X-Real-IP` header,
received HTTP 429 with `RATE_LIMITED`. This verifies the five-attempt quota
and the tested proxy-spoofing boundary for this client; it is not a per-person
guarantee or a sustained-load qualification. One during-request Docker
snapshot showed Inference at 274.7 MiB and 57.74% container CPU, and Web at
71.11 MiB and 0% container CPU. These are snapshots, not peaks.

The quota lives in an ANPR-scoped persistent `/quota` volume and stores only
an HMAC-derived IP/day key and count. It resets on the Europe/Istanbul
calendar day. Uploaded images and results are not intentionally persisted.
Other hosted projects, DNS, the Inference image, and model weights were not
changed.

## Bounded follow-up monitoring and accessibility audit (2026-09-25)

Fifteen once-per-minute HTTPS root probes returned HTTP 200 with successful
certificate verification. This is an availability check with negligible load,
not a sustained inference throughput test. At the end, both ANPR containers
were running and healthy with zero restarts. A bounded scan of their recent
application logs found zero error-like lines. One final idle snapshot showed
Inference at approximately 237 MiB and 0.34% container CPU, and Web at
approximately 88 MiB and 0% container CPU. Host available memory was about
7.0 GiB. These snapshots cannot establish peaks or causally attribute the
shared host's total load to ANPR.

In a headed browser, English and Turkish content, light and dark themes,
keyboard focus order, reduced-motion behavior, and 320/375-pixel layouts
were exercised. Automated axe-core WCAG A/AA and best-practice scans of the
home and information pages reported zero violations. Color contrast for text
over CSS gradients remained an automated *incomplete* check; selected base
token combinations were separately calculated above 4.5:1, but a complete
manual screen-reader and gradient-state review has not been performed. The
browser showed an accessible, non-stuck daily-limit message after a 429
response. The expected 429 appeared as a browser network-console error, not
an application exception.

## Web update and proxy proof (2026-09-25; activation preflight)

The current Web image was rebuilt from the reviewed local source as
`sha256:9a41b35a20bc17456d25e157c541483151b35e868414746d88061e907419f273`.
Its transferred Docker archive had SHA-256
`fd5f61016de71ac61e60bc600fea8a9b100242695686c02576d713b13c33eed2`;
the host checksum matched before `docker load`. Coolify restarted only the
ANPR Compose service. Both containers became healthy, and the public root
returned HTTP 200 with a valid HTTPS certificate. The Inference image and
private model bundle were unchanged.

Web gained a writable, ANPR-scoped persistent `/quota` volume. Before the
later activation, image processing was fail-closed because
`ANPR_TRUSTED_PROXY_IP_HEADER` was unset. With the temporary proxy-proof route
enabled, outside requests that
forged `X-Real-IP: 198.51.100.77` and `X-Real-IP: 2001:db8::77` both
reported that the proxy supplied a valid, replacement IP. The route returns
only booleans, never the observed address. That proof preceded activation.
The proof route was disabled afterward and remains HTTP 404.

A single post-restart idle measurement showed Web at about 52 MiB RAM and
0.08% container CPU; Inference at about 275 MiB RAM and 0.32% container CPU.
These are snapshots, not capacity or peak-load qualifications.

## Earlier boundary (historical; superseded by live activation)

On 2026-09-24, a read-only DNS lookup returned an A record for the target
hostname pointing to the known Coolify host. No DNS change was made. The
earlier TLS probe returned `TRAEFIK DEFAULT CERT`; this was superseded by the
verified Web-only HTTPS route below. The older ANPR staging service remains
separate. A working landing page does not authorize public image processing.

## Web-only edge verification (2026-09-24; historical)

With the owner's explicit approval, Coolify attached
`https://anpr-engine.vibesofters.com` to the **Web** service on internal port
`3000` only. Inference received no domain or published port. Coolify reported
both ANPR services healthy. HTTP redirected to HTTPS; an external TLS check
verified the hostname and Let's Encrypt chain. The `/` route redirected to
English, and the English and Turkish pages loaded with private/no-store cache
headers and restrictive CSP, frame, referrer, and permissions policies.

The public `/api/v1/inference` path returned `404`. A project-owned, non-plate
PNG sent to `/api/inference`, including a forged `X-Real-IP`, returned `403`.
This confirms the current fail-closed behavior, **not** the proxy's eventual
IP-header trustworthiness or the image-size rejection path. A synthetic log
canary was not visible in the sampled Web application logs. Proxy logs,
temporary-file behavior, cancellation, timeout, and overload behavior have
not yet passed a complete host-side audit. `ANPR_TRUSTED_PROXY_IP_HEADER`
remains unset, so public uploads cannot run Detection or Recognition.

The then-newer local code disabled the upload control and showed an unavailable
state until trusted-proxy and daily-quota settings were configured; that code
was subsequently deployed. A bounded browser deadline prevents a perpetual
processing indicator. The persistent quota is five valid processing attempts
per trusted IP per Istanbul calendar day, not a verified per-person limit.
Its SQLite volume stores only an HMAC token, day, and count.

## Earlier daily-limit Web candidate (historical; not the live image)

The reviewed Linux amd64 Web image is
`sha256:cbc0c2a7192b2e1de7f16441e113fee2ef34119595a85816364a803250e78f74`.
Its exact 89 MiB Docker archive is
`/tmp/anpr-engine-web-daily-limit-v2-20260924.tar`, SHA-256
`5f6b326313e97af09b2036f07f247481c22bebd231f05f86dc3373c949a03a6f`.
The previous Web image and earlier daily-limit archives are **not** this
candidate. Do not switch Coolify's image reference before the archive is
transferred, checked, and loaded on the host. The private Inference image is
unchanged. In a local isolated Compose smoke run, one synthetic image returned
HTTP 200 through the project models and created exactly one quota entry. The
smoke stack and its quota volume were removed afterward. Web lint, typecheck,
build, two focused browser/API flow tests, 76 focused Python tests, and the
public-candidate scan passed. None of this proves the host proxy or live
upload path; the live site remains fail-closed.

The first routed run revealed Next.js image-cache writes failing on the
read-only root. The ANPR-only Compose definition now mounts a 32 MiB,
non-executable tmpfs at `/app/apps/web/.next/cache`; after an ANPR-only
restart, an optimized image request returned `200` and fresh Web logs showed
no cache-write errors. No shared proxy configuration, DNS record, other
Coolify project, model weight, or Git publication state was changed.

## Local release-candidate evidence (2026-09-24)

The owner accepted the locally presented appearance and EN/TR workflow in
this review conversation. This records product/visual acceptance; it is not
legal approval of every public claim or an assistive-technology audit.

The allowlisted current-source contexts produced these Linux amd64 images:

| Component | Local image identity |
| --- | --- |
| Web | `sha256:a5770204f1cd4ceb78cc4debd5d088221f66253bb92cc0fcb874bb615fb47a5b` |
| Inference | `sha256:27d7192fba1ae159ce3443b1789729389940c642c5681eecafab489cbd4602e3` |

Both images run as UID/GID 10001 and contain license inventories. The
allowlisted build contexts exclude weights and datasets; an image-filesystem
check found no `.pt`, `.pth`, `.jpg` or `.jpeg` payloads. The approved model
pair loaded through a read-only private mount in a short, local amd64-on-arm64
container smoke test. Both services became healthy, only Web published a
loopback port, and one synthetic request returned the two project model IDs.
The local smoke stack and its networks were removed afterward. Emulated
timings are not Coolify-host measurements.

## Coolify private startup check (2026-09-24)

The two immutable Linux amd64 image IDs above were transferred to the Coolify
host, checked against the transfer checksum, and loaded without rebuilding on
that shared server. Coolify used the approved read-only private model bundle;
its Detection and Recognition hashes matched the registered identities. The
separate `anpr-public-candidate` service started with no domain attached and
no published inference port. The inference readiness log reported `ready`,
and the Web startup log reported Next.js ready. Both services were then
stopped and Coolify showed both as `Exited`. Docker-wide cleanup was explicitly
disabled. No public route, real-image request, or synthetic end-to-end request
was exercised in this host-side check; no host CPU/RAM measurement was taken
for this specific candidate. Earlier Phase 2E staging measurements remain
separate evidence and must not be represented as candidate measurements.

A fresh external TLS probe still received Traefik's self-signed default
certificate for the target hostname. The DNS A record pointed to the expected
host, but that alone does not make the site publicly ready. The internal
service credential is configured in Coolify and must never be copied into
this document or Git. This check did not change DNS or repository visibility.

The web lint, typecheck, build, flow test and production advisory query passed.
The Python suite passed with one optional private-bundle test skipped in the
general run and then passed separately with the private bundle. One
cancellation test raced under parallel build load because it cancelled
before worker execution; the test now waits for the gated worker, and the
suite passed again under concurrent build. The public-candidate scan and
Compose interpolation checks passed. This evidence does not verify a public
proxy, TLS, image distribution decision or production capacity.

The candidate preserves one CPU inference process, one web process, one active
image at a time, zero queue, no published inference port, a read-only private
model mount, read-only roots, bounded temporary storage, and the previously
tested CPU/RAM/PID caps. `restart: "no"` deliberately retains the conservative
staging behavior: a failed process needs operator intervention. Reconsider
that tradeoff explicitly before a live release; do not silently add replicas,
workers, retries, or automatic restarts on the shared host.

## Values to supply through Coolify, never through Git

| Compose input | Requirement |
| --- | --- |
| `ANPR_PRIVATE_MODEL_BUNDLE_SOURCE` | Owner-controlled absolute host directory containing only the approved model bundle; mount read-only. Never commit or print the path publicly. |
| `ANPR_INTERNAL_SERVICE_CREDENTIAL` | New high-entropy private runtime value, shared only by web and inference. Do not put it in image layers, build arguments, browser variables, logs, or this document. |
| `ANPR_APPLICATION_VERSION` | Reviewed release identifier, with both model identities verified at startup. |

The Compose file pins the approved image IDs directly. A new build requires
replacing those IDs only after rebuilding, inspecting, and verifying both
images. Do not substitute mutable tags during deployment.
Coolify's service parser rejects Compose's required-variable expression in a
bind-mount source, so the model source uses ordinary substitution. The owner
must set it to the verified private bundle directory before deployment;
inspect the generated Compose and stop if it is empty or points elsewhere.
Coolify also converts required-variable expressions in environment values to
their explanatory text, so the credential and application version use ordinary
substitution. Both must be set and verified in Coolify before starting either
container. An unset or explanatory value is a hard stop, not a default.

The exact HTTPS `ANPR_PUBLIC_ORIGIN` is already in the candidate. Do not set
`ANPR_LOCAL_DIRECT_MODE=1` for a public route. The current candidate sets
`ANPR_TRUSTED_PROXY_IP_HEADER=x-real-ip` after the proxy's client-IP header
replacement was demonstrated. Removing this setting fails public inference
closed.

## Pre-activation release order and stop gates (historical)

This checklist records the original gating plan. The live status and observed
verification are at the top of this document; do not read these future-tense
steps as a claim that public inference is still disabled. Git publication and
the remaining manual accessibility/capacity reviews are separate work.

1. **Finish the local release candidate.** Obtain owner approval for the
   English/Turkish copy, theme, favicon, privacy notice and public claims.
   Complete native Turkish and assistive-technology checks. Run the full
   source, dependency/notice, secret, private-data, image-layer and container
   inventory scans. Rebuild the exact inference and web images from that
   reviewed state. Confirm that no weights, real plates or private governance
   records are embedded. The current workstation has limited free space; use
   an isolated builder if a CPU-PyTorch rebuild would exhaust it.
2. **Keep Git publication separate.** The original repository history
   contained private material and was retained in a private legacy repository.
   The public repository must be created from a separately audited current-file
   snapshot with one root commit; it must not import or expose the original
   history. Container release does not require making the repository public
   first.
3. **Review the public Compose candidate off-host.** Validate interpolation,
   the selected immutable image identities, model mount, notices, health
   checks, private-only inference network and resource caps. Inspect
   Coolify's generated/deployable Compose as well as the source definition.
   Do not use the staging image tags as the final release identity.
4. **Verify the public edge in a bounded window.** Before allowing uploads,
   verify that the hostname resolves to the intended host, Coolify routes
   only the web service, HTTPS presents a valid certificate for the hostname,
   port 3000 is the *internal* target, and no inference port is exposed.
   Prove the trusted proxy **replaces** a caller-supplied `X-Real-IP` value.
   Test body limits, buffering/spooling, cache behavior, log redaction,
   cancellation, timeout alignment, overload responses and ANPR-only
   rollback. Leave `ANPR_TRUSTED_PROXY_IP_HEADER` unset until that proof
   passes; a working landing page alone is insufficient.
   The new Web image provides a temporary `/api/proxy-proof` endpoint. Set
   `ANPR_PROXY_PROOF_ENABLED=1` on Web only while the trust flag is **unset**.
   From outside the host, send one request with
   `X-Real-IP: 198.51.100.77` and another with
   `X-Real-IP: 2001:db8::77`. Require `header_present=true` and the
   matching `forged_*_replaced=true` for both. The response never includes
   the observed address. If either test fails, leave inference disabled.
   Set `ANPR_PROXY_PROOF_ENABLED=0` after testing; the route then returns 404.
5. **Explicit release approval.** Record exact image IDs, model hashes,
   actual-host resource observations, license/notice disposition, privacy
   text and rollback receipt. Only then authorize public routing and set
   `ANPR_TRUSTED_PROXY_IP_HEADER=x-real-ip` if the header-replacement test
   passed. Make one synthetic end-to-end request before any authorized real
   upload. Watch shared-host pressure and stop only ANPR if thresholds fail.

## What a DNS change would do

Changing this hostname's A/AAAA/CNAME record would redirect visitors to a
different endpoint once caches expire. A wrong value could make the ANPR page
unavailable, send uploads to an unintended server, or prevent certificate
issuance. Editing only `anpr-engine.vibesofters.com` should not redirect other
subdomains, but changing a wildcard or parent-domain record could. The
current A record already points to the intended host; **no DNS edit is planned
or authorized**. Coolify routing and TLS still need their own checks.

## Rollback

Remove the ANPR web route or stop only the ANPR Compose service. Confirm that
both ANPR components exit and the public hostname no longer accepts uploads.
Do not run Docker-wide cleanup, alter other Coolify resources, delete the
private model bundle, or change DNS as part of the ordinary ANPR rollback.
