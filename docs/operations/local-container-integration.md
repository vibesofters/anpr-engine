# Phase 2D local container integration

This is a **local, CPU-only integration setup**, not a Coolify deployment or
production sizing decision. The owner reports that the Sharp/libvips licensing
question has been reviewed and permits this phase. That is an owner-reported
decision, not an independent legal conclusion.

## Build and run

Use `scripts/prepare_phase2d_context.py` to create separate, allowlisted,
temporary contexts outside the repository, then build each image with its
corresponding `deploy/docker/*.Dockerfile`. Do not build from the repository
root. The web context contains only web source, generated contract types,
processing notice, lockfile, and Docker build inputs. The inference context
contains only the Phase 2B runtime allowlist, pinned CPU requirements, and
Docker build inputs. Neither context contains weights, datasets, training
tools, experiment artifacts, local review-browser code, external OCR code,
or secrets.

From the repository root, build with temporary allowlisted contexts:

```sh
WEB_CONTEXT="$(mktemp -d /tmp/anpr-phase2d-web.XXXXXX)"
INFERENCE_CONTEXT="$(mktemp -d /tmp/anpr-phase2d-inference.XXXXXX)"
python3 scripts/prepare_phase2d_context.py web "$WEB_CONTEXT"
python3 scripts/prepare_phase2d_context.py inference "$INFERENCE_CONTEXT"
docker build --platform linux/arm64 -f "$WEB_CONTEXT/deploy/docker/web.Dockerfile" -t anpr-engine-web:phase2d-local "$WEB_CONTEXT"
docker build --platform linux/arm64 -f "$INFERENCE_CONTEXT/deploy/docker/inference.Dockerfile" -t anpr-engine-inference:phase2d-local "$INFERENCE_CONTEXT"
```

The local build target is **Linux arm64, glibc (Debian bookworm)**. The base
images are pinned by digest in the Dockerfiles. The inference dependency lock
uses hashed Linux arm64 CPU-only wheels. The runtime images run as UID/GID
10001, with read-only root filesystems and a bounded memory-backed `/tmp`
(web 64 MiB; inference 128 MiB). No persistent request volume is mounted.
The inference image has one process and one model copy per approved model;
the service admits one active inference globally and has no queue.

Runtime variable names and purposes (never put values in this repository):

| Name | Purpose |
| --- | --- |
| `ANPR_INTERNAL_SERVICE_CREDENTIAL` | Shared BFF-to-private-service bearer credential, injected only at runtime; at least 32 characters. |
| `ANPR_PRIVATE_MODEL_BUNDLE_SOURCE` | Host location of the approved private model bundle; Compose mounts it read-only at `/models`. Do not publish its value. |
| `ANPR_LOCAL_PORT` | Optional loopback-only web port, default 3000. |
| `ANPR_PUBLIC_ORIGIN` | Exact same-origin URL for browser request validation; set by Compose. |
| `ANPR_PRIVATE_INFERENCE_URL` | Internal service URL, set by Compose and never sent to browser code. |
| `ANPR_LOCAL_DIRECT_MODE` | Restricts local direct mode to a loopback origin with no trusted edge proxy; not a Coolify setting. |
| `ANPR_SERVICE_ENV` | Enables fail-closed production service checks. |
| `ANPR_APPLICATION_VERSION` | Non-sensitive operational version. |
| `ANPR_RUNTIME_REPOSITORY_ROOT` | Container-only runtime source root. |
| `ANPR_PRIVATE_MODEL_BUNDLE` | Container-only model mount target. |
| `ANPR_INFERENCE_DEVICE` | CPU in Phase 2D. |
| `ANPR_INFERENCE_DEADLINE_SECONDS` | Per-image service deadline, capped at 120 seconds. |

With the two required owner-provided environment variables set in the local
shell, use `docker compose -f deploy/docker/compose.local.yaml up -d` and open
the loopback web origin. Stop with `docker compose -f
deploy/docker/compose.local.yaml down`. Do not store the credential in a
Compose env file. The web port is published to `127.0.0.1` only. Inference
has no host port and is on an internal Docker network shared only with the
web service.

`/en` is the web health check. The Python service exposes internal `/healthz`
and `/readyz`; readiness remains false if either private model hash or loading
fails. Compose waits for inference readiness before starting web. The
version-specific image inventories, package licenses, Debian copyright files,
Node/Python runtime licenses, selected native binary hashes, and upstream
Sharp/libvips notices are under `/licenses` in each image. Inspect the built
images, not the lockfile alone, before any distribution decision.

Compose's 1 CPU / 4 GiB inference and 0.5 CPU / 1 GiB web caps, 128/64 MiB
tmpfs sizes, startup checks, and current 120/125-second service/BFF deadlines
are conservative **local test settings**, not qualified Coolify limits. A
future reverse proxy must not buffer uploads to persistent disk and must be
tested with the same 10 MiB raw-body limit; the BFF and hardened decoder
also enforce JPEG/PNG, framing, ≤20 MP, and ≤8192 pixels per side. There is
no reverse proxy in this local Compose path. A container's read-only root
prevents filesystem retention, but Docker Desktop/host diagnostics and browser
downloads remain outside that guarantee. The browser keeps the active image,
prediction, and review state only until page reset/unload. Logs must be
checked for sensitive content on every changed build.

The non-secret Phase 2E intake and Coolify variable guide is
`docs/operations/coolify-staging-preflight.md`. Before staging or deployment:
inventory the actual Coolify host, benchmark
the approved pair there, set final CPU/memory/concurrency/deadline limits,
verify proxy buffering and timeouts, inspect the target-architecture image
and notices, verify domain/TLS and privacy/security gates, and obtain the
separate explicit deployment approval. This local arm64 result does not
qualify another architecture or host.
