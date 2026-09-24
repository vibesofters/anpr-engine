# Phase 2E Coolify staging preflight

Status: **PRIVATE STAGING COMPLETED AND STOPPED; PUBLIC PROXY UNVERIFIED**.
This is a non-secret intake and environment-variable guide, not a production
deployment configuration. The Phase 2D Linux arm64/glibc results are not
Coolify-host measurements. Subsequent Linux amd64 images were checked in a
bounded, loopback-only Coolify staging run. That run does not authorize a
public upload route, production deployment, DNS change, repository
publication, commit, push, or history rewrite.

## Owner-supplied host snapshot

The owner supplied these non-secret observations. The observation time was
not specified; they are **not** inference benchmarks or a reservation of
resources for ANPR Engine.

| Fact | Owner-supplied observation |
| --- | --- |
| CPU | x86_64; 8 vCPUs. |
| RAM | 15 GiB total; 8.6 GiB available at observation. |
| Swap | None. |
| Root filesystem | 150 GB total; 71 GB available. |
| Docker | Engine 29.4.2; 51 active containers. |
| Load averages | 1.39, 1.40, 1.36 at observation. |
| OS and Coolify versions | Ubuntu 24.04.4 LTS and Coolify 4.3.23 were observed in the later private staging checks. |
| Supported limit behavior | The scoped ANPR containers showed enforced CPU, memory, PID, read-only-root, tmpfs, and network limits in the later private staging checks. |

The Phase 2D Linux arm64 images are **not native** to this x86_64 host. Linux
amd64/Debian-glibc CPU-only images were subsequently loaded and run in the
bounded private staging exercise. Their successful loopback test does not
verify the public proxy, TLS, domain, production capacity, or all failure
paths.

## Historical pre-staging host inventory checklist

The following was the intake checklist before the bounded run. Refresh only
if another run is separately approved; collect facts without hostnames,
addresses, credentials, tokens, container names, project names, private
paths, or other projects' data:

| Field | Needed answer |
| --- | --- |
| CPU | Architecture and vCPU count available to containers. |
| Memory | Total RAM and currently available RAM; aggregate idle and busy pressure, with no per-project identifiers. |
| Storage | Free space on the Docker image/build and temporary staging volumes; filesystem type if needed to assess buffering. No paths need be published. |
| Platform | Operating-system name/version, kernel architecture, Docker Engine/container runtime and version, and Coolify version. |
| Limits | Whether CPU, memory, PID, read-only filesystem, tmpfs, network isolation, and bind-mount limits are supported and enforced. |
| Pressure | Aggregate CPU, memory, disk, and swap pressure at idle and during ordinary co-hosted workload; do not list other services. |
| Architecture match | Already resolved: the Phase 2D arm64 images are not native. Verify the off-host Linux amd64/Debian-glibc image IDs and selected native binaries on this host before staging. |

The owner used Coolify and owner-controlled terminal sessions for the bounded
run. Any future run should send only sanitized values from the short manual
runbook in `docs/operations/coolify-manual-staging-runbook.md`. Do **not** send a
dashboard session, address, SSH config, private key, password, token, public
IP, registry credential, container names from other projects, or private
paths. The observations to date support only the private loopback path; no
public-route capacity or latency conclusion is possible.

## Coolify staging environment-variable guide

Examples below are inert or non-secret. They are **not** benchmark-selected
limits. Enter secrets only through an owner-controlled runtime secret
mechanism after host capabilities are known. Never pass secret values as
Docker build arguments, include them in image layers, commit them, or paste
them into a report.

| Variable | Scope and purpose | Safe example or rule |
| --- | --- | --- |
| `ANPR_INTERNAL_SERVICE_CREDENTIAL` | Web and inference runtime only; authenticates the private hop. | `<owner-injected-secret>`; same value in both services, never in a build context or browser variable. |
| `ANPR_PRIVATE_MODEL_BUNDLE_SOURCE` | Host-side staging orchestrator only; source of the read-only bind mount. | `<owner-controlled-private-location>`; do not record the real path publicly. |
| `ANPR_PRIVATE_MODEL_BUNDLE` | Inference container only; mounted bundle target. | `/models`, read-only. |
| `ANPR_SERVICE_ENV` | Inference container; enables production fail-closed checks. | `production` for staging behavior. |
| `ANPR_INFERENCE_DEVICE` | Inference container; CPU-only selection. | `cpu`. |
| `ANPR_INFERENCE_DEADLINE_SECONDS` | Inference container; whole-image deadline, including validation. | `<measured-value>` in the service's supported 1–120 second range. |
| `ANPR_APPLICATION_VERSION` | Inference container; safe operational version. | `phase2e-staging` (example label only). |
| `ANPR_RUNTIME_REPOSITORY_ROOT` | Inference container; packaged code/config root. | `/app`. |
| `ANPR_SERVICE_HOST` | Inference container; listener inside the private network only. | `0.0.0.0` inside the container; **no host/public port**. |
| `ANPR_SERVICE_PORT` | Inference container; internal listener port. | `8080`. |
| `ANPR_PRIVATE_INFERENCE_URL` | Web server runtime only; private service address. | `http://inference:8080` on the isolated application network. |
| `ANPR_PUBLIC_ORIGIN` | Web server runtime; exact browser origin for CSRF checks. | `<access-restricted-staging-origin>`; no public upload route in Phase 2E. |
| `ANPR_TRUSTED_PROXY_IP_HEADER` | Web server runtime; enables IP admission only behind a verified header-replacing proxy. | Leave unset until proxy trust and replacement behavior are tested; then `x-real-ip` if verified. |
| `ANPR_LOCAL_DIRECT_MODE` | Web server runtime; special loopback-only mode from Phase 2D. | Leave unset for a proxy-based staging path; consider `1` only for a verified loopback/tunnel path. |

`ANPR_LOCAL_PORT` is a Phase 2D local Compose convenience, not a Coolify
capacity setting. `NEXT_PUBLIC_*` must never carry the private service URL,
credential, or model location. The selected staging origin and proxy strategy
must be checked before any upload test; an unverified proxy path must remain
closed. All staging images must use read-only private model mounts, one web
process, one CPU model-bearing process, one active inference globally, and
queue depth zero.

## Benchmark gate for the owner-operated staging run

Before starting containers, establish aggregate baseline resource pressure
and a stop threshold that protects co-hosted services. Bound the test window
and use only synthetic, non-identifying images. Confirm target architecture,
libc, image notices, model hashes, internal-only networking, read-only roots,
and no public upload route. Measure idle/peak memory, CPU, model-load cost,
cold/warm/serial latency, two simultaneous submissions, cancellation,
timeout, restart/recovery, proxy/request spooling, temporary storage, logs,
health/readiness, and shutdown. Compare one resident model pair with any
load-on-demand option only if the current architecture supports it without
raising concurrency, worker count, or queue depth. Stop immediately if the
pre-agreed pressure threshold is crossed or another service is affected.
Record measurements and provisional limits only after this host-based run.
Remove the staging stack, temporary images/artifacts, and test data after
verification; preserve only sanitized benchmark evidence. Rollback is to
stop and remove the ANPR staging stack, leaving co-hosted services untouched.
