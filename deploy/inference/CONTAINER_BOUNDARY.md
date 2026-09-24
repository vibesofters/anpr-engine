# Phase 2B inference container boundary

This was the Phase 2B build-boundary specification. Phase 2D has now built
and tested a **local Linux arm64/glibc CPU image** under
`deploy/docker/inference.Dockerfile`; it has not been deployed or qualified
for the uninventoried Coolify host. See
`docs/operations/local-container-integration.md` for the local build and
runtime boundary.

The later Docker implementation must use a multi-stage build and non-root
runtime identity; install only locked runtime dependencies; copy only paths in
`runtime-allowlist.txt`; and omit tests, training commands, datasets, notebooks,
reports, caches, compilers, shells where the selected base permits, and every
private input. The runtime root filesystem should be read-only. The Phase 2B
service needs no writable request-data directory.

Private `detection.pt`, `recognition.pt`, and `bundle-manifest.json` files must
arrive through an approved read-only mount or secret-volume mechanism outside
the public build context. Readiness remains false until the manifest and both
SHA-256 values verify and the models load.

Run one process and one worker with Uvicorn access logs disabled. Binding to a
container interface is permitted only when Coolify publishes no host/public
port and connects the service solely to the future BFF on a private application
network. `/healthz` and `/readyz` are the health probes.

The Phase 2D Dockerfile is authorized only for local container integration.
Its temporary build context is constructed from this runtime allowlist; the
final image inventory and exclusion checks must be repeated for any future
host architecture or changed dependency graph. Container build did not occur
in Phase 2B, and deployment remains unauthorized.
