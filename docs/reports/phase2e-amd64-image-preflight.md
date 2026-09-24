# Phase 2E Linux amd64 image preflight

Date: 2026-09-23. Status: **AWAITING_MANUAL_STAGING_MEASUREMENTS**.
These are off-host Docker Desktop cross-build facts, **not** Coolify host
performance or a production qualification. The owner reported prior
Sharp/libvips legal review permitting Phase 2D/2E work; this report is a
technical inventory, not an independent legal conclusion.

| Code-only image | Local image ID | Size | Runtime |
| --- | --- | ---: | --- |
| `anpr-engine-web:phase2e-amd64-local` | `sha256:fc1d4814d2f99f7d4dddc2d49a27b0d518fc9132a475abc1aead2cc441c5eb86` | 92,277,281 bytes | Linux amd64, Debian/glibc, non-root UID/GID 10001, Node 22.23.2, Next 16.3.6. |
| `anpr-engine-inference:phase2e-amd64-local` | `sha256:743ff2f1413164ca69098f548873e7bf862fcfb7fcae6d13aeb8f8c6065d3964` | 309,704,282 bytes | Linux amd64, Debian/glibc, non-root UID/GID 10001, Python 3.12.14, CPU-only PyTorch 2.13.0+cpu, OpenCV 4.12.0.88. |

Both were built from the approved source with 22 web and 52 inference
allowlisted context files, pinned base-image digests, locked dependencies,
and no model weights, datasets, experiment artifacts, training entrypoints,
secrets, or external OCR code. The amd64 PyTorch wheel is hash-locked in
`deploy/docker/requirements.cpu-amd64.txt`. The inference runtime excludes
PyTorch development headers, share files, and test executables; its CPU
import succeeded under local amd64 emulation with CUDA unavailable. Local
emulation timing is deliberately **not** reported as host performance.
An ephemeral local amd64 container with the approved private bundle mounted
read-only and no network or published port returned HTTP 200 from `/readyz`,
which requires the two model hashes and model loading to pass. That container
was stopped and removed. This is only an amd64 compatibility smoke check;
it is not a Coolify resource measurement.

The final web image's `/licenses/inventory.json` records 18 traced npm
packages and exactly two **npm-supplied** native binaries; Node and Debian
base binaries are accounted for by their pinned runtime/base versions and
versioned license/package records rather than this npm-native count. No arm64, musl, or WASM Sharp
binary was found in the runtime. Their selected hashes are:

| Native binary | SHA-256 |
| --- | --- |
| `@img/sharp-libvips-linux-x64@1.3.3` / `libvips-cpp.so.8.18.6` | `536cee19ab906cb5185ad519bf87647693f5d7b55e608c92eb28607f89ca5125` |
| `@img/sharp-linux-x64@0.35.4` / `sharp-linux-x64-0.35.4.node` | `649fd3c3f98401061963aa496976c21b120a67c382657d9a26a60df251a317e2` |

The web notice bundle includes each installed npm component's license or
qualified upstream-license source; the selected libvips package's version
manifest, bundled-component summary, and tagged upstream third-party notice;
Node 22.23.2's versioned license; and 88 Debian package records with 86
copyright files plus alias mapping. The Python bundle records all 25
installed distributions with nonempty, present license files and 190
Python-environment ELF native artifacts with individual SHA-256 values. It also includes the
Python 3.12.14 license and 105 Debian package records with 102 copyright
files plus alias mapping. These in-image notices are evidence of what was
packaged; any later distribution must still honor the owner-reviewed
licensing obligations.

Only the Coolify host can establish actual container limits, model-load
memory, request latency, co-hosted impact, proxy buffering, and safe
timeouts. The supplied host snapshot (x86_64, 8 vCPUs, 15 GiB RAM with
8.6 GiB available at observation, no swap, 71 GB free root storage,
Docker 29.4.2, 51 active containers, load 1.39/1.40/1.36) is not a
benchmark. OS/Coolify versions and enforced limit support remain pending.
The local arm64 Phase 2D images are not native to that host; these amd64
images have not been loaded or run there. The owner-operated, loopback-only
measurement procedure is in
`docs/operations/coolify-manual-staging-runbook.md`.
The staging Compose file passed configuration validation with inert
placeholder values: inference has no host port and an internal-only network;
web publishes only `127.0.0.1:39083`; both use read-only roots, bounded
memory-backed `/tmp`, non-root users, no restart, no image pull, and a
combined protective cap of 1 CPU and 2.5 GiB. Those caps are provisional
**test safety limits**, not measured safe operating limits.
