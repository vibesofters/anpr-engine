# Phase 2D local container integration evidence

Date: 2026-09-23. Scope: Linux arm64/glibc on local Docker Desktop only.
This is not Coolify qualification, public distribution, or deployment.

The owner reports that the Sharp/libvips licensing question was reviewed and
permits this phase. This is not an independent legal determination. The
in-image evidence below does not, by itself, establish legal sufficiency for
later distribution.

## Built-image inventory

| Runtime image | Local image ID | Size | Contents and notice evidence |
| --- | --- | ---: | --- |
| `anpr-engine-web:phase2d-local` | `sha256:aae869d811673e8b6074d739541987476ba8527f430450c101e725a5d6afdc65` | 92,558,618 bytes | Node 22.23.2, Next 16.3.6 standalone, 18 traced npm packages, selected `sharp@0.35.4` and `@img/sharp-libvips-linux-arm64@1.3.3`, 88 Debian packages. `/licenses/inventory.json`, `/licenses/npm/`, `/licenses/base/NODE-LICENSE`, 86 Debian copyright files, Debian doc-alias map, and version inventory. |
| `anpr-engine-inference:phase2d-local` | `sha256:f60de9197b793294c0db62caf7423717c6b052bdfd664610e329847f9c083eb0` | 270,441,038 bytes | Python 3.12.14, 25 installed Python distributions including `torch==2.13.0+cpu` and OpenCV 4.12.0, 105 Debian packages. `/licenses/inventory.json`, distribution license files, `/licenses/base/PYTHON-LICENSE`, 103 Debian copyright files, Debian doc-alias map, and version inventory. |

Both final image configurations run as UID/GID 10001. Docker build contexts
contained 22 and 52 allowlisted files respectively, with no weights,
datasets, experiment artifacts, training entrypoints, local review browser,
external OCR implementation, or secrets. Runtime image inspection found no
model weights, dataset/artifact tree, or copied private paths. The private
models exist only on the read-only runtime mount.

The web image's native inventory records:

| Selected native file | SHA-256 |
| --- | --- |
| `@img/sharp-libvips-linux-arm64@1.3.3/lib/libvips-cpp.so.8.18.6` | `264d3092d69de80f5acdb71c930efec8db5bd9627f41659ed3416566b9ae34b4` |
| `@img/sharp-linux-arm64@0.35.4/lib/sharp-linux-arm64-0.35.4.node` | `4f53dd2f4ce3ca21ec01f908f7c475d9ca0c9d0d6555180abcbd6e8e4f09722d` |

The libvips package's `versions.json` identifies libvips 8.18.6 and its
bundled component versions. Its image notice directory includes that exact
file, the package README, our versioned component summary, and the publisher's
tagged `v1.3.3` third-party notice. The selected native addon has its original
Apache-2.0 `LICENSE`. No musl, WASM, or other architecture's Sharp binary
is present. The 18-package npm inventory records each actual traced package,
version, declared license, and bundled license/notice location. Where npm
tarballs omitted license text, the image includes an upstream license source
and URL with an explicit provenance qualification; the edge-runtime and
`client-only` upstream files are **not** represented as version-matched npm
tarball licenses. The edge-runtime source is pinned to the commit resolved
by its three exact package-version tags. Node and Python licenses are copied from their official
version tags; Debian copyright records are preserved from the pinned bases.
The base images also contain standard runtime tooling inherited from their
upstream image layers; package-manager presence must not be mistaken for an
application dependency.

The owner-reported Sharp/libvips review permits this local phase. Any source,
relinking, or recipient-specific obligations for a **later distributed** image
must be fulfilled according to that review; an upstream URL and this report
are not themselves a corresponding-source or relinking package. A changed
binary, architecture, or distribution scenario requires an image-specific
inventory and review of the new facts.

## Integration and boundary results

- Actual approved Detection and Recognition SHA-256 identities verified at
  startup. A synthetic, non-identifying vehicle/plate image completed browser
  → same-origin BFF → private Python service → Detection → crop refinement →
  Recognition → browser review. The result returned a Detection box and a
  Recognition prediction; it was initially unresolved. Browser confirmation,
  correction, unresolved state, session clearing, versioned JSON download,
  and annotated PNG download succeeded. English and Turkish page titles and
  flows were checked. No real image was used.
- A blank synthetic image returned `no_plate_detected` through the actual
  model pair. Concurrent same-session submissions returned one `200` and one
  `409`. Missing acknowledgement returned `400`; malformed PNG returned
  `422`; an unavailable private service returned safe `503 MODEL_NOT_READY`.
  Focused tests additionally cover timeouts, malformed and oversized inputs,
  incorrect origins, duplicate requests, and model failure handling.
- A separate negative container mounted the approved private bundle read-only
  but overlaid the Detection checkpoint with a non-model device file. Its
  `/readyz` returned `503`, while unit tests independently check the exact
  Detection and Recognition hash mismatch exceptions. No private checkpoint
  was changed or copied for this test.
- The web container publishes only `127.0.0.1:3000`. Inference publishes no
  host port and belongs only to the internal application network. Host
  connection to `127.0.0.1:8080` failed. Both runtime roots are read-only;
  only bounded `/tmp` memory filesystems are writable. `docker diff` showed
  no web changes and only the inference mount point, not retained request
  artifacts. Logs were checked for the synthetic plate canaries, local host
  paths, bearer headers, filenames, media payload markers, predictions, and
  corrections; none matched.
- Verification: 111 focused Python tests passed; Next.js TypeScript check,
  lint, and web flow test passed; production Next build passed; public-candidate
  validation passed (54 JSON/JSONL files, 13 YAML files, 8 schemas, 2 OpenAPI
  contracts, 36 documentation links, 216 public files scanned). Browser
  integration was exercised with Playwright against the final rebuilt images.
  The exact final image IDs above also passed a final real-model synthetic
  smoke test. `docker compose down` removed both containers and networks;
  the loopback web port then refused connections. The browser test session was
  closed and its locally generated synthetic download/snapshot files were
  removed after verification.

## Remaining gates

The Coolify host has not been inventoried. Its architecture, libc, CPU, RAM,
GPU/runtime status, Docker version, storage, competing workloads, and actual
capacity are unknown. Phase 2E must benchmark this model pair on that host,
choose final resource and timeout limits, verify reverse-proxy upload
buffering and deadlines, reproduce image inventories and notices for the
target architecture, and verify domain/TLS, privacy, security, and recovery
behavior. Public repository release, image distribution, and deployment
remain subject to their separate approval gates.
