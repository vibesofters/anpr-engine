# Private model runtime configuration

## Legacy local browser disposition

`scripts/launch_anpr_v1_browser.py` is owner/development tooling only. It binds
to `127.0.0.1` and retains multipart, multi-image batch, artifact, review, and
local persistence behavior for controlled evidence work. It is not the future
public Next.js service and does not implement the Phase 2A public request,
retention, route, logging, or concurrency contracts.

The launcher, `anpr_engine.integration.browser`, its embedded UI, and its review
store are prohibited public deployment/container entry points. Their exclusion
is machine-tested through
`packages/api-contract/policy/public-deployment-exclusions.json`. Do not expose
the loopback service through a proxy, tunnel, port publication, or non-loopback
binding. Local artifacts remain the owner's responsibility and must never be
treated as public-service retention behavior.

Model weights are not distributed in this repository. The local launcher
requires an explicit `--model-bundle` directory. The directory is expected to
contain exactly:

- `detection.pt`
- `recognition.pt`
- `bundle-manifest.json`

The runtime verifies the manifest classification and both approved SHA-256
identities before becoming healthy. Missing or mismatched files fail startup and
readiness. Health responses expose only immutable public model identities, never
the private filesystem location.

Example with a non-secret placeholder:

```text
python scripts/launch_anpr_v1_browser.py \
  --repository . \
  --model-bundle /path/to/owner-controlled-model-bundle \
  --device cpu \
  --no-open-browser
```

The bundle path is local runtime configuration. Do not commit it, place weights
inside the Git tree, or treat the software license as a model-weight license.
