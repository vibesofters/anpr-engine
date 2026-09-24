# API contract package

The JSON Schemas under `schemas/` and the OpenAPI 3.1 document at
`schemas/api/v1/anpr-inference-v1.openapi.yaml` are the source of truth. Run:

```text
uv run python scripts/generate_api_contracts.py
uv run python scripts/generate_api_contracts.py --check
uv run python scripts/validate_public_candidate.py schemas
uv run python scripts/validate_public_candidate.py openapi
```

The generator derives deterministic Python `TypedDict` and TypeScript
interface/type representations with the standard library, then applies the
pinned Ruff formatter to its Python output. Generated files must not be edited
by hand. Phase 2A deliberately introduces no Node.js, TypeScript runtime,
Next.js, ASGI framework, or HTTP service.

`jsonschema` 4.26.0 is a pinned MIT-licensed development dependency for Draft
2020-12 validation. `openapi-spec-validator` 0.9.0 is a pinned Apache-2.0
development dependency for OpenAPI 3.1 validation. Neither is required by the
ANPR model runtime. Final repository-wide license compatibility review remains
open.

Contract fixtures are small text records or synthetic, non-identifying images
created in memory by tests. They contain no private real images, TEST data, or
independent holdout material.
