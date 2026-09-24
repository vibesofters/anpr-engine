# ANPR Engine

ANPR Engine is a project-owned Turkish automatic number-plate recognition
research and demonstration pipeline. It combines project Detection, geometric
crop refinement, project Recognition, structural validation, and explicit human
review. It is not represented as a high-availability or unattended production
service.

## Public candidate boundary

This source candidate does not include model weights, private datasets, real
images or transcriptions, internal experiment evidence, review exports, or
deployment credentials. The two runtime weights remain private and separately
provisioned. Their approved SHA-256 identities are documented in the curated
public model registry.

The intended runtime sequence is:

`INPUT → PROJECT DETECTION → BLUE_BAND_GLYPH_QUAD_V1 → PROJECT RECOGNITION → STRUCTURAL VALIDATION → HUMAN REVIEW`

Predictions are unresolved until a person confirms or corrects them. Structural
validity and confidence are informational and do not establish identity or
ownership.

## Documentation

- [Repository root overview](../README.md)
- [System architecture](architecture/system-architecture.md)
- [Detection architecture and cited influences](architecture/detection-architecture.md)
- [Project-model candidate](models/anpr-v1-project-models-only-release.md)
- [Recognition family summary](reference/recognition-four-family-summary.md)
- [Public capability matrix](reference/public-capability-matrix.md)
- [Recognition training workflow](training/recognition-training.md)
- [Evaluation methodology](evaluation/methodology.md)
- [Private model runtime](operations/private-model-runtime.md)
- [Privacy and responsible use](policies/privacy-and-responsible-use.md)
- [Export schema](operations/anpr-v1-project-models-only-export-schema.md)
- [Public limitations](policies/public-candidate-limitations.md)
- [Dependency review input](reference/dependency-license-review-input.md)
- [Development setup](development/setup.md)
- [Real-image processing trust boundary](architecture/real-image-processing-boundary.md)
- [English/Turkish terminology](reference/terminology.en-tr.md)

Versioned Draft 2020-12 and OpenAPI 3.1 contracts govern the single-image
JPEG/PNG boundary, privacy acknowledgement, in-memory processing, cleanup,
human review, and current-session export. The bilingual Next.js frontend and
private HTTP inference service are live as a bounded public demonstration at
`https://anpr-engine.vibesofters.com`. Human review remains mandatory. This
does not establish sustained capacity or independent field accuracy. See the
[web workflow](operations/web-frontend-bff.md) and
[live release record](operations/public-coolify-release.md).

## Local checks

```text
uvx --from uv==0.12.5 uv sync --locked --all-groups
uvx --from uv==0.12.5 uv run --locked ruff format --check src scripts tests
uvx --from uv==0.12.5 uv run --locked ruff check src scripts tests
uvx --from uv==0.12.5 uv run --locked mypy src tests packages/api-contract/generated/python
uvx --from uv==0.12.5 uv run --locked pytest
```

Private-model smoke tests additionally require `ANPR_PRIVATE_MODEL_BUNDLE` to
name an owner-controlled directory containing the approved private bundle. The
path and weights must never be committed.

## Licensing status

Project-owned software code is licensed under Apache License 2.0; see the
repository's `LICENSE` and `NOTICE`. This does not cover private model weights,
private datasets, real images/transcriptions, third-party dependencies, or the
separately licensed Barlow Condensed files.
