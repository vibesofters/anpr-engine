# Project-model result export candidate

Status: schema description for private review; not public-release approval.

The project-only review export separates model observations from human resolution. It includes project Detection and Recognition results, informational confidence, structural validation, human assessment/correction, resolution source, immutable model hashes, crop-refinement identity, safe errors, and audit-chain status.

`resolution_source` is `OWNER_CONFIRMED_PROJECT_MODEL`, `OWNER_CORRECTED`, or `UNRESOLVED`. Confidence and structural validation cannot resolve a prediction. Model weights, private checkpoint paths, source-image paths, record-level manifests, and private dataset identities are excluded from the public model registry.
