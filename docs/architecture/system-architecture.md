# System architecture

ANPR Engine separates project-owned model code from private trained parameters
and data. The local research pipeline is:

`input → Detection → crop refinement → Recognition → structural validation → human review`

Detection proposes plate regions. The deterministic crop-refinement stage may
adjust the selected region. Recognition produces character and type evidence.
The profile and decoder report structural validity without treating it as proof
of identity. A human then confirms, corrects, or leaves the result unresolved.

The retained Python review browser is local-only and is not the public-service
architecture. The uncommitted local public candidate now has a bilingual
Next.js frontend/BFF and a separate private Python inference service. Only the
BFF may receive public image requests in a future, separately approved
deployment. No container, network placement, or public operation is approved.

Private weights are supplied at runtime from outside Git and verified against
immutable public SHA-256 identities. Datasets, record-level manifests, real
transcriptions, and exact evaluation populations remain outside the public
candidate.

See [private model runtime](../operations/private-model-runtime.md) and
[privacy and responsible use](../policies/privacy-and-responsible-use.md).
The [web boundary](../operations/web-frontend-bff.md) describes Phase 2C.
