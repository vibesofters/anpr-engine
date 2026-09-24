# Contributing

Thank you for helping improve ANPR Engine. Use an issue to discuss a proposed change before investing in a large pull request. Normal changes go through a pull request with passing CI and resolved review conversations. Security reports belong in [SECURITY.md](SECURITY.md), never in public issues.

Keep contributions within the project-owned Detection and Recognition pipeline. Do not add pretrained or external OCR models, private model weights, real plate images or transcriptions, dataset manifests, review exports, secrets, absolute workstation paths, or third-party material whose redistribution terms have not been checked. Use fictional synthetic fixtures only. Do not include claims about field accuracy or production reliability without approved evidence.

For a code change, explain its purpose, affected contract, risks, and tests. Run the relevant checks described in [development setup](docs/development/setup.md), and include or update focused tests and documentation. Changes to model identities, public API or export schemas, privacy boundaries, dependencies, licensing, and deployment settings require explicit maintainer review. A pull request is not authorization to publish private data or weights.

By contributing, you confirm you have the right to submit your work under the repository's applicable license once that license is finalized. Until then, do not infer that private model weights or real datasets are included in any software-license grant.
