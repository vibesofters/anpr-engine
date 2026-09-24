# Dependency and license-review input

Status: review input only; no legal or Apache-2.0 compatibility conclusion.

`pyproject.toml` and `uv.lock` are the Python dependency authorities. Direct
runtime dependencies are NumPy, OpenCV (headless), Pillow, Pydantic, PyYAML, and
PyTorch. Hatchling is the build dependency. Development dependencies are
Hatchling, jsonschema, mypy, openapi-spec-validator, pre-commit, pytest, Ruff,
types-jsonschema, and types-PyYAML. `jsonschema` 4.26.0 is MIT-licensed; the
`types-jsonschema` 4.26.0.20260518 stubs are Apache-2.0-licensed. They are pinned
development-only schema-validation/type-checking dependencies.
`openapi-spec-validator` 0.9.0 is an Apache-2.0 development-only dependency used
for OpenAPI 3.1 validation. Every direct and transitive package still requires
version-specific license review.

Phase 2B adds two pinned direct runtime dependencies. Starlette 1.6.0 provides
the minimal ASGI application, routing, raw request-stream, and JSON response
layer; it declares BSD-3-Clause and brings AnyIO 4.15.1 (MIT), idna 3.20
(BSD-3-Clause), and typing-extensions 4.16.0 (PSF-2.0). Uvicorn 0.53.0 provides
the single-process ASGI server; it declares BSD-3-Clause and brings Click 8.5.0
(BSD-3-Clause) and h11 0.16.0 (MIT). No optional extras or multipart dependency
are installed. These metadata findings are license-review inputs, not a legal
or Apache-2.0 compatibility conclusion.

Phase 2C adds a separately locked `apps/web/package.json` and
`apps/web/package-lock.json`. The verified current supported stable Next.js
release used here is 16.3.6 (MIT). Direct runtime dependencies are React and
React DOM 19.3.0 (MIT) and Sharp 0.35.4 (Apache-2.0). TypeScript 6.0.3 is a
development-only Apache-2.0 dependency; ESLint 9.39.5 and
`eslint-config-next` 16.3.6 are development-only MIT dependencies, and the
three `@types/*` packages are MIT. The lockfile declares transitive
`@img/sharp-libvips-*` 1.3.3 platform
packages as LGPL-3.0-or-later, `caniuse-lite` as CC-BY-4.0, and additional
MIT, Apache-2.0, BSD-3-Clause, ISC, 0BSD, and PSF-related terms. The npm audit
reported zero known advisories at Phase 2C implementation time. These are
package metadata observations only, **not** a qualified compatibility or
redistribution determination. Final public distribution and any container
containing libvips remain blocked pending review and required notices.

The detailed [Phase 2C web redistribution review](web-dependency-redistribution-review.md)
supersedes the shorthand npm paragraph for packaging decisions. In particular,
the selected Linux `@img/sharp-libvips-*` binary package carries more than
libvips: it bundles multiple separately licensed native components, including
LGPL and MPL material. Its npm tarball has a summary README but no full
third-party license bundle. Upstream's POSIX build script statically links
libvips and other dependencies into a shared object. The final image's
architecture/libc, base-image contents, source/relinking path, notices, and
recipient scenario remain unverified. No notice bundle is claimed to exist yet.

Barlow Condensed Bold and SemiBold are used only by the synthetic-data workflow.
The font files remain with `OFL.txt`, `SOURCE.json`, and `METADATA.pb`. They are
third-party SIL OFL 1.1 material, not project-owned software, model weights, or
ANPR Engine branding. Redistribution and notice handling remain unresolved.

Private weights, private datasets, real images/transcriptions, and materials
with separate terms are excluded from the intended Apache-2.0 software scope.
