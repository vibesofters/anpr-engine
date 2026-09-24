# Phase 2C web dependency redistribution review

Status: **OWNER-REPORTED REVIEW; LOCAL PHASE 2D AUTHORIZED** (2026-09-23).
The repository owner reports that the Sharp/libvips licensing question was
reviewed and permits Phase 2D. This is an owner-reported legal decision, not
an independent legal conclusion by the implementation agent. Do not reopen
the same approval gate without a new concrete finding. The inventory below
is the pre-build technical assessment; the Phase 2D image-specific inventory
supersedes its predictions. The Coolify host remains uninventoried.

## Evidence and method

- Authority for the exact npm graph: `apps/web/package.json` and
  `apps/web/package-lock.json` (lockfile version 3). The lock contains **54**
  non-development package entries, including mutually exclusive optional
  platform packages. A lock entry is not proof that a package is installed in
  a particular image. The macOS development installation is not a Linux image.
- Primary distributions inspected: installed package licenses/READMEs,
  npm's file manifest for `@img/sharp-libvips-linux-x64@1.3.3` and
  `@img/sharp-linux-x64@0.35.4`, and the Linux x64 libvips package's
  `versions.json`. The Linux x64 libvips tarball contains `README.md`,
  `versions.json`, `lib/index.js`, a GLib header, package metadata, and
  `lib/libvips-cpp.so.8.18.6` (18,621,496 bytes). **It contains no separate
  license text or `THIRD-PARTY-NOTICES.md`.** The Sharp native addon package
  contains `LICENSE` and `lib/sharp-linux-x64-0.35.4.node`.
- [Sharp's install guide](https://sharp.pixelplumbing.com/install/) says npm
  selects prebuilt platform packages at install time and documents glibc/musl,
  CPU, and lockfile caveats. The tagged
  [sharp-libvips packaging source](https://github.com/lovell/sharp-libvips/blob/v1.3.3/README.md),
  [build script](https://github.com/lovell/sharp-libvips/blob/v1.3.3/build/posix.sh),
  and [third-party notice summary](https://github.com/lovell/sharp-libvips/blob/v1.3.3/THIRD-PARTY-NOTICES.md)
  identify the native bundle's build and component families. The script uses
  static dependency builds and explicitly links libvips statically into the
  distributed libvips-cpp shared object. This is materially different from
  assuming each LGPL component is a separately replaceable `.so`.
- [Next.js deployment documentation](https://nextjs.org/docs/app/api-reference/config/next-config-js/output)
  distinguishes a full Node install from traced standalone output. This
  repository has **not** selected either final image layout. Next also vendors
  compiled components inside its package; the npm graph alone does not fully
  enumerate these or the browser bundle. The final build requires an SBOM,
  browser-asset inventory, native-binary scan, and notice verification.

## Locked npm production graph

All entries below are exact versions from the lockfile. Licenses are corroborated
where local license files exist; otherwise these are **package-declared
licenses requiring distribution-file verification**, not an approval. `D`
means direct, `T` transitive, `O` optional. For MIT, Apache-2.0, ISC,
BSD-3-Clause, and 0BSD components, preserve each package's actual copyright
and license text where distributed; do not replace distinct copyright notices
with a generic license label. Proposed fulfillment: the package's original
license file in the installed image **and** a generated, version-pinned
`/licenses/npm/` notice bundle alongside the image. This bundle does not yet
exist and must be generated and checked against the final image. The public
repository review record is not a substitute for notices in a distributed image.

| Package | Version | Role | Declared license |
| --- | --- | --- | --- |
| next | 16.3.6 | D | MIT |
| react; react-dom | 19.3.0 each | D | MIT |
| sharp | 0.35.4 | D | Apache-2.0 |
| @next/env | 16.3.6 | T | MIT |
| @swc/helpers | 0.5.23 | T | Apache-2.0 |
| @img/colour | 1.1.0 | T | MIT |
| baseline-browser-mapping | 2.11.25 | T | Apache-2.0 |
| caniuse-lite | 1.0.30001810 | T | CC-BY-4.0 |
| client-only | 0.0.1 | T | MIT |
| detect-libc | 2.1.2 | T | Apache-2.0 |
| nanoid | 3.3.19 | T | MIT |
| picocolors | 1.1.1 | T | ISC |
| postcss | 8.5.23 | T | MIT |
| scheduler | 0.28.0 | T | MIT |
| semver | 7.8.5 | T | ISC |
| source-map-js | 1.2.1 | T | BSD-3-Clause |
| styled-jsx | 5.1.6 | T | MIT |
| tslib | 2.8.1 | T | 0BSD |
| @emnapi/runtime | 1.11.3 | O | MIT |

`caniuse-lite`'s installed `LICENSE` is CC-BY-4.0, not a software MIT
license. Its package metadata identifies Ben Briggs as author and
`browserslist/caniuse-lite` as source; this is an attribution input, not a
final determination of all data creators. If its database or derived material
is actually conveyed, determine the credited creator, source link, license
link, changes statement, and
placement under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).
It may be build-only despite its non-dev lock classification; verify whether
it exists in the final image **or browser assets** before deciding the notice.

### Optional platform binaries in the lock

The lock includes all of the following alternatives, not all of which would
be installed. Only the selected target's entries belong in a final inventory.

| Package family | Locked version | Platforms | Declared license / planned role |
| --- | --- | --- | --- |
| `@img/sharp-linux-*` | 0.35.4 | arm, arm64, ppc64, riscv64, s390x, x64 (glibc) | Apache-2.0; native `.node` addon |
| `@img/sharp-linuxmusl-*` | 0.35.4 | arm64, x64 (musl) | Apache-2.0; native `.node` addon |
| `@img/sharp-libvips-linux-*` | 1.3.3 | arm, arm64, ppc64, riscv64, s390x, x64 (glibc) | LGPL-3.0-or-later package declaration; bundled native library and dependencies |
| `@img/sharp-libvips-linuxmusl-*` | 1.3.3 | arm64, x64 (musl) | LGPL-3.0-or-later package declaration; bundled native library and dependencies |
| `@next/swc-linux-{arm64,x64}-{gnu,musl}` | 16.3.6 | Linux arm64/x64, glibc/musl | MIT; native compiler addon, possibly build-only in final layout |
| `@img/sharp-darwin-{arm64,x64}` and matching `sharp-libvips` | 0.35.4 / 1.3.3 | macOS | Apache-2.0 / LGPL-3.0-or-later; development host, not a Linux deployment |
| `@next/swc-darwin-{arm64,x64}` | 16.3.6 | macOS | MIT; development/build host |
| `@img/sharp-{freebsd,wasm32,webcontainers-wasm32,win32-*}`; `@next/swc-win32-*` | 0.35.4 / 16.3.6 | non-Linux alternatives | mixed Apache-2.0, LGPL-3.0-or-later, MIT as recorded per lock entry; exclude from a Linux image unless unexpectedly present |

The selected Linux Sharp addon, libvips package, and any SWC addon need their
own exact npm integrity, native file list, linked-library list, and preserved
license evidence recorded after the architecture/libc and packaging strategy
are chosen. Do not ship cross-platform optional packages merely because the
lock mentions them. A base image may additionally contain Node, OpenSSL,
glibc/musl, CA data, OS packages, and other licensed binaries **outside npm's
lockfile**; these cannot be inventoried before the base image is chosen.

## Native libvips bundle: components and source

The Linux x64 package's `versions.json` gives the versions below; the tagged
[upstream notice summary](https://github.com/lovell/sharp-libvips/blob/v1.3.3/THIRD-PARTY-NOTICES.md)
gives license families, not every full copyright/license text. The tagged
[build script](https://github.com/lovell/sharp-libvips/blob/v1.3.3/build/posix.sh)
is the source/build provenance entry point. These are components **inside**
the native bundle, not separate npm dependencies. The values must be
reconfirmed for any non-x64 target rather than inferred from this table.

| Component(s), Linux x64 version | Upstream-notice license family |
| --- | --- |
| libvips 8.18.6; GLib 2.89.4; fribidi 1.0.16; libexif 0.6.26; libheif 1.23.2; librsvg 2.62.91; pango 1.58.2; proxy-libintl 0.5 | LGPLv3 choice (the upstream notice explains later-version election) |
| cairo 1.18.4 | MPL-2.0 |
| aom 3.15.0 | BSD-2-Clause plus AOM Patent License 1.0 |
| cgif 0.5.3; expat 2.8.3; harfbuzz 14.3.1; lcms 2.19.1; libffi 3.8.0; libnsgif (version not separately listed); libultrahdr 2.0.2; libxml2 2.15.3 | MIT |
| highway 1.4.0 | BSD-3-Clause |
| libarchive 3.8.9; libimagequant 2.4.1 | BSD-2-Clause |
| fontconfig 2.18.3 | fontconfig license (BSD-like) |
| freetype 2.14.3 | FreeType license |
| libpng 1.6.58 | libpng license |
| libtiff 4.7.2 | libtiff license (BSD-like) |
| libwebp 1.6.0 | new BSD license |
| mozjpeg build `0826579` | zlib, IJG, BSD-3-Clause |
| pixman 0.46.4 | MIT |
| zlib-ng 2.3.3 | zlib license |

This table is not a determination that all source, notice, patent, or modified
build obligations are met. In particular, the npm libvips tarball's short
README table does not supply full GPL/LGPL/MPL texts, component copyrights,
or a version-matched corresponding-source/relinking offer.

## Required packaging plan, conditional on lawyer review

1. Select the actual CPU architecture, libc, base image, Node version,
   install mode, and whether Next uses full `node_modules` or standalone.
   Exclude development-only dependencies from the runtime image; if the build
   toolchain is copied in instead, inventory and notice it as shipped content.
   Produce the final file/SBOM inventory, including native files and shipped
   browser JavaScript, from that exact image. Preserve npm integrity values
   and hash each native binary. **Do not build the image in this review.**
2. Put a version-specific `/licenses/npm/` bundle in any distributed image
   containing the exact copyright/license/NOTICE files of every included npm
   package, including vendored Next components and browser assets as applicable.
   Keep the package's original license files when present. For
   `caniuse-lite`, add a CC-BY-4.0 attribution only if material is conveyed,
   after tracing it.
3. Put a separate `/licenses/native/` bundle with the selected libvips
   package's upstream third-party notice, full applicable license texts and
   component copyright notices. Preserve Apache-2.0's Sharp/addon notices.
   This is a **proposed location**, not a completed notice package.
4. For LGPL components, ask counsel to choose and document the compliant
   corresponding-source and recombination/relinking route for this **statically
   aggregated** native `.so`, and whether a source offer, build scripts,
   object/application material, or installation information is required for
   the exact recipient scenario. The [GNU LGPLv3 combined-work terms](https://www.gnu.org/licenses/lgpl%2Bgpl-3.0-standalone.html)
   describe notice, license, source/relinking, and possible installation duties;
   no assertion is made that one option is already satisfied. For cairo,
   assess [MPL-2.0 source availability](https://www.mozilla.org/en-US/MPL/2.0/)
   for any conveyed executable and preserve its notices.
5. Verify all notices and source/relinking materials **inside or accompanying**
   the exact distributed image; if the service is only remotely accessed,
   distinguish server delivery to an infrastructure operator from browser
   delivery to public users. Do not infer that hosting automatically eliminates
   obligations. Repeat review if versions, binary provenance, base image, or
   distribution method changes.

## Specific question and evidence bundle for qualified counsel

**Question:** For an ANPR Engine Next.js server image containing unmodified
`sharp@0.35.4` and the selected `@img/sharp-libvips-<Linux target>@1.3.3`, with
libvips 8.18.6 and multiple LGPL libraries statically linked into
`libvips-cpp.so.8.18.6`, what notices, license texts, corresponding source,
relinkable material, source offer, and installation information must accompany
(a) delivery of the image to a hosting/operator entity and (b) any later
public image distribution? Does the upstream source/build repository alone
satisfy any of those duties? Also confirm MPL-2.0 cairo and CC-BY-4.0
`caniuse-lite` handling for the exact contents shipped to recipients.

The owner reports that this question was reviewed and approved for local
Phase 2D. The following evidence list is retained for traceability, not as a
request to repeat the same review absent a new concrete finding.

Provide counsel if a new finding arises: this record; `apps/web/package.json` and lockfile;
the selected npm tarball manifests and `versions.json`; the tagged upstream
build script and third-party notices linked above; the exact eventual SBOM,
native hashes and dependency scan; base-image manifest; and a diagram of who
receives image copies versus browser assets. The agent has not received the
reviewer's written determination or independently verified legal sufficiency.
Phase 2D local container work is owner-authorized; public distribution and
deployment have separate gates.
