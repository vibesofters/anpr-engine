import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { createHash } from "node:crypto";

const root = "/build/apps/web";
const standalone = join(root, ".next/standalone");
const destination = "/build/licenses/npm";
mkdirSync(destination, { recursive: true });
const packages = [];
const native_artifacts = [];
const upstreamLicense = {
  "@edge-runtime/cookies": "https://raw.githubusercontent.com/vercel/edge-runtime/440c123a37284d6a852ce453af810ad484ecfc01/LICENSE.md",
  "@edge-runtime/ponyfill": "https://raw.githubusercontent.com/vercel/edge-runtime/440c123a37284d6a852ce453af810ad484ecfc01/LICENSE.md",
  "@edge-runtime/primitives": "https://raw.githubusercontent.com/vercel/edge-runtime/440c123a37284d6a852ce453af810ad484ecfc01/LICENSE.md",
  "@next/env": "https://raw.githubusercontent.com/vercel/next.js/v16.3.6/license.md",
  "client-only": "https://raw.githubusercontent.com/facebook/react/v18.2.0/LICENSE",
  "regenerator-runtime": "https://raw.githubusercontent.com/facebook/regenerator/v0.13.4/LICENSE",
};

// Next's file tracer may copy optional native targets. Keep exactly the
// selected Linux/glibc pair for this pinned base architecture.
const architecture = process.arch === "arm64" ? "arm64" : process.arch === "x64" ? "x64" : null;
if (process.platform !== "linux" || !architecture) throw new Error("unsupported_web_target");
const imageDirectory = join(standalone, "apps/web/node_modules/@img");
if (existsSync(imageDirectory)) {
  const selected = new Set([`sharp-linux-${architecture}`, `sharp-libvips-linux-${architecture}`]);
  for (const name of readdirSync(imageDirectory)) {
    if (name.startsWith("sharp-") && !selected.has(name)) {
      rmSync(join(imageDirectory, name), { recursive: true, force: true });
    }
  }
}
rmSync(join(standalone, "apps/web/node_modules/@emnapi/runtime"), { recursive: true, force: true });

function visit(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) visit(path);
    else if (entry.name === "package.json" && path.includes("/node_modules/")) {
      const metadata = JSON.parse(readFileSync(path, "utf8"));
      if (!metadata.name || !metadata.version) continue;
      const installed = join(root, "node_modules", metadata.name);
      const original = existsSync(installed) ? installed : dirname(path);
      const output = join(destination, metadata.name.replaceAll("/", "__") + "@" + metadata.version);
      mkdirSync(output, { recursive: true });
      const copied = [];
      for (const file of readdirSync(original)) {
        if (/^(licen[sc]e|copying|notice)(\.|$)/i.test(file)) {
          cpSync(join(original, file), join(output, file));
          copied.push(file);
        }
      }
      if (metadata.name.startsWith("@img/sharp-libvips-")) {
        cpSync(join(original, "README.md"), join(output, "README.md"));
        cpSync("/build/deploy/docker/THIRD-PARTY-NOTICES.sharp-libvips-v1.3.3.md", join(output, "THIRD-PARTY-NOTICES.md"));
        copied.push("README.md", "THIRD-PARTY-NOTICES.md");
        if (existsSync(join(original, "versions.json"))) {
          cpSync(join(original, "versions.json"), join(output, "versions.json"));
          copied.push("versions.json");
        }
      }
      function scanNative(dir) {
        for (const child of readdirSync(dir, { withFileTypes: true })) {
          const file = join(dir, child.name);
          if (child.isDirectory()) scanNative(file);
          else if (/\.(?:node|so(?:\.[0-9.]+)?)$/.test(child.name)) {
            native_artifacts.push({ package: metadata.name, path: relative(standalone, file), sha256: createHash("sha256").update(readFileSync(file)).digest("hex") });
          }
        }
      }
      scanNative(dirname(path));
      packages.push({ name: metadata.name, version: metadata.version, declared_license: metadata.license ?? null, copied });
    }
  }
}

visit(standalone);
for (const item of packages) {
  if (item.copied.length === 0 && upstreamLicense[item.name]) {
    const url = upstreamLicense[item.name];
    const response = await fetch(url);
    if (!response.ok) throw new Error(`upstream_license_unavailable:${item.name}`);
    const output = join(destination, item.name.replaceAll("/", "__") + "@" + item.version);
    writeFileSync(join(output, "UPSTREAM-LICENSE.txt"), await response.text());
    writeFileSync(join(output, "SOURCE-URL.txt"), url + "\n");
    item.copied.push("UPSTREAM-LICENSE.txt", "SOURCE-URL.txt");
    item.upstream_license_note = item.name === "client-only"
      ? "Associated React upstream license, not a version-matched license file from the npm tarball"
      : "Version-tagged upstream license; npm tarball omits a license file";
  }
  if (item.name.startsWith("@img/sharp-libvips-")) {
    const url = "https://raw.githubusercontent.com/lovell/sharp-libvips/v1.3.3/THIRD-PARTY-NOTICES.md";
    const response = await fetch(url);
    if (!response.ok) throw new Error("libvips_upstream_notice_unavailable");
    const output = join(destination, item.name.replaceAll("/", "__") + "@" + item.version);
    writeFileSync(join(output, "UPSTREAM-THIRD-PARTY-NOTICES.md"), await response.text());
    writeFileSync(join(output, "UPSTREAM-SOURCE-URL.txt"), url + "\n");
    item.copied.push("UPSTREAM-THIRD-PARTY-NOTICES.md", "UPSTREAM-SOURCE-URL.txt");
  }
  if (item.name === "next") {
    const compiled = join(standalone, "apps/web/node_modules/next/dist/compiled");
    if (existsSync(compiled)) {
      const source = join(root, "node_modules/next/dist/compiled");
      const target = join(destination, "next@" + item.version, "compiled");
      function licenses(dir) {
        for (const entry of readdirSync(dir, { withFileTypes: true })) {
          const path = join(dir, entry.name);
          if (entry.isDirectory()) licenses(path);
          else if (/^(licen[sc]e|copying|notice)(\.|$)/i.test(entry.name)) {
            const out = join(target, relative(source, path));
            mkdirSync(dirname(out), { recursive: true });
            cpSync(path, out);
          }
        }
      }
      licenses(source);
    }
  }
}
packages.sort((a, b) => a.name.localeCompare(b.name));
writeFileSync("/build/licenses/inventory.json", JSON.stringify({ schema_version: "anpr-web-image-license-inventory-v1", platform: `linux/${architecture === "x64" ? "amd64" : "arm64"}/glibc`, packages, native_artifacts }, null, 2) + "\n");
if (!packages.some((item) => item.name === "sharp") ||
    !packages.some((item) => item.name === `@img/sharp-libvips-linux-${architecture}`) ||
    !packages.some((item) => item.name === `@img/sharp-linux-${architecture}`)) {
  throw new Error("native_image_dependencies_not_traced");
}
console.log(`Collected notices for ${packages.length} traced npm packages`);
