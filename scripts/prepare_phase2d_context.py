#!/usr/bin/env python3
"""Create a minimal, allowlisted Docker context outside the repository."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "deploy/inference/runtime-allowlist.txt"


def allowed_paths(service: str) -> list[Path]:
    if service == "web":
        fixed = [
            "apps/web/package.json",
            "apps/web/package-lock.json",
            "apps/web/next.config.ts",
            "apps/web/next-env.d.ts",
            "apps/web/tsconfig.json",
            "packages/api-contract/generated/typescript/contracts.ts",
            "packages/api-contract/fixtures/valid/processing-notice.json",
            "deploy/docker/web.Dockerfile",
            "deploy/docker/create_web_notices.mjs",
            "deploy/docker/THIRD-PARTY-NOTICES.sharp-libvips-v1.3.3.md",
        ]
        source = sorted((ROOT / "apps/web/src").rglob("*"))
        public = ROOT / "apps/web/public"
        if public.is_dir():
            source.extend(sorted(public.rglob("*")))
        return [ROOT / item for item in fixed] + [item for item in source if item.is_file()]
    target = "amd64" if service == "inference-amd64" else "arm64"
    dockerfile = "inference.amd64.Dockerfile" if target == "amd64" else "inference.Dockerfile"
    fixed = [
        f"deploy/docker/requirements.cpu-{target}.in",
        f"deploy/docker/requirements.cpu-{target}.txt",
        f"deploy/docker/{dockerfile}",
        "deploy/docker/create_python_notices.py",
    ]
    return [ROOT / item for item in fixed] + [
        ROOT / item for item in RUNTIME.read_text().splitlines() if item
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("web", "inference", "inference-amd64"))
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error("destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    paths = allowed_paths(args.service)
    for original in paths:
        resolved = original.resolve(strict=True)
        if not resolved.is_relative_to(ROOT) or not original.is_file() or original.is_symlink():
            parser.error("unsafe context entry")
        relative = original.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    if any(
        part in {"artifacts", "datasets", ".worktrees", "node_modules", ".next"}
        for path in destination.rglob("*")
        for part in path.relative_to(destination).parts
    ):
        parser.error("excluded path entered build context")
    print(f"{args.service} context: {len(paths)} allowlisted files")


if __name__ == "__main__":
    main()
