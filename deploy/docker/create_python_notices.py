"""Collect package-shipped license files for the exact installed virtualenv."""

from __future__ import annotations

import json
import hashlib
import platform
import shutil
import sys
from pathlib import Path

environment, destination = (Path(value) for value in sys.argv[1:3])
site = next((environment / "lib").glob("python*/site-packages"))
destination.mkdir(parents=True, exist_ok=True)
packages = []
for metadata in sorted(site.glob("*.dist-info")):
    record = metadata / "METADATA"
    name = metadata.name.removesuffix(".dist-info")
    license_files = []
    output = destination / name
    for source in metadata.rglob("*"):
        if source.is_file() and any(word in source.name.lower() for word in ("license", "copying", "notice")):
            target = output / source.relative_to(metadata)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            license_files.append(str(target.relative_to(destination)))
    declared_license = None
    if record.is_file():
        for line in record.read_text(errors="replace").splitlines():
            if line.startswith("License-Expression: "):
                declared_license = line.removeprefix("License-Expression: ")
                break
    packages.append({"distribution": name, "license_expression": declared_license, "license_files": license_files})
native_artifacts = []
for source in sorted(environment.rglob("*")):
    if not source.is_file():
        continue
    with source.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            continue
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    native_artifacts.append({
        "path": str(source.relative_to(environment)),
        "size_bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
    })
(destination / "inventory.json").write_text(json.dumps({
    "schema_version": "anpr-python-image-license-inventory-v1",
    "platform": f"linux/{'amd64' if platform.machine() == 'x86_64' else 'arm64' if platform.machine() == 'aarch64' else platform.machine()}/glibc",
    "packages": packages,
    "native_artifacts": native_artifacts,
}, indent=2) + "\n")
print(f"Collected licenses for {len(packages)} Python distributions")
