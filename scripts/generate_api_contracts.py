#!/usr/bin/env python3
"""Generate deterministic public contract types from the Phase 2A JSON Schemas."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
SCHEMA_PATHS = (
    ROOT / "schemas/api/v1/anpr-image-inference-metadata-v1.schema.json",
    ROOT / "schemas/api/v1/anpr-image-inference-response-v1.schema.json",
    ROOT / "schemas/api/v1/anpr-api-error-v1.schema.json",
    ROOT / "schemas/exports/v1/anpr-session-result-v1.schema.json",
    ROOT / "schemas/review/v1/anpr-human-review-v1.schema.json",
    ROOT / "schemas/privacy/v1/anpr-processing-notice-v1.schema.json",
    ROOT / "schemas/privacy/v1/anpr-data-lifecycle-v1.schema.json",
    ROOT / "schemas/internal/v1/anpr-private-inference-response-v1.schema.json",
)
PYTHON_OUTPUT = ROOT / "packages/api-contract/generated/python/contracts.py"
TYPESCRIPT_OUTPUT = ROOT / "packages/api-contract/generated/typescript/contracts.ts"


def pascal(value: str) -> str:
    return "".join(part[0].upper() + part[1:] for part in re.split(r"[^A-Za-z0-9]+", value) if part)


def load_schemas() -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in SCHEMA_PATHS]


def literal_python(value: object) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    return json.dumps(value, ensure_ascii=False)


def python_literal(values: list[object]) -> str:
    if len(values) == 1:
        return f"Literal[{literal_python(values[0])}]"
    rendered = "\n".join(f"    {literal_python(value)}," for value in values)
    return f"Literal[\n{rendered}\n]"


def python_tuple_constant(name: str, values: tuple[str, ...]) -> str:
    rendered = "\n".join(f"    {json.dumps(value)}," for value in values)
    return f"{name}: Final[tuple[str, ...]] = (\n{rendered}\n)"


def literal_typescript(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return json.dumps(value, ensure_ascii=False)


class TypeGenerator:
    def __init__(self, language: str) -> None:
        self.language = language
        self.objects: dict[str, tuple[dict[str, Any], set[str]]] = {}
        self.aliases: dict[str, str] = {}

    def register_definition(self, name: str, schema: dict[str, Any]) -> None:
        rendered = self.type_for(schema, name)
        if schema.get("type") != "object":
            self.aliases[pascal(name)] = rendered

    def type_for(self, schema: dict[str, Any], name: str) -> str:
        if "$ref" in schema:
            return pascal(str(schema["$ref"]).rsplit("/", 1)[-1])
        if "const" in schema:
            value = schema["const"]
            return (
                python_literal([value]) if self.language == "python" else literal_typescript(value)
            )
        if "enum" in schema:
            values = schema["enum"]
            if self.language == "python":
                return python_literal(values)
            return " | ".join(literal_typescript(value) for value in values)
        choices = schema.get("oneOf") or schema.get("anyOf")
        if choices:
            types = [
                self.type_for(choice, f"{name}Choice{index}")
                for index, choice in enumerate(choices)
            ]
            separator = " | "
            return separator.join(dict.fromkeys(types))
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            return " | ".join(
                dict.fromkeys(self.type_for({"type": item}, name) for item in schema_type)
            )
        if schema_type == "object":
            object_name = pascal(str(schema.get("title") or name))
            self.objects.setdefault(object_name, (schema, set(schema.get("required", []))))
            for prop, child in schema.get("properties", {}).items():
                self.type_for(child, f"{object_name}{pascal(prop)}")
            return object_name
        if schema_type == "array":
            item_type = self.type_for(schema.get("items", {}), f"{name}Item")
            return (
                f"tuple[{item_type}, ...]"
                if self.language == "python"
                else f"ReadonlyArray<{item_type}>"
            )
        mapping = {
            "string": "str" if self.language == "python" else "string",
            "number": "float" if self.language == "python" else "number",
            "integer": "int" if self.language == "python" else "number",
            "boolean": "bool" if self.language == "python" else "boolean",
            "null": "None" if self.language == "python" else "null",
        }
        return mapping.get(schema_type, "object" if self.language == "python" else "unknown")

    def render_objects(self) -> str:
        blocks: list[str] = []
        for name, (schema, required) in self.objects.items():
            properties = schema.get("properties", {})
            if self.language == "python":
                lines = [f"class {name}(TypedDict):"]
                if not properties:
                    lines.append("    pass")
                for prop, child in properties.items():
                    annotation = self.type_for(child, f"{name}{pascal(prop)}")
                    if prop not in required:
                        annotation = f"NotRequired[{annotation}]"
                    lines.append(f"    {prop}: {annotation}")
            else:
                lines = [f"export interface {name} {{"]
                for prop, child in properties.items():
                    optional = "" if prop in required else "?"
                    annotation = self.type_for(child, f"{name}{pascal(prop)}")
                    lines.append(f"  readonly {prop}{optional}: {annotation};")
                lines.append("}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def render_aliases(self) -> str:
        if self.language == "python":
            return "\n".join(f"{name} = {value}" for name, value in self.aliases.items())
        return "\n".join(f"export type {name} = {value};" for name, value in self.aliases.items())


def schema_versions(schemas: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(schema["properties"]["schema_version"]["const"]) for schema in schemas)


def error_codes(schemas: list[dict[str, Any]]) -> tuple[str, ...]:
    error_schema = next(schema for schema in schemas if schema["title"] == "AnprApiErrorV1")
    return tuple(error_schema["$defs"]["ErrorCode"]["enum"])


def render_python(schemas: list[dict[str, Any]]) -> str:
    generator = TypeGenerator("python")
    for schema in schemas:
        for key, definition in schema.get("$defs", {}).items():
            generator.register_definition(key, definition)
        generator.type_for(schema, str(schema["title"]))
    versions = schema_versions(schemas)
    errors = error_codes(schemas)
    typing_imports = ["Final", "Literal"]
    if any(
        set(schema.get("properties", {})) - required
        for schema, required in generator.objects.values()
    ):
        typing_imports.append("NotRequired")
    typing_imports.append("TypedDict")
    unformatted = (
        "# Generated by scripts/generate_api_contracts.py. Do not edit.\n"
        "from __future__ import annotations\n\n"
        f"from typing import {', '.join(typing_imports)}\n\n"
        + python_tuple_constant("SCHEMA_VERSIONS", versions)
        + "\n"
        + python_tuple_constant("ERROR_CODES", errors)
        + "\n\n"
        + generator.render_aliases()
        + "\n\n"
        + generator.render_objects()
        + "\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", "contracts.py", "-"],
        input=unformatted,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def render_typescript(schemas: list[dict[str, Any]]) -> str:
    generator = TypeGenerator("typescript")
    for schema in schemas:
        for key, definition in schema.get("$defs", {}).items():
            generator.register_definition(key, definition)
        generator.type_for(schema, str(schema["title"]))
    versions = ", ".join(json.dumps(value) for value in schema_versions(schemas))
    return (
        "// Generated by scripts/generate_api_contracts.py. Do not edit.\n"
        f"export const SCHEMA_VERSIONS = [{versions}] as const;\n"
        "export const ERROR_CODES = ["
        + ", ".join(json.dumps(value) for value in error_codes(schemas))
        + "] as const;\n\n"
        + generator.render_aliases()
        + "\n\n"
        + generator.render_objects()
        + "\n"
    )


def write_or_check(path: Path, content: str, *, check: bool) -> bool:
    if check:
        return path.is_file() and path.read_text(encoding="utf-8") == content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    schemas = load_schemas()
    outputs = {
        PYTHON_OUTPUT: render_python(schemas),
        TYPESCRIPT_OUTPUT: render_typescript(schemas),
    }
    mismatches = [
        str(path.relative_to(ROOT))
        for path, content in outputs.items()
        if not write_or_check(path, content, check=arguments.check)
    ]
    if mismatches:
        raise SystemExit("generated contract drift: " + ", ".join(mismatches))
    action = "verified" if arguments.check else "generated"
    print(f"{action} {len(outputs)} contract outputs")


if __name__ == "__main__":
    main()
