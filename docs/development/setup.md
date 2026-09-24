# Development setup

Use Python 3.11 or 3.12 and the pinned `uv` version. From the repository root:

```text
uv sync --locked --all-groups
uv run pytest
uv run ruff format --check src scripts tests
uv run ruff check src scripts tests
uv run mypy src
```

Architecture and training tests create only synthetic tensors or small
non-identifying fixtures. They do not require private datasets or weights.
Private-runtime smoke tests are opt-in and require an owner-controlled external
bundle as described in the [runtime guide](../operations/private-model-runtime.md).

Do not commit local datasets, checkpoints, generated artifacts, model-bundle
paths, secrets, or credentials. Public examples must remain synthetic and
non-identifying. Use the Recognition training template and explicit CLI rather
than restoring internal experiment launchers or private governance controls.
