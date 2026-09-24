# Public Recognition training workflow

The sanitized training path supports the four disclosed project Recognition
families without private datasets, weights, experiment IDs, paths, or governance
records. It provides model construction, target validation, CTC or fixed-slot
loss, optimizer and scheduler construction, percentage-only validation metrics,
checkpoint identity, strict reload, and explicit resume.

## Dataset interface

Supply a directory outside the repository with this structure:

```text
authorized-dataset/
  train/
    images/
      fixture.png
    labels/
      fixture.txt
  validation/
    images/
    labels/
```

JPEG and PNG image stems must have matching UTF-8 `.txt` labels. Filenames are
never interpreted as labels. Labels must already be normalized and accepted by
the TR V1 structural profile. The loader resizes images to the documented
`3 × 32 × 160` input contract. Dataset suitability, authorization, split
integrity, and preprocessing quality remain the user's responsibility.

The public workflow intentionally has no test or independent-holdout access.
Do not point it at private project data without the applicable legal, privacy,
and governance approvals.

## Configuration and execution

Start from `configs/public/recognition-training-template.yaml`. Change `family`
and `foundation_config` together to select one of the four foundation files.
The public parser rejects absolute paths, traversal, known private path tokens,
unknown fields, and embedded private dataset identity fields.

```text
uv run python scripts/train_recognition.py \
  --config configs/public/recognition-training-template.yaml \
  --dataset-root /path/to/authorized-dataset \
  --output /path/to/local-checkpoint.pt
```

Resume is never automatic. An existing checkpoint is loaded only when the user
passes `--resume`; the loader verifies model family, architecture metadata,
parameter count, configuration identity, and optionally the file SHA-256.

The CLI reports model/configuration/checkpoint identity and validation
percentages. Checkpoints and user data should remain outside Git. This workflow
is reproducibility scaffolding, not a claim that training on arbitrary data will
reproduce the project's private evidence or selected private weight.
