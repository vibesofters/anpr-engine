# Public capability matrix

This matrix records the Phase 1B public candidate as inspected before the
public-scope correction. “Private assets” means that architecture-level and
synthetic execution is public, while meaningful project inference still needs
owner-supplied weights and training or evaluation needs a user-supplied lawful
dataset.

| Capability | Present before correction | Public entry point before correction | Executable | Public test | Private assets needed | Safe template | Phase 1B action | Corrective disposition |
|---|---|---|---|---|---|---|---|---|
| Detection architecture | Yes | `anpr_engine.detection.model` | Yes | Yes | No for architecture smoke; weights for meaningful inference | Foundation/config examples | Retained | Retain and document cited influences |
| Detection training | Yes | `anpr_engine.detection.training` | Yes | Yes | User dataset for meaningful training | `configs/public/detection-training-template.yaml` | Retained | Retain and document |
| Detection evaluation | Yes | `anpr_engine.detection.metrics` and training evaluation helpers | Yes | Yes | User dataset and checkpoint | Detection template | Retained | Retain and document percentage-only public reporting |
| CNN + BiLSTM + CTC | Yes | `recognition.enhanced_model` | Forward only as a coherent public workflow | Forward smoke only | No for synthetic architecture tests | Foundation YAML | Training orchestration removed | Add family-neutral public training path |
| CNN + BiLSTM + Fixed Slots | Yes | `recognition.fixed_slot` | Forward and isolated loss step | Yes | No for synthetic tests | Foundation YAML | Training orchestration removed | Integrate existing target/loss logic into public training path |
| CNN Tokenizer + Transformer + CTC | Yes | `recognition.transformer_model` | Forward only as a coherent public workflow | Forward smoke only | No for synthetic tests | Foundation YAML | Training orchestration removed | Add family-neutral public training path |
| CNN Tokenizer + Transformer + Fixed Slots | Yes | `recognition.transformer_slot_model` | Forward only as a coherent public workflow | Forward smoke only | Private selected weight for project inference | Foundation YAML and recognition template | Training orchestration removed | Add training path and selected-architecture reproduction contract |
| Recognition training | No coherent public path | Architecture-specific primitives only | No | Partial | User dataset for meaningful training | Incomplete placeholder template | Generic and private governance logic were removed together | Replace with a small sanitized dataset/config/train/checkpoint workflow |
| Recognition evaluation | Yes | `recognition.evaluation` and `recognition.metrics` | Yes | Yes | User validation data for meaningful results | No dedicated workflow document | Retained | Retain; document percentage-only public reporting |
| Checkpoint loading and verification | Yes | `recognition.checkpointing` | Yes | Yes for one family | Checkpoint for actual use | Foundation identities | Retained | Extend tests across all four families and explicit resume |
| Synthetic fixture generation | Yes | `recognition.synthetic` | Yes | Indirect/limited | No | `configs/public/synthetic-fixture-template.yaml` | Retained | Retain as small, non-identifying fixtures only |
| End-to-end inference | Yes | `integration.anpr_v1.AnprV1Pipeline` | Yes with private bundle | Yes | Approved private Detection and Recognition weights | Private-runtime guide | Retained and sanitized | Retain; keep weights outside Git |
| Browser review | Yes | `scripts/launch_anpr_v1_browser.py` | Yes with private bundle | Yes | Approved private weights for inference | Private-runtime guide | Retained and sanitized | Retain as local review tooling, not a public service |
| Structural validation | Yes | Recognition profile/decoder and integration release logic | Yes | Yes | No for synthetic unit tests | Recognition foundations | Retained | Retain; state that validity is not identity proof |
| Human confirmation/correction | Yes | `integration.review_store` and browser UI | Yes locally | Yes | Inference result for practical use | Browser workflow | Retained and sanitized | Retain locally; predictions remain unresolved until review |

The correction deliberately does not restore private experiment identifiers,
detached-process controls, owner approval gates, exact dataset evidence,
record-level review material, private paths, or external OCR material.

## Correction result

The corrective disposition is implemented. All four families now use
`anpr_engine.recognition.public_training` and the explicit
`scripts/train_recognition.py` entry point for user-supplied data, target and
shape validation, architecture-appropriate loss, percentage-only validation,
strict checkpoint save/reload, and user-controlled resume. Synthetic public
tests cover a finite training step and checkpoint round trip for every family.
The selected Transformer + Fixed Slots output contract remains independently
checked. Capabilities that require private weights or lawful user data remain
clearly conditional rather than being represented as self-contained public
inference or evidence.
