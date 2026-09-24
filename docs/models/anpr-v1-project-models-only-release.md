# ANPR Engine project-models-only candidate

Status: curated public source model card. The trained weights remain private;
public source availability is not a claim of independent field accuracy or
unattended production readiness.

## Active architecture

`INPUT → PROJECT DETECTION → BLUE_BAND_GLYPH_QUAD_V1 → PROJECT RECOGNITION → STRUCTURAL VALIDATION → HUMAN REVIEW`

| Component | Immutable private identity |
|---|---|
| Detection | `0d5e9b52fae5638e0ef1badafdeb53fe0771d056e43c4953d13cceed520e71ea` |
| Recognition | `25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7` |
| Crop refinement | `BLUE_BAND_GLYPH_QUAD_V1` |

The software uses only project-developed Detection and Recognition implementations. Model weights remain private and are not licensed or distributed with the source candidate.

## Detection description

The detector is a project-owned, project-trained PyTorch implementation informed by established YOLO-family research. It is single-stage, one-class, and anchor-free, with stride 8/16/32 features, FPN/PAN-style fusion, decoupled heads, focal-style objectness loss, GIoU regression, dynamic assignment, and non-maximum suppression. These established concepts are not claimed as entirely novel.

Primary references: YOLO [Redmon et al., arXiv:1506.02640], YOLOX/SimOTA [Ge et al., arXiv:2107.08430], FPN [Lin et al., arXiv:1612.03144], PANet [Liu et al., arXiv:1803.01534], focal loss [Lin et al., arXiv:1708.02002], and GIoU [Rezatofighi et al., arXiv:1902.09630].

## Recognition development history

1. CNN + BiLSTM + CTC — recurrent/CTC control.
2. CNN + BiLSTM + Fixed Slots — initially expected to be strongest because explicit slots avoid CTC collapse.
3. CNN Tokenizer + Transformer + CTC — improved sequence representation but retained CTC alignment, length, repeat, and type weaknesses.
4. CNN Tokenizer + Transformer + Fixed Slots — selected architecture.

Selected-model development-validation metrics: raw exact 93.37%, structurally decoded exact 93.37%, character accuracy 98.96%, character error rate 1.04%, length accuracy 98.98%, and type accuracy 98.47%. These are not independent field-holdout or production-accuracy claims.

Primary concept references: Schuster and Paliwal (1997), Graves et al. (ICML 2006), and Vaswani et al. (arXiv:1706.03762).

Predictions remain unresolved until a human confirms the model prediction or supplies a correction. Confidence and structural validity are informational only. See `model-registry/public/` for curated model identities and limitations.
