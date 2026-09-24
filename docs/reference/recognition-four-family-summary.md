# Project Recognition architecture families

The project evaluated four project-owned Recognition architecture families:

| Family | Development-validation evidence | Decision |
|---|---|---|
| CNN + BiLSTM + CTC | raw exact 13.45%; decoded exact 14.83%; character accuracy 68.48%; CER 31.52%; type 62.07% | control; alignment and exact-sequence limitations prevented selection |
| CNN + BiLSTM + Fixed Slots | raw exact 18.62%; decoded exact 17.93%; character accuracy 70.68%; CER 29.32%; length 90.69%; type 87.93% | exact-sequence evidence remained insufficient |
| CNN Tokenizer + Transformer + CTC | raw exact 34.83%; decoded exact 41.38%; character accuracy 78.76%; CER 21.24%; length 49.31%; type 26.90% | CTC length, repeat, alignment, and type limitations prevented selection |
| CNN Tokenizer + Transformer + Fixed Slots | raw/decoded exact 93.37%; character accuracy 98.96%; CER 1.04%; length 98.98%; type 98.47% | selected architecture |

These are percentage-only development-validation results. They are not
independent field-holdout or production-accuracy claims. Exact datasets,
population sizes, split membership, run evidence, and weights remain private.

Established concepts are attributed to Schuster and Paliwal, “Bidirectional
Recurrent Neural Networks” (1997); Graves et al., “Connectionist Temporal
Classification” (ICML 2006); and Vaswani et al., “Attention Is All You Need”
(arXiv:1706.03762).
