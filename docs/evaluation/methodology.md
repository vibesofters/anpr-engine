# Evaluation methodology

Detection evaluation separates localization and operating-point behavior and
supports precision, recall, F1, IoU, miss, false-positive, and top-result
analysis. Recognition evaluation separates raw model output from structurally
decoded output and reports exact normalized result accuracy, character error or
accuracy, length accuracy, type agreement, validity, and family-specific
diagnostics.

Public results are percentage-only. Exact sample counts, denominators, split
membership, dataset identities, record-level errors, and private manifests are
not public. The retained family percentages are development-validation evidence,
not an independent holdout, field-performance, production, or reliability
claim.

The public training CLI emits only exact, character, and length percentages for
the supplied validation data. It does not access a test set or independent
holdout. Users must prevent source, identity, vehicle, capture-session, or
near-duplicate leakage when preparing their own data.

Structural validity and confidence remain diagnostic signals. Neither proves
that the transcription is correct or that a vehicle or person has been
identified. A prediction becomes resolved only through explicit human
confirmation or correction.
