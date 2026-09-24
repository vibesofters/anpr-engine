# Detection architecture and cited influences

The detector is a project-owned, project-trained PyTorch implementation. It is
a one-class, anchor-free, multi-scale detector with stride 8/16/32 features,
FPN/PAN-style fusion, decoupled prediction heads, focal-style objectness loss,
GIoU box regression, dynamic positive assignment, and non-maximum suppression.

The implementation is informed by established work and is not presented as
independent of that work:

- Redmon et al., [You Only Look Once](https://arxiv.org/abs/1506.02640).
- Ge et al., [YOLOX](https://arxiv.org/abs/2107.08430), including the SimOTA
  assignment direction.
- Lin et al., [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144).
- Liu et al., [Path Aggregation Network](https://arxiv.org/abs/1803.01534).
- Lin et al., [Focal Loss](https://arxiv.org/abs/1708.02002).
- Rezatofighi et al., [Generalized IoU](https://arxiv.org/abs/1902.09630).

Public source includes the architecture, loss, training, evaluation, checkpoint,
and inference contracts. Meaningful training and evaluation require a lawful
user-supplied dataset; meaningful project inference requires the approved
private detector weight. Public documentation reports only approved percentages,
not exact private populations or record-level evidence.
