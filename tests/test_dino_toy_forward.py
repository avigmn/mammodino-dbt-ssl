"""Shape and finite-loss checks for toy DINO (no CIFAR download)."""

from __future__ import annotations

import torch

from mammodino_ssl.models.dino_ssl import create_dino_ssl
from mammodino_ssl.train.dino_loss import DINOLogitCenter, dino_cross_entropy


def test_dino_forward_output_shapes() -> None:
    m = create_dino_ssl(image_size=224, num_prototypes=512)
    x = torch.randn(2, 3, 224, 224)
    ts = m.forward_student(x)
    assert ts.shape == (2, 512)
    with torch.no_grad():
        tt = m.forward_teacher(x)
    assert tt.shape == (2, 512)


def test_dino_loss_finite() -> None:
    m = create_dino_ssl(image_size=224, num_prototypes=1024)
    center = DINOLogitCenter(1024, center_momentum=0.9)
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        t_logits = m.forward_teacher(x)
    s_logits = m.forward_student(x)
    c = center.center
    loss = dino_cross_entropy(s_logits, t_logits, c, student_temp=0.1, teacher_temp=0.04)
    assert torch.isfinite(loss).all()
    center.update_from_teacher_logits(t_logits.detach())
    loss2 = dino_cross_entropy(s_logits, t_logits, center.center, student_temp=0.1, teacher_temp=0.04)
    assert torch.isfinite(loss2).all()
