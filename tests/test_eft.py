"""EFT-S on a toy base model: starts as the frozen model, trains only its head, keeps the shape."""
from types import SimpleNamespace

import numpy as np
import torch

import eft


class ToyDepth(torch.nn.Module):
    """Stands in for DA-V2: depth from a conv, tokens from patch averages."""

    def __init__(self, dim=8):
        super().__init__()
        self.backbone = torch.nn.Conv2d(3, dim, 14, stride=14)
        self.head = torch.nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, pixel_values, output_hidden_states=False):
        tokens = self.backbone(pixel_values).flatten(2).transpose(1, 2)
        tokens = torch.cat([tokens.mean(1, keepdim=True), tokens], 1)
        depth = torch.nn.functional.softplus(self.head(pixel_values))[:, 0] + 1
        return SimpleNamespace(predicted_depth=depth, hidden_states=(tokens,))


def test_scale_adapter_starts_frozen_and_trains_only_the_head():
    torch.manual_seed(0)
    base = ToyDepth()
    model = eft.ScaleAdapter(base, dim=8)
    x = torch.randn(2, 3, 28, 42)
    assert torch.equal(model(x).predicted_depth, base(x).predicted_depth)

    frozen = {k: v.clone() for k, v in base.state_dict().items()}
    trainer = eft.EFT(model, (28, 42), lr_head=1e-2, w_blur=1.0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        trainer.step([rng.integers(0, 255, (56, 84, 3), dtype=np.uint8) for _ in range(2)])
    assert all(torch.equal(v, frozen[k]) for k, v in trainer.student.base.state_dict().items())
    assert trainer.student.head[-1].weight.abs().sum() > 0

    # one factor per image: the log-depth difference to the frozen model is constant over pixels
    x = x.to(eft.DEVICE)
    diff = torch.log(trainer.student(x).predicted_depth) - torch.log(trainer.student.base(x).predicted_depth)
    assert torch.allclose(diff, diff.flatten(1)[:, :1, None], atol=1e-5)


def test_scale_adapter_fused_step_matches_the_generic_path(monkeypatch):
    """EFT-S runs the frozen base once for teacher and student; the losses are the same as with
    separate teacher and student passes."""
    torch.manual_seed(0)
    base = ToyDepth()
    imgs = [np.random.default_rng(1).integers(0, 255, (56, 84, 3), dtype=np.uint8) for _ in range(2)]
    losses = []
    for fused in (True, False):
        torch.manual_seed(1)
        model = eft.ScaleAdapter(base, dim=8)
        torch.nn.init.normal_(model.head[-1].weight, std=0.1)  # a non-zero scale, so the paths can differ
        trainer = eft.EFT(model, (28, 42), lr_head=0.0, w_blur=1.0)
        if not fused:
            monkeypatch.setattr(eft, "ScaleAdapter", type("NotAnAdapter", (), {}))
            trainer.teacher = base  # what the teacher is in real training: the frozen model (zero head)
        losses.append(trainer.step(imgs))
        monkeypatch.undo()
    assert np.allclose(losses[0], losses[1], atol=1e-5)
