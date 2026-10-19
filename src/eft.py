"""Equivariance Fine-Tuning (EFT): teach a pixels-only metric depth model to respect zoom,
using unlabeled images only.

For an unlabeled image x and a random zoom box b (factor s, random centre):
    teacher  T = frozen copy of the original model
    student  S = the model being tuned
    L_zoom   = | log S(zoom_b(x)) brought back to b's grid  -  log T(x)[b] |
    L_anchor = | log S(x) - log T(x) |
The teacher on the *unzoomed* image is the target for the zoomed one, so the student has to
find out from the image itself that it was zoomed. L_anchor keeps it close to the original
model on normal images. No depth labels are used anywhere.
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
BASE_H = 392  # input height for training and for the frozen-vs-EFT comparison


def input_hw(h, w, base_h=BASE_H):
    """Keep the aspect ratio, both sides a multiple of the ViT patch (14)."""
    return base_h, int(14 * round(base_h * w / h / 14))


def to_input(img_u8, hw):
    """uint8 HxWx3 -> normalised 1x3xhxw tensor at the model's input size."""
    x = torch.from_numpy(img_u8).permute(2, 0, 1).float()[None] / 255
    x = F.interpolate(x, size=hw, mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    return ((x - MEAN) / STD).to(DEVICE)


def log_depth(model, x):
    d = model(pixel_values=x).predicted_depth
    if d.ndim == 3:
        d = d[:, None]
    return torch.log(d.clamp(min=1e-3))


def random_box(h, w, rng, s_max=3.0):
    s = math.exp(rng.uniform(0, math.log(s_max)))
    cw, ch = int(round(w / s)), int(round(h / s))
    x0 = int(rng.integers(0, w - cw + 1))
    y0 = int(rng.integers(0, h - ch + 1))
    return s, (x0, y0, cw, ch)


class EFT:
    def __init__(self, model, input_hw, lr_head=5e-5, lr_backbone=5e-6, w_anchor=1.0, w_zoom=1.0, s_max=3.0, seed=0):
        self.student = model.to(DEVICE).train()
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.hw = input_hw
        self.w_anchor = w_anchor
        # w_zoom = 0 gives the "anchor only" baseline: same data, steps and flips, no zoom loss
        self.w_zoom = w_zoom
        self.s_max = s_max
        self.rng = np.random.default_rng(seed)
        backbone = [p for n, p in self.student.named_parameters() if "backbone" in n]
        head = [p for n, p in self.student.named_parameters() if "backbone" not in n]
        # fp16 on the GPU roughly halves the step time; the losses are taken in fp32
        self.amp = DEVICE.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.opt = torch.optim.AdamW([{"params": head, "lr": lr_head},
                                      {"params": backbone, "lr": lr_backbone}], weight_decay=1e-4)

    def step(self, imgs):
        """imgs: list of uint8 images (any size). Returns the two loss values."""
        h, w = self.hw
        xs, zs, boxes = [], [], []
        for img in imgs:
            if self.rng.random() < 0.5:
                img = img[:, ::-1]
            x = to_input(np.ascontiguousarray(img), self.hw)
            s, (x0, y0, cw, ch) = random_box(h, w, self.rng, self.s_max)
            z = F.interpolate(x[..., y0:y0 + ch, x0:x0 + cw], size=self.hw, mode="bilinear", align_corners=False)
            xs.append(x)
            zs.append(z)
            boxes.append((x0, y0, cw, ch))
        x, z = torch.cat(xs), torch.cat(zs)

        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=self.amp):
            with torch.no_grad():
                t = log_depth(self.teacher, x)
            s_x = log_depth(self.student, x)
            s_z = log_depth(self.student, z)
        t, s_x, s_z = t.float(), s_x.float(), s_z.float()

        l_zoom = 0
        for k, (x0, y0, cw, ch) in enumerate(boxes):
            back = F.interpolate(s_z[k:k + 1], size=(ch, cw), mode="area")
            l_zoom = l_zoom + (back - t[k:k + 1, :, y0:y0 + ch, x0:x0 + cw]).abs().mean()
        l_zoom = l_zoom / len(boxes)
        l_anchor = (s_x - t).abs().mean()
        loss = self.w_zoom * l_zoom + self.w_anchor * l_anchor

        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(l_zoom), float(l_anchor)


class FixedSizeDepth:
    """Wrap a transformers depth model so it runs at the training input height, with the
    same .predict() interface as models.py. Used to compare the frozen and EFT models
    under exactly the same preprocessing."""

    def __init__(self, model):
        self.model = model.to(DEVICE).eval()

    @torch.no_grad()
    def predict(self, img, K=None):
        h, w = img.shape[:2]
        d = torch.exp(log_depth(self.model, to_input(img, input_hw(h, w))))
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        return {"depth": d[0, 0].cpu().numpy(), "fx": None}
