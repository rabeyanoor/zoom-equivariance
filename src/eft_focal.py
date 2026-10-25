"""EFT-F: equivariance fine-tuning of Depth Pro's focal head with a focal-consistency loss.

Depth Pro outputs depth = f_px * C(I) / W, where the camera-free C comes from the depth network and f_px from
a separate field-of-view head that reads the image and *detached* encoder features (HF modeling_depth_pro).
With the true focal, C already follows a zoom (alpha_K ~ 0, Sec. 5), so the zoom error is the focal head's.
A digital zoom by s multiplies the true focal in pixels by s, which gives free supervision for that head:

    L = | log t(Z_s I) - (log t0(I) - log s) |  +  w_blur | log t(B I) - log t0(I) |  +  w_anchor | log t(I) - log t0(I) |

with t = tan(fov / 2) (log f_px = log(W / 2) - log t), t0 the frozen original head, Z_s a zoom and B a
detail loss with the same field of view. Only the head's own layers after the frozen fov ViT are tuned
(TRAINABLE); the depth network is untouched. Everything the tuned layers read is frozen, so it is computed
once per training view (encode) and the head is fitted on that cache (fov_from_cache).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

TRAINABLE = ("fov_encoder.neck.", "conv.", "head.")  # parameter prefixes inside model.fov_model


def log_tan_half(fov_deg):
    return torch.log(torch.tan(0.5 * torch.deg2rad(fov_deg.float())))


@torch.no_grad()
def encode(model, pixel_values):
    """The frozen inputs of the tuned layers: the global encoder feature and the fov ViT tokens."""
    feats = model.depth_pro(pixel_values=pixel_values, return_dict=True).features
    fm = model.fov_model
    size = fm.config.fov_model_config.image_size
    small = F.interpolate(pixel_values, size=(size, size), mode="bilinear", align_corners=False)
    tokens = fm.fov_encoder.model(pixel_values=small)[0]
    return feats[0], tokens


def fov_from_cache(fov_model, global_feat, tokens, height, width):
    """DepthProFovModel.forward after its frozen ViT, from cached inputs (same result as the full forward)."""
    from transformers.models.depth_pro.modeling_depth_pro import reconstruct_feature_maps

    enc = fov_model.fov_encoder
    hidden = enc.neck(tokens)
    exponent = int(math.log2(width / enc.out_size))
    fov_feat = reconstruct_feature_maps(hidden, batch_size=tokens.shape[0], padding=0,
                                        output_size=(height // 2 ** exponent, width // 2 ** exponent))
    g = fov_model.activation(fov_model.conv(global_feat))
    return fov_model.head(fov_feat + g).flatten()


def trainable_parameters(fov_model):
    return {n: p for n, p in fov_model.named_parameters() if n.startswith(TRAINABLE)}


def random_view(img, rng, s_max=3.0):
    """One training view of a uint8 image: ("plain", 1), ("blur", 1) or ("zoom", s) with a random box."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    h, w = img.shape[:2]
    kind = rng.choice(["plain", "blur", "zoom", "zoom"])
    s = math.exp(rng.uniform(0, math.log(s_max)))
    if kind == "zoom":
        cw, ch = round(w / s), round(h / s)
        x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        return np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw]), kind, w / cw
    if kind == "blur":
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
        small = F.interpolate(t, size=(max(14, int(h / s)), max(14, int(w / s))), mode="area")
        back = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
        return back[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy(), kind, 1.0
    return np.ascontiguousarray(img), kind, 1.0


def loss_terms(pred, target, kind, w_blur=1.0, w_anchor=1.0):
    """pred, target: log tan(fov/2) per view (target already includes -log s for zooms); kind: 0 plain, 1 blur,
    2 zoom. Returns the weighted loss and the mean zoom error."""
    err = (pred - target).abs()
    w = torch.where(kind == 2, torch.ones_like(err), torch.where(kind == 1, w_blur, w_anchor) * torch.ones_like(err))
    zoom = err[kind == 2].mean() if (kind == 2).any() else err.new_zeros(())
    return (w * err).mean(), zoom
