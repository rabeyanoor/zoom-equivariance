"""One small interface for every depth model we test.

    m = load("moge2-s")
    out = m.predict(img_uint8, K=None)   # out["depth"]: HxW metres, out["fx"]: focal in px or None

Models fall into three groups, and the paper's story depends on which group a model is in:
  - "pixels only": metric depth straight from pixels, no camera output (Depth Anything V2
    metric, ZoeDepth). They cannot tell a zoomed photo from a closer object except through
    perspective cues.
  - "predicts camera": they output a focal length / rays next to depth (Depth Pro, MoGe-2,
    UniDepthV2). Zoom equivariance says the focal must scale by s.
  - "takes camera": they need K as input (Metric3D v2). Given the right K they should be
    equivariant by design; given a stale K (e.g. EXIF from the unzoomed photo) they are not.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
THIRD = ROOT / "third_party"
torch.set_num_threads(int(os.environ.get("ZE_THREADS", "6")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_hw(depth, hw):
    t = torch.as_tensor(depth).float()
    while t.ndim < 4:
        t = t[None]
    if tuple(t.shape[-2:]) != tuple(hw):
        t = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


class HFDepth:
    """transformers depth models (DA-V2 metric, ZoeDepth, Depth Pro)."""

    def __init__(self, repo, kind):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.kind = kind
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo).eval().to(DEVICE)

    @torch.no_grad()
    def predict(self, img, K=None):
        h, w = img.shape[:2]
        inputs = self.proc(images=img, return_tensors="pt").to(DEVICE)
        out = self.model(**inputs)
        if self.kind == "dav2":
            return {"depth": _to_hw(out.predicted_depth.cpu(), (h, w)), "fx": None}
        post = self.proc.post_process_depth_estimation(
            out, target_sizes=[(h, w)], **({"source_sizes": [(h, w)]} if self.kind == "zoe" else {}))[0]
        fx = None
        if self.kind == "depthpro":
            fx = float(post["focal_length"])
        return {"depth": _to_hw(post["predicted_depth"].cpu(), (h, w)), "fx": fx}


class MoGe2:
    def __init__(self, repo):
        from moge.model.v2 import MoGeModel

        self.model = MoGeModel.from_pretrained(repo).eval().to(DEVICE)

    @torch.no_grad()
    def predict(self, img, K=None):
        h, w = img.shape[:2]
        x = torch.from_numpy(img).permute(2, 0, 1).float().to(DEVICE) / 255
        # fp16 is MoGe's default and is what we use on the GPU; the CPU has no fp16 kernels
        out = {k: v.cpu() for k, v in self.model.infer(x, use_fp16=DEVICE.type == "cuda").items()}
        depth = out["depth"].numpy().copy()
        depth[~out["mask"].numpy()] = np.nan
        # MoGe returns intrinsics normalised by image width / height
        fx = float(out["intrinsics"][0, 0]) * w
        # principal point in pixels (pixel centres at integer coordinates)
        cx = float(out["intrinsics"][0, 2]) * w - 0.5
        return {"depth": depth, "fx": fx, "cx": cx}


class UniDepth2:
    def __init__(self, repo):
        from unidepth.models import UniDepthV2

        self.model = UniDepthV2.from_pretrained(repo).eval().to(DEVICE)

    @torch.no_grad()
    def predict(self, img, K=None):
        x = torch.from_numpy(img).permute(2, 0, 1)
        out = self.model.infer(x)
        fx = float(out["intrinsics"][0, 0, 0])
        cx = float(out["intrinsics"][0, 0, 2])
        return {"depth": out["depth"][0, 0].cpu().numpy(), "fx": fx, "cx": cx}


def _mmcv_shim():
    """Metric3D imports a few helpers from the old mmcv 1.x, which does not build on new
    torch. mmengine has the same Config; the other two are only used for logging."""
    import types

    try:
        import mmcv.utils  # noqa: F401
        return
    except ImportError:
        pass
    from mmengine import Config, DictAction

    utils = types.ModuleType("mmcv.utils")
    utils.Config, utils.DictAction = Config, DictAction
    utils.collect_env = lambda: {}
    utils.get_git_hash = lambda *a, **k: "unknown"
    mmcv = types.ModuleType("mmcv")
    mmcv.utils = utils
    sys.modules["mmcv"], sys.modules["mmcv.utils"] = mmcv, utils


class Metric3D:
    """Metric3D v2 ViT-S. Needs fx; works in a canonical camera with f = 1000 px."""

    INPUT = (616, 1064)
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self):
        _mmcv_shim()
        sys.path.insert(0, str(THIRD / "Metric3D"))
        self.model = torch.hub.load(str(THIRD / "Metric3D"), "metric3d_vit_small",
                                    pretrain=True, source="local", trust_repo=True).eval().to(DEVICE)

    @torch.no_grad()
    def predict(self, img, K):
        h, w = img.shape[:2]
        scale = min(self.INPUT[0] / h, self.INPUT[1] / w)
        nh, nw = int(h * scale), int(w * scale)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float()
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)[0]
        x = (x - torch.from_numpy(self.MEAN)[:, None, None]) / torch.from_numpy(self.STD)[:, None, None]
        ph, pw = self.INPUT[0] - nh, self.INPUT[1] - nw
        top, left = ph // 2, pw // 2
        # pad with zeros after normalisation == padding with the mean colour before it
        x = F.pad(x, (left, pw - left, top, ph - top))[None].to(DEVICE)
        pred, _, _ = self.model.inference({"input": x})
        pred = pred[0, 0, top:top + nh, left:left + nw].float().cpu()
        depth = _to_hw(pred, (h, w)) * (K[0, 0] * scale / 1000.0)
        return {"depth": np.clip(depth, 0, 300), "fx": None}


def _dav2_fixed(domain, tag=None):
    """DA-V2-S at the EFT input size; with a tag, loads fine-tuned weights from
    $EFT_DIR/dav2-s-<tag>-<domain>.pt (tag "eft" or "anchor")."""
    from transformers import AutoModelForDepthEstimation

    import eft as eft_lib

    m = AutoModelForDepthEstimation.from_pretrained(f"depth-anything/Depth-Anything-V2-Metric-{domain}-Small-hf")
    if tag:
        ckpt = Path(os.environ.get("EFT_DIR", ROOT / "outputs" / "eft")) / f"dav2-s-{tag}-{domain.lower()}.pt"
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return eft_lib.FixedSizeDepth(m)


# name -> (group, loader). "dav2" outdoor/indoor weights are picked per dataset.
REGISTRY = {
    "dav2-s": ("pixels only", lambda dom: HFDepth(f"depth-anything/Depth-Anything-V2-Metric-{dom}-Small-hf", "dav2")),
    "dav2-b": ("pixels only", lambda dom: HFDepth(f"depth-anything/Depth-Anything-V2-Metric-{dom}-Base-hf", "dav2")),
    "zoedepth": ("pixels only", lambda dom: HFDepth("Intel/zoedepth-nyu-kitti", "zoe")),
    "depthpro": ("predicts camera", lambda dom: HFDepth("apple/DepthPro-hf", "depthpro")),
    "moge2-s": ("predicts camera", lambda dom: MoGe2("Ruicheng/moge-2-vits-normal")),
    "unidepth2-s": ("predicts camera", lambda dom: UniDepth2("lpiccinelli/unidepth-v2-vits14")),
    "metric3d-s": ("takes camera", lambda dom: Metric3D()),
    # frozen vs fine-tuned, same preprocessing (section 6 of the paper)
    "dav2-s-fixed": ("pixels only", lambda dom: _dav2_fixed(dom)),
    "dav2-s-eft": ("pixels only (EFT)", lambda dom: _dav2_fixed(dom, "eft")),
    "dav2-s-anchor": ("pixels only (anchor-only FT)", lambda dom: _dav2_fixed(dom, "anchor")),
    "dav2-s-eftb": ("pixels only (EFT + blur invariance)", lambda dom: _dav2_fixed(dom, "eftb")),
}


def load(name, domain="Outdoor"):
    """domain only matters for Depth Anything, which ships separate indoor/outdoor heads."""
    return REGISTRY[name][1](domain)
