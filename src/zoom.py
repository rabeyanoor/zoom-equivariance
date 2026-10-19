"""Digital zoom as an exact camera transform.

Cropping a box (x0, y0, w, h) out of an image and resizing it back to (W, H) is the same
as taking the picture with a different pinhole camera:

    fx' = fx * W / w        cx' = (cx - x0 + 1/2) * W / w - 1/2
    fy' = fy * H / h        cy' = (cy - y0 + 1/2) * H / h - 1/2

(the half pixels come from pixel centres: resizing uses align_corners=False)

and every pixel that survives the crop keeps its metric depth. So a model that really
understands the camera should (a) scale its focal estimate by s = W / w and (b) predict
the same depth for the same 3D point, no matter how much we zoom.
"""
import numpy as np
import torch
import torch.nn.functional as F


def zoom_box(h, w, s, cx_frac=0.5, cy_frac=0.5):
    """Crop box for zoom factor s. (cx_frac, cy_frac) is the crop centre as a fraction of
    the image, clipped so the box stays inside. Returns integer (x0, y0, cw, ch)."""
    cw, ch = int(round(w / s)), int(round(h / s))
    x0 = int(round(cx_frac * w - cw / 2))
    y0 = int(round(cy_frac * h - ch / 2))
    x0 = min(max(x0, 0), w - cw)
    y0 = min(max(y0, 0), h - ch)
    return x0, y0, cw, ch


def zoom_K(K, box, out_hw):
    x0, y0, cw, ch = box
    H, W = out_hw
    sx, sy = W / cw, H / ch
    K = np.array(K, dtype=np.float64).copy()
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - x0 + 0.5) * sx - 0.5
    K[1, 2] = (K[1, 2] - y0 + 0.5) * sy - 0.5
    return K


def _resize(x, hw, mode):
    t = torch.from_numpy(np.ascontiguousarray(x)).float()
    t = t.permute(2, 0, 1)[None] if t.ndim == 3 else t[None, None]
    kw = {} if mode in ("nearest", "area") else {"align_corners": False}
    if mode == "bicubic":
        kw["antialias"] = True
    out = F.interpolate(t, size=hw, mode=mode, **kw)[0]
    return out.permute(1, 2, 0).numpy() if x.ndim == 3 else out[0].numpy()


def crop(x, box):
    x0, y0, cw, ch = box
    return x[y0:y0 + ch, x0:x0 + cw]


def zoom_image(img, box, mode="bicubic"):
    """uint8 HxWx3 -> zoomed uint8 image of the same size (bicubic, like a phone's digital zoom)."""
    out = _resize(crop(img, box).astype(np.float32), img.shape[:2], mode)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def blur_control(img, s):
    """Same loss of detail as a zoom by s (down to 1/s, back up), but the field of view does
    not change. If depth moves under this too, the cause is interpolation, not the camera."""
    h, w = img.shape[:2]
    small = _resize(img.astype(np.float32), (int(round(h / s)), int(round(w / s))), "area")
    out = _resize(small, (h, w), "bicubic")
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def downsample(img, f):
    """Area-downsample by an integer factor f (a lower-resolution camera with the same field of view)."""
    h, w = img.shape[:2]
    out = _resize(img.astype(np.float32), (h // f, w // f), "area")
    return np.clip(out + 0.5, 0, 255).astype(np.uint8) if img.dtype == np.uint8 else out


def scale_K(K, in_hw, out_hw):
    """Intrinsics after resizing the whole image from in_hw to out_hw."""
    return zoom_K(K, (0, 0, in_hw[1], in_hw[0]), out_hw)


def jpeg(img, quality=75):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def resize_image(img, hw):
    out = _resize(img.astype(np.float32), hw, "bicubic")
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def resize_depth(depth, hw):
    return _resize(depth, hw, "bilinear")


def unzoom_depth(depth_zoomed, box):
    """Bring a depth map predicted on the zoomed image back onto the crop's own pixel grid,
    so it can be compared with the original image's depth or GT inside the box."""
    _, _, cw, ch = box
    return _resize(depth_zoomed, (ch, cw), "area")
