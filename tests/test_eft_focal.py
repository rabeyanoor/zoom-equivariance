"""EFT-F: the cached focal-head path reproduces Depth Pro's own field of view, and only the head is tuned."""
import numpy as np
import pytest
import torch

import eft_focal as ef

transformers = pytest.importorskip("transformers")


def tiny_depthpro():
    from transformers import DepthProConfig, DepthProForDepthEstimation, Dinov2Config

    def vit():
        return Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                            image_size=384, patch_size=16)

    torch.manual_seed(0)
    cfg = DepthProConfig(image_model_config=vit(), patch_model_config=vit(), fov_model_config=vit(),
                         fusion_hidden_size=32, intermediate_hook_ids=(1, 0), intermediate_feature_dims=(16, 16),
                         scaled_images_feature_dims=(32, 32, 16), use_fov_model=True)
    return DepthProForDepthEstimation(cfg).eval()


def test_cached_path_matches_model():
    m = tiny_depthpro()
    x = torch.randn(2, 3, 1536, 1536)
    with torch.no_grad():
        full = m(pixel_values=x).field_of_view
        mine = ef.fov_from_cache(m.fov_model, *ef.encode(m, x), 1536, 1536)
    assert torch.allclose(full, mine, atol=1e-5)
    names = set(ef.trainable_parameters(m.fov_model))
    assert names and all(not n.startswith("fov_encoder.model.") for n in names)
    assert any(n.startswith("head.") for n in names) and any(n.startswith("conv.") for n in names)


def test_views_and_loss():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (60, 90, 3), dtype=np.uint8)
    seen = set()
    for _ in range(40):
        v, kind, s = ef.random_view(img, rng)
        seen.add(kind)
        assert v.dtype == np.uint8 and (s == 1.0) == (kind != "zoom")
        if kind == "zoom":
            assert abs(s - 90 / v.shape[1]) < 1e-9 and 1 <= s <= 3.0 + 1e-6
        else:
            assert v.shape == img.shape
    assert seen == {"plain", "blur", "zoom"}
    pred, tgt = torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 2.0, 3.0])
    loss, zoom = ef.loss_terms(pred, tgt, torch.tensor([0, 1, 2]), w_blur=0.5, w_anchor=2.0)
    assert torch.isclose(loss, torch.tensor((2.0 + 1.0 + 3.0) / 3)) and zoom == 3.0
