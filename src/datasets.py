"""Loaders for the evaluation sets. Each sample is a dict:

    img   : HxWx3 uint8
    depth : HxW float32 metres, 0 where there is no ground truth
    K     : 3x3 intrinsics of img
    name  : unique id

Only official files are used (see scripts/download.sh); nothing is resampled at load time.
"""
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def _K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------- KITTI (outdoor, LiDAR)
def kitti_list(root=RAW / "kitti" / "depth_selection" / "val_selection_cropped"):
    return sorted((root / "image").glob("*.png"))


def kitti_load(img_path):
    root = img_path.parents[1]
    gt_path = root / "groundtruth_depth" / img_path.name.replace("_image_", "_groundtruth_depth_")
    k_path = root / "intrinsics" / img_path.name.replace(".png", ".txt")
    img = np.array(Image.open(img_path).convert("RGB"))
    depth = np.array(Image.open(gt_path), dtype=np.float32) / 256.0
    K = np.loadtxt(k_path).reshape(3, 3)
    return {"img": img, "depth": depth, "K": K, "name": "kitti/" + img_path.stem}


# ---------------------------------------------------------------- NYUv2 (indoor, Kinect)
NYU_K = _K(518.8579, 519.4696, 325.5824, 253.7362)


class NYU:
    """The 654 official test images of the labelled NYUv2 set (h5 / MATLAB v7.3)."""

    def __init__(self, root=RAW / "nyu"):
        self.f = h5py.File(root / "nyu_depth_v2_labeled.mat", "r")
        self.test = loadmat(root / "splits.mat")["testNdxs"].ravel() - 1

    def __len__(self):
        return len(self.test)

    def __getitem__(self, i):
        j = int(self.test[i])
        img = np.array(self.f["images"][j]).transpose(2, 1, 0)   # stored as (3, W, H)
        depth = np.array(self.f["depths"][j], dtype=np.float32).T
        # usual practice: only score the area the Kinect actually covers (Eigen crop)
        valid = np.zeros_like(depth, bool)
        valid[45:471, 41:601] = True
        depth[~valid] = 0
        return {"img": np.ascontiguousarray(img), "depth": depth, "K": NYU_K.copy(), "name": f"nyu/{j:04d}"}


# ---------------------------------------------------------------- DIODE (laser, indoor + outdoor)
DIODE_K = _K(886.81, 927.06, 512.0, 384.0)


def diode_list(split, root=RAW / "diode" / "val"):
    """split: 'indoors' or 'outdoor'"""
    return sorted((root / split).rglob("*.png"))


def diode_load(img_path):
    stem = str(img_path)[:-4]
    img = np.array(Image.open(img_path).convert("RGB"))
    depth = np.load(stem + "_depth.npy")[..., 0].astype(np.float32)
    mask = np.load(stem + "_depth_mask.npy") > 0
    depth[~mask] = 0
    tag = "diode-in" if "/indoors/" in str(img_path) else "diode-out"
    return {"img": img, "depth": depth, "K": DIODE_K.copy(), "name": f"{tag}/{img_path.stem}"}


# ---------------------------------------------------------------- one entry point
DOMAIN = {"kitti": "Outdoor", "nyu": "Indoor", "diode-in": "Indoor", "diode-out": "Outdoor"}
MAX_DEPTH = {"kitti": 80.0, "nyu": 10.0, "diode-in": 50.0, "diode-out": 150.0}


def sample_ids(dataset, n, seed=0):
    """A fixed random subset, so every model sees exactly the same images."""
    if dataset == "kitti":
        total = len(kitti_list())
    elif dataset == "nyu":
        total = len(NYU())
    else:
        total = len(diode_list("indoors" if dataset == "diode-in" else "outdoor"))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total, size=min(n, total), replace=False).tolist())


class Loader:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == "kitti":
            self.items = kitti_list()
        elif dataset == "nyu":
            self.nyu = NYU()
        else:
            self.items = diode_list("indoors" if dataset == "diode-in" else "outdoor")

    def __call__(self, i):
        if self.dataset == "kitti":
            return kitti_load(self.items[i])
        if self.dataset == "nyu":
            return self.nyu[i]
        return diode_load(self.items[i])
