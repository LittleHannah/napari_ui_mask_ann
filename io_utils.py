"""
io_utils.py
-----------
纯 IO 工具：图像加载、文件路径管理、prefix 发现。
不依赖 napari / magicgui，可单独测试。
"""
import os
import glob
import shutil
import re

import numpy as np
from PIL import Image


# ------------------------------------------------------------------
# 图像加载
# ------------------------------------------------------------------

def load_rgb_png(path: str) -> np.ndarray:
    """读取任意 PNG，强制转为 uint8 RGB。"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_he_rgb(path: str) -> np.ndarray:
    """读取 H&E PNG（与 load_rgb_png 相同，保留语义命名）。"""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# ------------------------------------------------------------------
# 嵌入归一化
# ------------------------------------------------------------------

def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对矩阵每行做 L2 归一化，返回 shape 不变的数组。"""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


# ------------------------------------------------------------------
# Prefix 解析与发现
# ------------------------------------------------------------------

def _parse_prefix_from_filename(fname: str) -> str | None:
    """从文件名中提取 rXXX_cYYY 格式的 prefix，找不到返回 None。"""
    m = re.search(r"(r\d+_c\d+)", os.path.basename(fname))
    return m.group(1) if m else None


def discover_prefixes(input_dir: str) -> list[str]:
    """
    扫描 input_dir，返回同时拥有 HE + PCA + ST_RGB 三个文件的 prefix 列表。
    prefix 格式: rXXX_cYYY（例如 r0_c512）
    """
    pca_dir = os.path.join(input_dir, "pca50")
    he_dir  = os.path.join(input_dir, "HE")
    rgb_dir = os.path.join(input_dir, "pca_rgb_no")

    pca_files = sorted(glob.glob(os.path.join(pca_dir, "r*_c*_pca.npy")))
    prefixes = []
    for p in pca_files:
        prefix = _parse_prefix_from_filename(p)
        if prefix is None:
            continue
        he_path  = os.path.join(he_dir,  f"{prefix}_he.png")
        rgb_path = os.path.join(rgb_dir, f"{prefix}_rgb.png")
        if os.path.exists(he_path) and os.path.exists(rgb_path):
            prefixes.append(prefix)
    return prefixes


# ------------------------------------------------------------------
# 路径集中管理
# ------------------------------------------------------------------

def paths_for_prefix(input_dir: str, prefix: str) -> dict[str, str]:
    """
    返回某 prefix 所有相关文件的路径字典，统一命名规范。
    包括原始文件路径和编辑副本路径。
    """
    he_path      = os.path.join(input_dir, "HE",          f"{prefix}_he.png")
    rgb_png_path = os.path.join(input_dir, "pca_rgb_no",  f"{prefix}_rgb.png")
    pca_path     = os.path.join(input_dir, "emb",       f"{prefix}_emb.npy")

    mask_dir       = os.path.join(input_dir, "sam2_merged_masks")
    mask_path      = os.path.join(mask_dir,  f"{prefix}_merged_mask.npy")
    mask_label_csv = os.path.join(mask_dir,  f"{prefix}_merged_mask_info.csv")

    edited_mask_path = mask_path.replace(".npy", "_edited.npy")
    edited_csv_path  = mask_label_csv.replace(".csv", "_edited.csv")

    return dict(
        he_path=he_path,
        rgb_png_path=rgb_png_path,
        pca_path=pca_path,
        mask_path=mask_path,
        mask_label_csv=mask_label_csv,
        edited_mask_path=edited_mask_path,
        edited_csv_path=edited_csv_path,
    )


def ensure_edited_mask_exists(input_dir: str, prefix: str) -> dict[str, str]:
    """
    若 edited mask 不存在，从原始 mask 复制一份。
    返回 paths_for_prefix 字典。
    """
    p = paths_for_prefix(input_dir, prefix)
    if not os.path.exists(p["edited_mask_path"]):
        if os.path.exists(p["mask_path"]):
            shutil.copy(p["mask_path"], p["edited_mask_path"])
    return p
