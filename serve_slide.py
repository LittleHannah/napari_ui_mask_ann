"""
serve_slide.py
--------------
Slide-level viewer 入口。

流程：
  1. 用全局 pca3 + 坐标文件构建 slide-level RGB canvas（首次运行保存，之后读缓存）
  2. 用全局 PCA50 + 坐标文件构建 slide-level embedding canvas（首次运行保存，之后读缓存）
  3. 以 prefix="slide" 调用 build_viewer —— 共享所有 tile viewer 的功能（SAM2 / Lock / Merge / Save 等）
  4. HE 图层不存在时 viewer.py 自动隐藏，其余完全一致

运行：
    python serve_slide.py
"""

import os

import numpy as np
from PIL import Image

import napari
from viewer import build_viewer
from slide_viewer import _discover_tiles, build_global_rgb_canvas, build_global_emb_canvas

# ── 配置 ─────────────────────────────────────────────────────────────────────
INPUT_DIR = (
    "/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/"
    "Visium_HD_Human_Colon_Cancer_P2/maskann_size-512"
)
MASK_DIR = os.path.join(INPUT_DIR, "sam2_merged_masks")

# 全局 PCA 文件（whole-slide level，无 sign ambiguity 问题）
DATA_DIR    = os.path.dirname(INPUT_DIR)   # .../Visium_HD_Human_Colon_Cancer_P2/
PCA3_PATH   = os.path.join(DATA_DIR, "Visium_HD_Human_Colon_Cancer_P2_pca3.npy")
PCA_PATH    = os.path.join(DATA_DIR, "Visium_HD_Human_Colon_Cancer_P2_pca50.npy")
COORDS_PATH = os.path.join(DATA_DIR, "Visium_HD_Human_Colon_Cancer_P2_filtered_coords.npy")

EMB_DOWNSAMPLE = 1  # embedding 降采样倍数（ds=4 → ~128MB canvas）


def prepare_slide_files(input_dir: str, mask_dir: str,
                        pca3_path: str, pca_path: str, coords_path: str,
                        downsample: int = 4, tile_size: int = 512) -> str:
    """
    保证以下文件存在（缓存命中则跳过）：
      pca_rgb_no/slide_rgb.png    ← global pca3 scatter RGB
      emb/slide_emb.npy           ← global PCA50 embedding canvas
    返回 prefix = "slide"。
    """
    prefix   = "slide"
    rgb_path = os.path.join(input_dir, "pca_rgb_no", f"{prefix}_rgb.png")
    emb_path = os.path.join(input_dir, "emb",        f"{prefix}_emb.npy")

    # 确保 slide mask 存在（merge_slide_masks.ipynb 的输出）
    if not os.path.exists(os.path.join(mask_dir, "slide_merged_mask.npy")):
        raise FileNotFoundError(
            "slide_merged_mask.npy not found. 请先运行 merge_slide_masks.ipynb。"
        )

    # ── Discover tiles → canvas size ──
    tiles    = _discover_tiles(mask_dir)
    all_rows = sorted(set(t["row"] for t in tiles))
    all_cols = sorted(set(t["col"] for t in tiles))
    canvas_H = max(all_rows) + tile_size
    canvas_W = max(all_cols) + tile_size
    print(f"[Slide] canvas {canvas_H}×{canvas_W}, {len(tiles)} tiles")

    # ── Global RGB canvas (from pca3) ──
    os.makedirs(os.path.join(input_dir, "pca_rgb_no"), exist_ok=True)
    if not os.path.exists(rgb_path):
        st_rgb = build_global_rgb_canvas(pca3_path, coords_path, canvas_H, canvas_W)
        Image.fromarray(st_rgb).save(rgb_path)
        print(f"[Slide] Saved → {rgb_path}  ({os.path.getsize(rgb_path)/1e6:.1f} MB)")
    else:
        print(f"[Slide] ST_RGB cache: {rgb_path}")

    # ── Global embedding canvas ──
    os.makedirs(os.path.join(input_dir, "emb"), exist_ok=True)
    if not os.path.exists(emb_path):
        emb = build_global_emb_canvas(
            pca_path, coords_path, canvas_H, canvas_W, downsample=downsample,
        )
        np.save(emb_path, emb)
        print(f"[Slide] Saved → {emb_path}  ({os.path.getsize(emb_path)/1e6:.0f} MB)")
    else:
        print(f"[Slide] Emb cache: {emb_path}")

    return prefix


if __name__ == "__main__":
    prefix = prepare_slide_files(
        INPUT_DIR, MASK_DIR, PCA3_PATH, PCA_PATH, COORDS_PATH, EMB_DOWNSAMPLE
    )
    v = build_viewer(input_dir=INPUT_DIR, initial_prefix=prefix)
    napari.run()
