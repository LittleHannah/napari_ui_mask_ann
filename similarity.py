"""
similarity.py
-------------
DINOv3 embedding 相似度计算：纯 numpy 逻辑，不依赖 napari。
可单独测试，也方便以后替换为 GPU 加速版本。
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def world_to_emb_rc(
    world_yx,
    sy: float,
    sx: float,
    emb_h: int,
    emb_w: int,
) -> tuple[int, int]:
    """
    将 napari world 坐标（像素级 HE 坐标）映射到 embedding 网格的 (row, col)。

    world_yx : napari event.position[:2]，单位是 HE 像素
    sy, sx   : sim_scale = (he_h / emb_h, he_w / emb_w)，即每个 patch 覆盖的像素数
    """
    wy, wx = float(world_yx[0]), float(world_yx[1])
    r = int(np.clip(np.floor(wy / sy), 0, emb_h - 1))
    c = int(np.clip(np.floor(wx / sx), 0, emb_w - 1))
    return r, c


def compute_similarity(
    emb_norm: np.ndarray,   # shape (H*W, D)，已 L2 归一化
    r: int,
    c: int,
    emb_h: int,
    emb_w: int,
    params: dict,
) -> tuple[np.ndarray, float, float]:
    """
    以 (r, c) 处的 patch embedding 为 query，计算全图余弦相似度。

    处理流程：
      1. 余弦相似度（归一化向量点积） → [-1, 1]
      2. 映射到 [0, 1]
      3. 高斯平滑（sigma）
      4. Gamma 校正
      5. 分位数拉伸（q_low / q_high）

    返回 (sim01, lo, hi)：
      sim01 : float32 array，shape (emb_h, emb_w)，值域 [0, 1]
      lo/hi : 对应 contrast_limits
    """
    q = emb_norm[r * emb_w + c]          # (D,)
    sim_flat = emb_norm @ q               # (H*W,)
    sim_img  = sim_flat.reshape(emb_h, emb_w)
    sim01    = (sim_img + 1.0) * 0.5      # [-1,1] → [0,1]

    sigma = float(params["sigma"])
    if sigma > 0:
        sim01 = gaussian_filter(sim01, sigma=sigma)

    gamma = float(params["gamma"])
    if abs(gamma - 1.0) > 1e-6:
        sim01 = np.power(np.clip(sim01, 0, 1), gamma)

    ql = float(np.clip(params["q_low"],  0.0, 1.0))
    qh = float(np.clip(params["q_high"], 0.0, 1.0))
    if qh <= ql:
        qh = min(1.0, ql + 1e-3)

    lo, hi = np.quantile(sim01, [ql, qh])
    if hi <= lo + 1e-6:
        lo, hi = float(sim01.min()), float(sim01.max() + 1e-6)

    return sim01.astype(np.float32, copy=False), float(lo), float(hi)
