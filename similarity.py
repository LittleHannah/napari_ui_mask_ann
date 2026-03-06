"""
similarity.py
-------------
DINOv3 embedding 相似度计算：纯 numpy 逻辑，不依赖 napari。
可单独测试，也方便以后替换为 GPU 加速版本。
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def region_query_vec(
    emb_norm: np.ndarray,
    r: int,
    c: int,
    radius: int,
    emb_h: int,
    emb_w: int,
) -> np.ndarray:
    """
    在 (r, c) 为圆心、radius 为半径（patch 单位）的圆形区域内，
    平均所有 patch 的 embedding，返回归一化后的 query vector。
    """
    rs = np.arange(max(0, r - radius), min(emb_h, r + radius + 1))
    cs = np.arange(max(0, c - radius), min(emb_w, c + radius + 1))
    rr, cc = np.meshgrid(rs, cs, indexing="ij")
    dist = np.sqrt((rr - r) ** 2 + (cc - c) ** 2)
    in_circle = dist <= radius
    indices = rr[in_circle] * emb_w + cc[in_circle]
    vecs = emb_norm[indices]          # (N, D)
    avg  = vecs.mean(axis=0)          # (D,)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 1e-8 else avg


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
    query_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    以 (r, c) 处的 patch embedding（或外部传入的 query_vec）为 query，计算全图余弦相似度。

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
    q = query_vec if query_vec is not None else emb_norm[r * emb_w + c]   # (D,)
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
