"""
mask_utils.py
-------------
Mask / instance-table 工具：bbox 计算、instance table 重建。
不依赖 napari / magicgui，可单独测试。
"""
import numpy as np
import pandas as pd


def bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """
    给定布尔 mask，返回 (y0, x0, height, width)。
    mask 为空时返回 None。
    """
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return float(y0), float(x0), float(y1 - y0), float(x1 - x0)


def rebuild_instance_table_from_labels(
    label_img: np.ndarray,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    从 label 图像重建 instance table（id, area, bbox_y/x/h/w）。

    注意：predicted_iou / stability_score 等 SAM2 原始字段无法从编辑后的
    label 图恢复，因此这里只重建几何基础列。
    keep_cols: 若指定，只保留存在于结果中的列。
    """
    ids = np.unique(label_img)
    ids = ids[ids != 0]
    rows = []
    for i in ids:
        m = (label_img == i)
        area = int(m.sum())
        bb = bbox_from_mask(m)
        if bb is None:
            continue
        by, bx, bh, bw = bb
        rows.append(dict(id=int(i), area=area, bbox_y=by, bbox_x=bx, bbox_h=bh, bbox_w=bw))

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)

    if keep_cols is not None:
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols]

    return df


def merge_instance_table(
    old_df: pd.DataFrame,
    label_img: np.ndarray,
) -> pd.DataFrame:
    """
    merge 操作后更新 instance table：
    重建几何列，尽量保留 old_df 中的额外列（left join by id）。
    """
    keep_cols = list(old_df.columns)
    new_df = rebuild_instance_table_from_labels(label_img)

    new_df2 = new_df.merge(old_df, on="id", how="left", suffixes=("", "_old"))
    for col in ["area", "bbox_y", "bbox_x", "bbox_h", "bbox_w"]:
        if col in new_df.columns:
            new_df2[col] = new_df[col].values

    cols = [c for c in keep_cols if c in new_df2.columns]
    if cols:
        new_df2 = new_df2[cols]

    return new_df2
