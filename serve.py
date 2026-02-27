import napari
import shutil
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from magicgui import magicgui
import pandas as pd


def _load_rgb_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _load_he_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def build_viewer(
    he_rgb: np.ndarray,
    st_rgb: np.ndarray,
    emb_pca: np.ndarray,
    mask_path: str | None = None,
    mask_label_csv: str | None = None,
):
    v = napari.Viewer()

    he_h, he_w = he_rgb.shape[:2]
    emb_h, emb_w, emb_d = emb_pca.shape
    st_h, st_w = st_rgb.shape[:2]

    # 以 HE 像素为世界坐标系，把 ST/SIM 拉伸到 HE 尺寸（左上对齐）
    st_scale = (he_h / st_h, he_w / st_w)
    sim_scale = (he_h / emb_h, he_w / emb_w)  # world = emb * scale

    he = v.add_image(he_rgb, name="HE", rgb=True)
    st = v.add_image(st_rgb, name="ST_RGB", rgb=True, opacity=0.7, scale=st_scale)

    sim = v.add_image(
        np.zeros((emb_h, emb_w), dtype=np.float32),
        name="Similarity",
        colormap="turbo",
        opacity=0.75,
        scale=sim_scale,
        contrast_limits=(0.0, 1.0),
    )

    # ----------------------------
    # ✅ 读取并显示 mask + mask_label
    # ----------------------------
    lab = None
    if mask_path is not None:
        label_img = np.load(mask_path)  # (H,W) label map
        label_img = label_img.astype(np.uint32, copy=False)

        if label_img.shape == (he_h, he_w):
            # mask 已经是 HE 尺寸：直接叠加
            lab = v.add_labels(label_img, name="Mask", opacity=0.5)
        elif label_img.shape == (emb_h, emb_w):
            # mask 是 512x512：用 scale 拉伸到 HE 世界坐标
            lab = v.add_labels(label_img, name="Mask", opacity=0.5, scale=sim_scale)
        else:
            raise ValueError(
                f"mask shape {label_img.shape} 不匹配 HE {(he_h, he_w)} 或 EMB {(emb_h, emb_w)}"
            )

        # 读取实例表并挂到 features（可选）
        if mask_label_csv is not None:
            df = pd.read_csv(mask_label_csv)
            # 这张表通常是按 id 一行；napari 的 features 是“与 layer.data 行对应”，
            # 对 labels layer 不强制，但挂上去方便你自己后续查询/保存
            lab.metadata["instance_table"] = df

    # 如果没提供 mask，就给一个空 HE 尺寸 mask
    if lab is None:
        init_mask = np.zeros((he_h, he_w), dtype=np.uint32)
        lab = v.add_labels(init_mask, name="Mask", opacity=0.5)

    # ----------------------------
    # Query 标记：大圆圈 + 十字
    # ----------------------------
    query_circle = v.add_points(
        np.zeros((0, 2), dtype=np.float32),
        name="QueryCircle",
        size=80,
    )
    try:
        query_circle.face_color = "transparent"
    except Exception:
        query_circle.face_color = [0, 0, 0, 0]
    try:
        query_circle.edge_color = "cyan"
        query_circle.edge_width = 4
    except Exception:
        try:
            query_circle.edge_color = "cyan"
        except Exception:
            pass

    query_cross = v.add_shapes(
        [],
        shape_type="line",
        name="QueryCross",
        edge_color="cyan",
        edge_width=3,
    )

    # 默认显示
    he.visible = True
    st.visible = True
    sim.visible = True
    lab.visible = True

    # ----------------------------
    # 预计算 embedding norm
    # ----------------------------
    emb = emb_pca.astype(np.float32, copy=False).reshape(-1, emb_d)
    emb_norm = _normalize_rows(emb)

    sy, sx = sim_scale

    def world_to_emb_rc(world_yx):
        wy, wx = float(world_yx[0]), float(world_yx[1])
        r = int(np.clip(np.floor(wy / sy), 0, emb_h - 1))
        c = int(np.clip(np.floor(wx / sx), 0, emb_w - 1))
        return r, c

    params = {
        "sigma": 0.6,
        "gamma": 0.7,
        "q_low": 0.05,
        "q_high": 0.995,
        "colormap": "turbo",
        "auto_show_sim": False,
    }
    last_query = {"r": None, "c": None, "world_yx": None}

    CMAPS = sorted(AVAILABLE_COLORMAPS.keys())

    def set_colormap_safe(cmap_name: str):
        try:
            sim.colormap = cmap_name
        except Exception:
            sim.colormap = "viridis"

    def compute_and_render_similarity(r, c, world_yx):
        q = emb_norm[r * emb_w + c]
        sim_flat = emb_norm @ q
        sim_img = sim_flat.reshape(emb_h, emb_w)

        sim01 = (sim_img + 1.0) * 0.5

        sigma = float(params["sigma"])
        if sigma > 0:
            sim01 = gaussian_filter(sim01, sigma=sigma)

        gamma = float(params["gamma"])
        if abs(gamma - 1.0) > 1e-6:
            sim01 = np.power(np.clip(sim01, 0, 1), gamma)

        ql = float(np.clip(params["q_low"], 0.0, 1.0))
        qh = float(np.clip(params["q_high"], 0.0, 1.0))
        if qh <= ql:
            qh = min(1.0, ql + 1e-3)

        lo, hi = np.quantile(sim01, [ql, qh])
        if hi <= lo + 1e-6:
            lo, hi = float(sim01.min()), float(sim01.max() + 1e-6)

        sim.data = sim01.astype(np.float32, copy=False)
        try:
            sim.contrast_limits = (float(lo), float(hi))
        except Exception:
            pass

        wy, wx = float(world_yx[0]), float(world_yx[1])
        query_circle.data = np.asarray([[wy, wx]], dtype=np.float32)

        L = 70
        query_cross.data = [
            np.array([[wy - L, wx], [wy + L, wx]], dtype=np.float32),
            np.array([[wy, wx - L], [wy, wx + L]], dtype=np.float32),
        ]

        if params["auto_show_sim"]:
            sim.visible = True

    @magicgui(
        sigma={"min": 0.0, "max": 3.0, "step": 0.05},
        gamma={"min": 0.1, "max": 3.0, "step": 0.05},
        q_low={"min": 0.0, "max": 0.5, "step": 0.005},
        q_high={"min": 0.5, "max": 1.0, "step": 0.005},
        colormap={"choices": CMAPS},
        auto_show_sim={"widget_type": "CheckBox"},
    )
    def controls(
        sigma: float = 0.6,
        gamma: float = 0.7,
        q_low: float = 0.05,
        q_high: float = 0.995,
        colormap: str = "turbo",
        auto_show_sim: bool = False,
    ):
        params["sigma"] = float(sigma)
        params["gamma"] = float(gamma)
        params["q_low"] = float(q_low)
        params["q_high"] = float(q_high)
        params["colormap"] = str(colormap)
        params["auto_show_sim"] = bool(auto_show_sim)

        set_colormap_safe(params["colormap"])

        if last_query["r"] is not None:
            compute_and_render_similarity(last_query["r"], last_query["c"], last_query["world_yx"])

    v.window.add_dock_widget(controls, area="right", name="Similarity Controls")

    def mouse_cb(viewer, event):
        if event.type == "mouse_press" and event.button == 1:
            world_yx = event.position[:2]
            r, c = world_to_emb_rc(world_yx)
            last_query["r"], last_query["c"], last_query["world_yx"] = r, c, world_yx
            compute_and_render_similarity(r, c, world_yx)
        yield

    v.mouse_drag_callbacks.append(mouse_cb)

    @magicgui(call_button="Add Instance")
    def add_instance():
        current_max = int(lab.data.max())
        new_id = current_max + 1
        lab.selected_label = new_id
        print(f"[Mask] New instance id = {new_id}")

    @magicgui(call_button="Delete Selected Instance")
    def delete_instance():
        sel = int(lab.selected_label)

        if sel == 0:
            print("[Mask] Background (0) cannot be deleted.")
            return

        mask = lab.data == sel
        if mask.any():
            lab.data[mask] = 0
            lab.refresh()
            print(f"[Mask] Deleted instance id = {sel}")
        else:
            print(f"[Mask] Instance id {sel} not found.")

    @magicgui(call_button="Save Mask (npy)")
    def save_mask():
        save_path = mask_path
        np.save(save_path, lab.data.astype(np.uint32))
        print(f"[Mask] Saved to {save_path}")

    # 添加到右侧面板
    v.window.add_dock_widget(add_instance, area="right", name="Mask Tools")
    v.window.add_dock_widget(delete_instance, area="right")
    v.window.add_dock_widget(save_mask, area="right")

    return v, (he, st, sim, lab, query_circle, query_cross)


if __name__ == "__main__":
    input_dir = "/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/Visium_HD_Human_Kidney_FFPE/maskann_size-512"
    coord = (0, 512)
    rgb_png_path = f"{input_dir}/pca_rgb_no/r{coord[0]}_c{coord[1]}_rgb.png"
    he_path = f"{input_dir}/HE/r{coord[0]}_c{coord[1]}_he.png"
    pca_path = f"{input_dir}/pca50/r{coord[0]}_c{coord[1]}_pca.npy"
    mask_path = f"{input_dir}/sam2_merged_masks/r{coord[0]}_c{coord[1]}_merged_mask.npy"
    mask_label = f"{input_dir}/sam2_merged_masks/r{coord[0]}_c{coord[1]}_merged_mask_info.csv"

    # input_dir = "/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/Visium_HD_Human_Colon_Cancer_P2/maskann_size-512"
    # rgb_png_path = f"/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/demo_data/r896_c2304_rgb.png"
    # he_path = f"/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/demo_data/r896_c2304.png"
    # pca_path = f"/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/demo_data/r896_c2304.npy"
    # mask_path = f"/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/demo_data/r896_c2304_mask.npy"
    # mask_label = f"/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/demo_data/r896_c2304_inst_info.csv"

    edited_mask_path = mask_path.replace(".npy", "_edited.npy")
    if not os.path.exists(edited_mask_path):    
        shutil.copy(mask_path, edited_mask_path)

    emb_pca = np.load(pca_path)
    st_rgb = _load_rgb_png(rgb_png_path)
    he_rgb = _load_he_rgb(he_path)

    v, layers = build_viewer(
        he_rgb=he_rgb,
        st_rgb=st_rgb,
        emb_pca=emb_pca,
        mask_path=edited_mask_path,
        mask_label_csv=mask_label,
    )
    napari.run()