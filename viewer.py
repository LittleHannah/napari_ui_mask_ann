"""
viewer.py
---------
napari viewer 构建器。
改动：
  1. Lock 可视化：专用 RGBA overlay 图层（黄色高亮锁定区域）+ 回调防重入守卫
  2. Merge 交互：点击模式选择 → 确认/取消
  3. UI：右侧 Qt 面板（Annotation Tools）+ 左侧 Qt 面板（Data Selector）
"""

import base64
import io
import os

import matplotlib.cm as mpl_cm
import numpy as np
import pandas as pd
import requests
import napari
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from magicgui import magicgui
from PIL import Image as PILImage
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QComboBox,
)

from io_utils import (
    load_he_rgb, load_rgb_png, normalize_rows,
    discover_prefixes, ensure_edited_mask_exists,
)
from mask_utils import rebuild_instance_table_from_labels, merge_instance_table
from similarity import world_to_emb_rc, compute_similarity, region_query_vec


# ── SAM2 worker thread ────────────────────────────────────────────────────────

class _SAM2Worker(QThread):
    """Sends image + box prompts to the SAM2 server in a background thread."""
    finished = Signal(list)   # list of np.ndarray bool H×W masks
    error    = Signal(str)

    def __init__(self, url: str, image_arr: np.ndarray, boxes: list):
        super().__init__()
        self._url       = url.rstrip("/") + "/segment"
        self._image_arr = image_arr   # H×W×3 uint8
        self._boxes     = boxes       # [[x1,y1,x2,y2], ...]

    def run(self):
        try:
            pil_img   = PILImage.fromarray(self._image_arr)
            buf       = io.BytesIO()
            pil_img.save(buf, format="PNG")
            img_b64   = base64.b64encode(buf.getvalue()).decode("ascii")

            payload = {"image_b64": img_b64, "boxes": self._boxes}
            resp    = requests.post(self._url, json=payload, timeout=120)
            resp.raise_for_status()
            data    = resp.json()

            H, W    = data["shape"]
            masks   = []
            for m_b64 in data["masks"]:
                raw  = base64.b64decode(m_b64)
                arr  = np.frombuffer(raw, dtype=np.uint8).reshape(H, W).astype(bool)
                masks.append(arr)

            self.finished.emit(masks)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Qt 样式表 ─────────────────────────────────────────────────────────────────
PANEL_STYLE = """
QWidget {
    background-color: #2b2b2b;
}
QLabel {
    color: #cccccc;
    font-size: 12px;
}
QGroupBox {
    color: #cccccc;
    font-weight: bold;
    font-size: 12px;
    border: 1px solid #4a4a4a;
    border-radius: 5px;
    margin-top: 12px;
    padding: 10px 6px 6px 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #3c3f41;
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
    min-height: 24px;
}
QPushButton:hover  { background-color: #4c5052; border-color: #777; }
QPushButton:pressed { background-color: #2a2a2a; }
QPushButton:disabled { background-color: #282828; color: #555; border-color: #383838; }

QPushButton#deleteBtn  { color: #ff8888; border-color: #7a3a3a; }
QPushButton#deleteBtn:hover { background-color: #4a2020; }

QPushButton#lockBtn    { color: #ffdd44; border-color: #7a6a1a; }
QPushButton#lockBtn:hover { background-color: #3a3010; }

QPushButton#mergeStartBtn { color: #66ccff; border-color: #336688; }
QPushButton#mergeStartBtn:hover { background-color: #1a3a55; }
QPushButton#mergeActiveBtn { background-color: #1a3d5c; color: #99ddff; border-color: #4a9aee; }

QPushButton#confirmBtn { background-color: #1a4a1a; color: #aaffaa; border-color: #3a8a3a; }
QPushButton#confirmBtn:hover { background-color: #2a6a2a; }

QPushButton#saveBtn {
    background-color: #2a4a2a; color: #ccffcc;
    border-color: #4a7a4a; font-weight: bold; min-height: 28px;
}
QPushButton#saveBtn:hover { background-color: #3a5a3a; }

QLabel#statusLabel  { color: #888888; font-size: 11px; padding: 2px 4px; }
QLabel#mergeLabel   { color: #66aaff; font-size: 11px; padding: 2px 4px; font-style: italic; }

QPushButton#sam2RunBtn  { background-color: #1a4a1a; color: #aaffaa; border-color: #3a8a3a; font-weight: bold; }
QPushButton#sam2RunBtn:hover { background-color: #2a6a2a; }
QPushButton#sam2RunBtn:disabled { background-color: #282828; color: #555; border-color: #383838; }
QPushButton#sam2DrawBtn { color: #ffdd44; border-color: #7a6a1a; }
QPushButton#sam2DrawBtn:hover { background-color: #3a3010; }
QPushButton#sam2DrawActiveBtn { background-color: #3a3010; color: #ffee66; border-color: #aa9922; }
QLabel#sam2StatusLabel { color: #888888; font-size: 11px; padding: 2px 4px; font-style: italic; }

QPushButton#doneBtn { color: #aaaaaa; border-color: #555; }
QPushButton#doneBtnActive { background-color: #1a3a5a; color: #55ddff; border-color: #2299cc; font-weight: bold; }

QPushButton#fixSimBtn { color: #cccccc; border-color: #555; }
QPushButton#fixSimActiveBtn { background-color: #3a1a5a; color: #cc88ff; border-color: #8844cc; font-weight: bold; }

QPushButton#loadBtn { background-color: #1a3a5a; color: #88ccff; border-color: #336699; font-weight: bold; min-height: 28px; }
QPushButton#loadBtn:hover { background-color: #2a4a6a; }

QComboBox { background-color: #3c3f41; color: #cccccc; border: 1px solid #555555; border-radius: 4px; padding: 3px 6px; }
QComboBox::drop-down { border: none; }
QLineEdit { background-color: #3c3f41; color: #cccccc; border: 1px solid #555555; border-radius: 4px; padding: 3px 6px; font-size: 11px; }
"""


# ── 主函数 ────────────────────────────────────────────────────────────────────
def build_viewer(input_dir: str, initial_prefix: str | None = None):

    v = napari.Viewer()

    # ── 图层初始化 ──────────────────────────────────────────────────────────
    he  = v.add_image(np.zeros((10, 10, 3), dtype=np.uint8), name="HE",       rgb=True)
    st  = v.add_image(np.zeros((10, 10, 3), dtype=np.uint8), name="ST_RGB",   rgb=True, opacity=0.7)
    sim = v.add_image(
        np.zeros((10, 10), dtype=np.float32),
        name="Similarity", colormap="turbo", opacity=0.75, contrast_limits=(0.0, 1.0),
    )
    lab = v.add_labels(np.zeros((10, 10), dtype=np.uint32), name="Mask", opacity=0.5)

    # RGBA overlay：lock（黄色）
    lock_overlay = v.add_image(
        np.zeros((10, 10, 4), dtype=np.float32),
        name="LockOverlay", rgb=True, opacity=0.6,
    )

    # SAM2 box prompts layer
    sam_boxes = v.add_shapes(
        [], shape_type="rectangle", name="SAMBoxes",
        edge_color="yellow", face_color="transparent", edge_width=2,
    )

    # Query 十字标记
    query_cross = v.add_shapes([], shape_type="line", name="QueryCross", edge_color="cyan", edge_width=3)

    # ── 数据集根目录 ─────────────────────────────────────────────────────────
    # data_root / experiment_name / size_suffix
    data_root   = os.path.dirname(os.path.dirname(input_dir))
    size_suffix = os.path.basename(input_dir)

    # ── 状态 ────────────────────────────────────────────────────────────────
    state = {
        "input_dir": input_dir, "prefix": None,
        "he_h": None, "he_w": None,
        "emb_h": None, "emb_w": None, "emb_d": None,
        "st_h": None,  "st_w": None,
        "st_scale": (1.0, 1.0), "sim_scale": (1.0, 1.0),
        "sy": 1.0, "sx": 1.0,
        "emb_norm": None,
        "last_query": {"r": None, "c": None, "world_yx": None},
        # lock
        "locked_ids": set(),
        "prev_labels_snapshot": None,
        # paths
        "edited_mask_path": None, "edited_csv_path": None, "raw_csv_path": None,
        "locked_ids_path": None,
        # merge
        "merge_active": False, "merge_candidates": set(),
        # sam2
        "sam2_drawing": False,
        "sam2_worker": None,    # holds reference to running QThread to prevent GC
        # done
        "done_prefixes": set(),
        # similarity
        "sim_fixed": False,
    }

    def _done_file() -> str:
        return os.path.join(state["input_dir"], "annotation_done.txt")

    def _load_done_set() -> set[str]:
        f = _done_file()
        if not os.path.exists(f):
            return set()
        with open(f) as fh:
            return {ln.strip() for ln in fh if ln.strip()}

    def _save_done_set():
        with open(_done_file(), "w") as f:
            for p in sorted(state["done_prefixes"]):
                f.write(p + "\n")

    state["done_prefixes"] = _load_done_set()

    def _discover_datasets() -> list[str]:
        """List experiment subdirs of data_root that contain a size_suffix subdir."""
        if not os.path.isdir(data_root):
            return [os.path.basename(os.path.dirname(input_dir))]
        result = []
        for d in sorted(os.listdir(data_root)):
            if os.path.isdir(os.path.join(data_root, d, size_suffix)):
                result.append(d)
        return result or [os.path.basename(os.path.dirname(input_dir))]

    params = {
        "sigma": 1.5, "gamma": 0.7,
        "q_low": 0.05, "q_high": 0.995,
        "colormap": "turbo", "auto_show_sim": False,
        "query_radius": 6,
    }
    CMAPS = sorted(AVAILABLE_COLORMAPS.keys())

    # lock callback 防重入守卫
    _lg = {"reverting": False}

    # ── 内部工具函数 ────────────────────────────────────────────────────────

    def _set_colormap_safe(name: str):
        try:   sim.colormap = name
        except Exception: sim.colormap = "viridis"

    def _world_to_label_rc(world_yx) -> tuple[int, int]:
        """world 坐标 → label 数组 (row, col)，自动适应 lab.scale。"""
        wy, wx = float(world_yx[0]), float(world_yx[1])
        sy_l = float(lab.scale[0]) or 1.0
        sx_l = float(lab.scale[1]) or 1.0
        h, w = lab.data.shape[:2]
        r = int(np.clip(np.floor(wy / sy_l), 0, h - 1))
        c = int(np.clip(np.floor(wx / sx_l), 0, w - 1))
        return r, c

    def _update_lock_overlay():
        """用黄色 RGBA 高亮所有被锁定的 instance 像素。"""
        locked = state["locked_ids"]
        shape  = lab.data.shape[:2]
        ov = np.zeros((*shape, 4), dtype=np.float32)
        if locked:
            mask = np.isin(lab.data, list(locked))
            ov[mask] = [1.0, 0.88, 0.0, 1.0]
        lock_overlay.data  = ov
        lock_overlay.scale = lab.scale

    def _render_similarity(r, c, world_yx):
        radius = int(params["query_radius"])
        qvec = None
        if radius > 0:
            qvec = region_query_vec(
                state["emb_norm"], r, c, radius, state["emb_h"], state["emb_w"],
            )
        sim01, lo, hi = compute_similarity(
            state["emb_norm"], r, c, state["emb_h"], state["emb_w"], params, query_vec=qvec,
        )
        sim.data = sim01
        try: sim.contrast_limits = (lo, hi)
        except Exception: pass

        wy, wx = float(world_yx[0]), float(world_yx[1])
        L = 70
        query_cross.data = [
            np.array([[wy - L, wx], [wy + L, wx]], dtype=np.float32),
            np.array([[wy, wx - L], [wy, wx + L]], dtype=np.float32),
        ]
        if params["auto_show_sim"]:
            sim.visible = True

    # ── SAM2 helpers ─────────────────────────────────────────────────────────

    def _get_layer_image_and_scale(layer_name: str):
        """Return (H×W×3 uint8 image, sy, sx) for the named layer."""
        if layer_name == "HE":
            img = he.data
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 255)).astype(np.uint8)
            return img, 1.0, 1.0
        elif layer_name == "ST_RGB":
            img = st.data
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 255)).astype(np.uint8)
            sy, sx = state["st_scale"]
            return img, sy, sx
        elif layer_name == "Similarity":
            sim_data = sim.data.astype(np.float32)
            cmap     = mpl_cm.get_cmap(params["colormap"])
            rgb      = (cmap(sim_data)[:, :, :3] * 255).astype(np.uint8)
            sy, sx   = state["sim_scale"]
            return rgb, sy, sx
        else:
            raise ValueError(f"Unknown layer: {layer_name}")

    def _boxes_to_pixel_coords(layer_name: str):
        """Convert SAMBoxes world-coord rectangles to pixel coords [x1,y1,x2,y2]."""
        img_arr, sy, sx = _get_layer_image_and_scale(layer_name)
        img_h, img_w    = img_arr.shape[:2]

        boxes_px = []
        for shape_data in sam_boxes.data:
            pts = np.array(shape_data)  # (4, 2)
            wy_min, wy_max = pts[:, 0].min(), pts[:, 0].max()
            wx_min, wx_max = pts[:, 1].min(), pts[:, 1].max()
            r1 = int(np.clip(wy_min / sy, 0, img_h - 1))
            r2 = int(np.clip(wy_max / sy, 0, img_h - 1))
            c1 = int(np.clip(wx_min / sx, 0, img_w - 1))
            c2 = int(np.clip(wx_max / sx, 0, img_w - 1))
            boxes_px.append([float(c1), float(r1), float(c2), float(r2)])
        return boxes_px

    def _resize_mask_to_lab(mask_hw: np.ndarray) -> np.ndarray:
        """Resize a bool mask to match lab.data.shape using nearest-neighbour."""
        target_h, target_w = lab.data.shape[:2]
        if mask_hw.shape == (target_h, target_w):
            return mask_hw
        pil_mask = PILImage.fromarray(mask_hw.astype(np.uint8) * 255)
        pil_resized = pil_mask.resize((target_w, target_h), PILImage.NEAREST)
        return np.array(pil_resized) > 127

    def _apply_sam2_results(masks_list: list):
        """Write SAM2 masks into lab.data, respecting locked IDs."""
        base_id = int(lab.data.max()) + 1
        data    = lab.data.copy()
        locked  = state["locked_ids"]
        for i, mask in enumerate(masks_list):
            resized      = _resize_mask_to_lab(mask)
            write_pixels = resized & (~np.isin(data, list(locked)) if locked else resized)
            data[write_pixels] = base_id + i
        lab.data = data
        lab.refresh()
        state["prev_labels_snapshot"] = lab.data.copy()
        _update_lock_overlay()
        print(f"[SAM2] Applied {len(masks_list)} mask(s), IDs {base_id}–{base_id + len(masks_list) - 1}")

    # ── Prefix 加载 ─────────────────────────────────────────────────────────
    def load_prefix(prefix: str):
        p = ensure_edited_mask_exists(state["input_dir"], prefix)

        st_rgb  = load_rgb_png(p["rgb_png_path"])
        emb_pca = np.load(p["pca_path"])
        st_h, st_w        = st_rgb.shape[:2]
        emb_h, emb_w, emb_d = emb_pca.shape

        if os.path.exists(p["he_path"]):
            he_rgb     = load_he_rgb(p["he_path"])
            he.data    = he_rgb
            he.visible = True
        else:
            # no HE (e.g. slide mode): use ST_RGB dims as reference, hide layer
            he_rgb     = np.zeros((st_h, st_w, 3), dtype=np.uint8)
            he.data    = he_rgb
            he.visible = False
            if sam2_layer_cb.currentText() == "HE":
                sam2_layer_cb.setCurrentIndex(1)   # fall back to ST_RGB

        he_h, he_w   = he_rgb.shape[:2]
        st_scale  = (he_h / st_h,  he_w / st_w)
        sim_scale = (he_h / emb_h, he_w / emb_w)

        # mask 优先级：edited > raw > blank
        label_img = None
        if os.path.exists(p["edited_mask_path"]):
            label_img = np.load(p["edited_mask_path"]).astype(np.uint32, copy=False)
        elif os.path.exists(p["mask_path"]):
            label_img = np.load(p["mask_path"]).astype(np.uint32, copy=False)
        if label_img is None:
            label_img = np.zeros((emb_h, emb_w), dtype=np.uint32)

        if label_img.shape == (he_h, he_w):
            lab.scale = (1.0, 1.0)
        elif label_img.shape == (emb_h, emb_w):
            lab.scale = sim_scale
        else:
            raise ValueError(
                f"mask shape {label_img.shape} not match HE {(he_h, he_w)} or EMB {(emb_h, emb_w)}"
            )

        st.data = st_rgb;  st.scale = st_scale
        sim.data = np.zeros((emb_h, emb_w), dtype=np.float32)
        sim.scale = sim_scale
        _set_colormap_safe(params["colormap"])
        try: sim.contrast_limits = (0.0, 1.0)
        except Exception: pass
        lab.data = label_img;  lab.refresh()

        df = None
        if os.path.exists(p["edited_csv_path"]):
            df = pd.read_csv(p["edited_csv_path"])
        elif os.path.exists(p["mask_label_csv"]):
            df = pd.read_csv(p["mask_label_csv"])
        if df is not None:
            lab.metadata["instance_table"] = df
        else:
            lab.metadata.pop("instance_table", None)
        state["raw_csv_path"] = p["mask_label_csv"]

        state.update(dict(
            prefix=prefix,
            he_h=he_h, he_w=he_w, st_h=st_h, st_w=st_w,
            emb_h=emb_h, emb_w=emb_w, emb_d=emb_d,
            st_scale=st_scale, sim_scale=sim_scale,
            sy=sim_scale[0], sx=sim_scale[1],
            edited_mask_path=p["edited_mask_path"],
            edited_csv_path=p["edited_csv_path"],
            locked_ids_path=p["locked_ids_path"],
            locked_ids=set(),
            merge_active=False, merge_candidates=set(),
        ))

        # 从文件恢复锁定 ID
        _lpath = p["locked_ids_path"]
        if os.path.exists(_lpath):
            import json
            with open(_lpath) as _f:
                state["locked_ids"] = set(json.load(_f))
            print(f"[Lock] Restored {len(state['locked_ids'])} locked ID(s) from file")

        emb = emb_pca.astype(np.float32, copy=False).reshape(-1, emb_d)
        state["emb_norm"] = normalize_rows(emb)

        query_cross.data  = []
        state["prev_labels_snapshot"] = lab.data.copy()

        _update_lock_overlay()
        _update_done_btn()
        print(f"[Load] {prefix} | HE={he_rgb.shape} ST={st_rgb.shape} EMB={emb_pca.shape} MASK={label_img.shape}")

    # ── Similarity controls widget (magicgui) ───────────────────────────────
    @magicgui(
        sigma={"min": 0.0, "max": 3.0, "step": 0.05},
        gamma={"min": 0.1, "max": 3.0, "step": 0.05},
        q_low={"min": 0.0, "max": 0.5, "step": 0.005},
        q_high={"min": 0.5, "max": 1.0, "step": 0.005},
        colormap={"choices": CMAPS},
        auto_show_sim={"widget_type": "CheckBox"},
        query_radius={"min": 0, "max": 20, "step": 1, "label": "Query Radius (patches)"},
    )
    def controls(
        sigma: float = 1.5, gamma: float = 0.7,
        q_low: float = 0.05, q_high: float = 0.995,
        colormap: str = "turbo", auto_show_sim: bool = False,
        query_radius: int = 6,
    ):
        params.update(dict(
            sigma=float(sigma), gamma=float(gamma),
            q_low=float(q_low), q_high=float(q_high),
            colormap=str(colormap), auto_show_sim=bool(auto_show_sim),
            query_radius=int(query_radius),
        ))
        _set_colormap_safe(params["colormap"])
        if state["last_query"]["r"] is not None:
            _render_similarity(
                state["last_query"]["r"], state["last_query"]["c"], state["last_query"]["world_yx"],
            )

    v.window.add_dock_widget(controls, area="right", name="Similarity Controls")

    # ── Qt Annotation Panel (right) ──────────────────────────────────────────

    panel = QWidget()
    panel.setObjectName("AnnotationPanel")
    panel_layout = QVBoxLayout(panel)
    panel_layout.setSpacing(10)
    panel_layout.setContentsMargins(8, 8, 8, 8)
    panel.setStyleSheet(PANEL_STYLE)

    # ---- Similarity Fix group ----
    sim_grp    = QGroupBox("Similarity")
    sim_layout = QVBoxLayout();  sim_layout.setSpacing(6)
    btn_fix_sim = QPushButton("📌 Fix Similarity Map")
    btn_fix_sim.setObjectName("fixSimBtn")
    sim_layout.addWidget(btn_fix_sim)
    sim_grp.setLayout(sim_layout)

    # ---- Mask group ----
    mask_grp    = QGroupBox("Mask")
    mask_layout = QVBoxLayout();  mask_layout.setSpacing(6)
    mask_row1   = QHBoxLayout();  mask_row1.setSpacing(6)
    btn_add = QPushButton("＋ Add Instance")
    btn_del = QPushButton("✕ Delete Selected");  btn_del.setObjectName("deleteBtn")
    mask_row1.addWidget(btn_add);  mask_row1.addWidget(btn_del)
    btn_del_unlocked = QPushButton("🗑 Delete All Unlocked")
    btn_del_unlocked.setObjectName("deleteBtn")
    mask_layout.addLayout(mask_row1)
    mask_layout.addWidget(btn_del_unlocked)
    mask_grp.setLayout(mask_layout)

    # ---- Lock group ----
    lock_grp    = QGroupBox("Lock  🔒")
    lock_layout = QHBoxLayout();  lock_layout.setSpacing(6)
    btn_lock     = QPushButton("Lock Selected");   btn_lock.setObjectName("lockBtn")
    btn_unlock   = QPushButton("Unlock")
    btn_unlock_a = QPushButton("Unlock All")
    lock_layout.addWidget(btn_lock)
    lock_layout.addWidget(btn_unlock)
    lock_layout.addWidget(btn_unlock_a)
    lock_grp.setLayout(lock_layout)

    # ---- Merge group ----
    merge_grp    = QGroupBox("Merge")
    merge_layout = QVBoxLayout();  merge_layout.setSpacing(6)
    lbl_merge_status = QLabel("Click 'Start' then pick instances in viewer")
    lbl_merge_status.setObjectName("statusLabel")
    mrg_btn_row  = QHBoxLayout();  mrg_btn_row.setSpacing(6)
    btn_merge_start   = QPushButton("▶ Start Select");  btn_merge_start.setObjectName("mergeStartBtn")
    btn_merge_confirm = QPushButton("✓ Confirm");       btn_merge_confirm.setObjectName("confirmBtn")
    btn_merge_cancel  = QPushButton("✕ Cancel")
    btn_merge_confirm.setEnabled(False)
    btn_merge_cancel.setEnabled(False)
    mrg_btn_row.addWidget(btn_merge_start)
    mrg_btn_row.addWidget(btn_merge_confirm)
    mrg_btn_row.addWidget(btn_merge_cancel)
    merge_layout.addWidget(lbl_merge_status)
    merge_layout.addLayout(mrg_btn_row)
    merge_grp.setLayout(merge_layout)

    # ---- SAM2 group ----
    sam2_grp    = QGroupBox("SAM2 Segmentation")
    sam2_layout = QVBoxLayout();  sam2_layout.setSpacing(6)

    sam2_row1 = QHBoxLayout();  sam2_row1.setSpacing(6)
    sam2_layer_cb = QComboBox()
    sam2_layer_cb.addItems(["HE", "ST_RGB", "Similarity"])
    sam2_layer_cb.setCurrentIndex(0)
    sam2_layer_cb.setToolTip("Layer image to send to SAM2")
    sam2_url_edit = QLineEdit()
    sam2_url_edit.setPlaceholderText("http://server:8000")
    sam2_url_edit.setText("http://localhost:8000")
    sam2_row1.addWidget(sam2_layer_cb, 1)
    sam2_row1.addWidget(sam2_url_edit, 2)

    sam2_row2 = QHBoxLayout();  sam2_row2.setSpacing(6)
    btn_sam2_draw  = QPushButton("✏ Draw Boxes");  btn_sam2_draw.setObjectName("sam2DrawBtn")
    btn_sam2_clear = QPushButton("🗑 Clear")
    btn_sam2_run   = QPushButton("⚡ Run SAM2");    btn_sam2_run.setObjectName("sam2RunBtn")
    sam2_row2.addWidget(btn_sam2_draw)
    sam2_row2.addWidget(btn_sam2_clear)
    sam2_row2.addWidget(btn_sam2_run)

    lbl_sam2_status = QLabel("Draw boxes, then click Run SAM2")
    lbl_sam2_status.setObjectName("sam2StatusLabel")

    sam2_layout.addLayout(sam2_row1)
    sam2_layout.addLayout(sam2_row2)
    sam2_layout.addWidget(lbl_sam2_status)
    sam2_grp.setLayout(sam2_layout)

    # ---- Save group ----
    save_grp    = QGroupBox("Save")
    save_layout = QVBoxLayout();  save_layout.setSpacing(6)
    btn_save = QPushButton("💾  Save Mask  (npy + csv)");  btn_save.setObjectName("saveBtn")
    btn_mark_done = QPushButton("◻  Mark Tile as Done");   btn_mark_done.setObjectName("doneBtn")
    save_layout.addWidget(btn_save)
    save_layout.addWidget(btn_mark_done)
    save_grp.setLayout(save_layout)

    panel_layout.addWidget(sim_grp)
    panel_layout.addWidget(mask_grp)
    panel_layout.addWidget(lock_grp)
    panel_layout.addWidget(merge_grp)
    panel_layout.addWidget(sam2_grp)
    panel_layout.addWidget(save_grp)
    panel_layout.addStretch()

    # ── Qt Data Selector Panel (left) ────────────────────────────────────────

    left_panel = QWidget()
    left_panel.setObjectName("DataSelectorPanel")
    left_layout = QVBoxLayout(left_panel)
    left_layout.setSpacing(10)
    left_layout.setContentsMargins(8, 8, 8, 8)
    left_panel.setStyleSheet(PANEL_STYLE)

    ds_grp    = QGroupBox("Data Selection")
    ds_layout = QVBoxLayout();  ds_layout.setSpacing(6)

    lbl_dataset = QLabel("Dataset folder:")
    dataset_cb  = QComboBox()
    dataset_cb.addItems(_discover_datasets())
    _cur_ds = os.path.basename(os.path.dirname(input_dir))
    _idx = dataset_cb.findText(_cur_ds)
    if _idx >= 0:
        dataset_cb.setCurrentIndex(_idx)

    lbl_prefix = QLabel("Prefix:")
    prefix_cb  = QComboBox()

    btn_load_prefix = QPushButton("Load Selected Prefix")
    btn_load_prefix.setObjectName("loadBtn")

    ds_layout.addWidget(lbl_dataset)
    ds_layout.addWidget(dataset_cb)
    ds_layout.addWidget(lbl_prefix)
    ds_layout.addWidget(prefix_cb)
    ds_layout.addWidget(btn_load_prefix)
    ds_grp.setLayout(ds_layout)

    left_layout.addWidget(ds_grp)
    left_layout.addStretch()

    # ── Action functions ─────────────────────────────────────────────────────

    def _refresh_btn_styles():
        """objectName 变更后强制重新应用样式。"""
        ss = panel.styleSheet()
        panel.setStyleSheet("")
        panel.setStyleSheet(ss)

    def _refresh_prefix_cb():
        """Repopulate prefix_cb with current dataset's prefixes (preserving selection)."""
        done  = state["done_prefixes"]
        prefs = discover_prefixes(state["input_dir"])
        items = [f"{p} ✓" if p in done else p for p in prefs]
        # Preserve current selection
        cur_text  = prefix_cb.currentText()
        cur_clean = cur_text[:-2] if cur_text.endswith(" ✓") else cur_text
        prefix_cb.blockSignals(True)
        prefix_cb.clear()
        prefix_cb.addItems(items)
        for i, item in enumerate(items):
            if (item[:-2] if item.endswith(" ✓") else item) == cur_clean:
                prefix_cb.setCurrentIndex(i)
                break
        prefix_cb.blockSignals(False)

    # -- Fix Similarity --
    def _toggle_fix_sim():
        fixed = not state["sim_fixed"]
        state["sim_fixed"] = fixed
        if fixed:
            btn_fix_sim.setText("🔓 Unfix Similarity")
            btn_fix_sim.setObjectName("fixSimActiveBtn")
        else:
            btn_fix_sim.setText("📌 Fix Similarity Map")
            btn_fix_sim.setObjectName("fixSimBtn")
        _refresh_btn_styles()

    # -- Mask --
    def _add_instance():
        new_id = int(lab.data.max()) + 1
        lab.selected_label = new_id
        print(f"[Mask] New instance id = {new_id}")

    def _delete_instance():
        sel = int(lab.selected_label)
        if sel == 0:
            print("[Mask] Background (0) cannot be deleted.");  return
        if sel in state["locked_ids"]:
            print(f"[Mask] Instance {sel} is LOCKED.");  return
        mask = (lab.data == sel)
        if mask.any():
            lab.data[mask] = 0
            lab.refresh()
            state["prev_labels_snapshot"] = lab.data.copy()
            _update_lock_overlay()
            print(f"[Mask] Deleted id = {sel}")
        else:
            print(f"[Mask] Id {sel} not found.")

    def _delete_unlocked():
        locked = state["locked_ids"]
        data   = lab.data
        all_ids  = set(np.unique(data).tolist()) - {0}
        unlocked = all_ids - locked
        if not unlocked:
            print("[Mask] No unlocked instances to delete.")
            return
        data = data.copy()
        data[np.isin(data, list(unlocked))] = 0
        lab.data = data
        lab.refresh()
        state["prev_labels_snapshot"] = lab.data.copy()
        _update_lock_overlay()
        print(f"[Mask] Deleted {len(unlocked)} unlocked instance(s): {sorted(unlocked)}")

    # -- Lock --
    def _lock_selected():
        sel = int(lab.selected_label)
        if sel == 0:
            print("[Lock] Cannot lock background.");  return
        state["locked_ids"].add(sel)
        # 首次 lock 时确保 snapshot 已建立（保护回调需要它）
        if state["prev_labels_snapshot"] is None:
            state["prev_labels_snapshot"] = lab.data.copy()
        _update_lock_overlay()
        print(f"[Lock] Locked id = {sel}")

    def _unlock_selected():
        sel = int(lab.selected_label)
        if sel in state["locked_ids"]:
            state["locked_ids"].remove(sel)
            _update_lock_overlay()
            print(f"[Lock] Unlocked id = {sel}")
        else:
            print(f"[Lock] Id {sel} not locked.")

    def _unlock_all():
        state["locked_ids"] = set()
        _update_lock_overlay()
        print("[Lock] Unlocked all.")

    # -- Merge state machine --
    def _start_merge():
        state["merge_active"]     = True
        state["merge_candidates"] = set()
        try:  lab.mode = "pan_zoom"
        except Exception: pass
        btn_merge_start.setText("● Selecting...")
        btn_merge_start.setObjectName("mergeActiveBtn")
        btn_merge_start.setEnabled(False)
        btn_merge_confirm.setEnabled(True)
        btn_merge_cancel.setEnabled(True)
        lbl_merge_status.setText("Click instances in viewer to select")
        _refresh_btn_styles()

    def _cancel_merge():
        state["merge_active"]     = False
        state["merge_candidates"] = set()
        btn_merge_start.setText("▶ Start Select")
        btn_merge_start.setObjectName("mergeStartBtn")
        btn_merge_start.setEnabled(True)
        btn_merge_confirm.setEnabled(False)
        btn_merge_cancel.setEnabled(False)
        lbl_merge_status.setText("Click 'Start' then pick instances in viewer")
        _refresh_btn_styles()

    def _confirm_merge():
        ids = sorted(i for i in state["merge_candidates"] if i != 0)
        if len(ids) < 2:
            print("[Merge] Select at least 2 instances.");  return
        locked_hit = [i for i in ids if i in state["locked_ids"]]
        if locked_hit:
            print(f"[Merge] LOCKED ids cannot be merged: {locked_hit}");  return
        tgt  = min(ids)
        data = lab.data
        for i in ids:
            if i != tgt:
                data[data == i] = tgt
        lab.data = data;  lab.refresh()
        print(f"[Merge] Merged {ids} → {tgt}")
        df = lab.metadata.get("instance_table", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            lab.metadata["instance_table"] = merge_instance_table(df, lab.data)
        state["prev_labels_snapshot"] = lab.data.copy()
        _update_lock_overlay()
        _cancel_merge()

    # -- SAM2 --
    def _sam2_toggle_draw():
        drawing = not state["sam2_drawing"]
        state["sam2_drawing"] = drawing
        if drawing:
            sam_boxes.mode = "add_rectangle"
            btn_sam2_draw.setText("◼ Stop Drawing")
            btn_sam2_draw.setObjectName("sam2DrawActiveBtn")
            lbl_sam2_status.setText("Draw rectangles on the viewer")
        else:
            sam_boxes.mode = "pan_zoom"
            btn_sam2_draw.setText("✏ Draw Boxes")
            btn_sam2_draw.setObjectName("sam2DrawBtn")
            n = len(sam_boxes.data)
            lbl_sam2_status.setText(f"{n} box(es) ready" if n else "Draw boxes, then click Run SAM2")
        _refresh_btn_styles()

    def _sam2_clear():
        sam_boxes.data = []
        lbl_sam2_status.setText("Boxes cleared")

    def _sam2_run():
        if state["prefix"] is None:
            lbl_sam2_status.setText("Load a prefix first")
            return
        if len(sam_boxes.data) == 0:
            lbl_sam2_status.setText("Draw at least one box first")
            return

        layer_name = sam2_layer_cb.currentText()
        url        = sam2_url_edit.text().strip()
        if not url:
            lbl_sam2_status.setText("Enter the server URL")
            return

        try:
            img_arr, _, _ = _get_layer_image_and_scale(layer_name)
            boxes_px      = _boxes_to_pixel_coords(layer_name)
        except Exception as exc:
            lbl_sam2_status.setText(f"Error: {exc}")
            return

        btn_sam2_run.setEnabled(False)
        lbl_sam2_status.setText(f"Running SAM2 on {layer_name} ({len(boxes_px)} box(es))…")

        worker = _SAM2Worker(url, img_arr, boxes_px)
        state["sam2_worker"] = worker

        def _on_done(masks_list):
            _apply_sam2_results(masks_list)
            btn_sam2_run.setEnabled(True)
            n = len(masks_list)
            lbl_sam2_status.setText(f"Done — {n} mask{'s' if n != 1 else ''} applied")
            if state["sam2_drawing"]:
                _sam2_toggle_draw()
            state["sam2_worker"] = None

        def _on_error(msg):
            btn_sam2_run.setEnabled(True)
            lbl_sam2_status.setText(f"Error: {msg}")
            print(f"[SAM2] Server error: {msg}")
            state["sam2_worker"] = None

        worker.finished.connect(_on_done)
        worker.error.connect(_on_error)
        worker.start()

    # -- Save --
    def _save_mask():
        if state["edited_mask_path"] is None:
            print("[Save] Load a prefix first.");  return
        np.save(state["edited_mask_path"], lab.data.astype(np.uint32))
        print(f"[Mask] Saved npy → {state['edited_mask_path']}")
        df = lab.metadata.get("instance_table", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(state["edited_csv_path"], index=False)
            print(f"[Mask] Saved csv → {state['edited_csv_path']}")
        else:
            rebuild_instance_table_from_labels(lab.data).to_csv(state["edited_csv_path"], index=False)
            print(f"[Mask] Saved rebuilt csv → {state['edited_csv_path']}")
        import json
        with open(state["locked_ids_path"], "w") as _f:
            json.dump(sorted(state["locked_ids"]), _f)
        print(f"[Lock] Saved {len(state['locked_ids'])} locked ID(s) → {state['locked_ids_path']}")

    # -- Mark Done --
    def _update_done_btn():
        prefix = state["prefix"]
        if prefix and prefix in state["done_prefixes"]:
            btn_mark_done.setText("✓  Done  (click to unmark)")
            btn_mark_done.setObjectName("doneBtnActive")
        else:
            btn_mark_done.setText("◻  Mark Tile as Done")
            btn_mark_done.setObjectName("doneBtn")
        _refresh_btn_styles()

    def _mark_done():
        prefix = state["prefix"]
        if prefix is None:
            print("[Done] Load a prefix first.");  return
        done = state["done_prefixes"]
        if prefix in done:
            done.remove(prefix)
            print(f"[Done] Unmarked {prefix}")
        else:
            done.add(prefix)
            print(f"[Done] Marked {prefix} as complete")
        _save_done_set()
        _update_done_btn()
        _refresh_prefix_cb()

    # -- Dataset / Prefix selectors --
    def _on_dataset_change():
        chosen  = dataset_cb.currentText()
        new_dir = os.path.join(data_root, chosen, size_suffix)
        if not os.path.isdir(new_dir):
            return
        state["input_dir"]    = new_dir
        state["done_prefixes"] = _load_done_set()
        _refresh_prefix_cb()
        print(f"[Dataset] Switched to {chosen}")

    def _load_prefix_btn():
        text = prefix_cb.currentText()
        if not text:
            print("[Load] No prefix selected.");  return
        clean = text[:-2] if text.endswith(" ✓") else text
        load_prefix(clean)

    # ── 连接按钮 ─────────────────────────────────────────────────────────────
    btn_fix_sim.clicked.connect(_toggle_fix_sim)
    btn_add.clicked.connect(_add_instance)
    btn_del.clicked.connect(_delete_instance)
    btn_del_unlocked.clicked.connect(_delete_unlocked)
    btn_lock.clicked.connect(_lock_selected)
    btn_unlock.clicked.connect(_unlock_selected)
    btn_unlock_a.clicked.connect(_unlock_all)
    btn_merge_start.clicked.connect(_start_merge)
    btn_merge_confirm.clicked.connect(_confirm_merge)
    btn_merge_cancel.clicked.connect(_cancel_merge)
    btn_sam2_draw.clicked.connect(_sam2_toggle_draw)
    btn_sam2_clear.clicked.connect(_sam2_clear)
    btn_sam2_run.clicked.connect(_sam2_run)
    btn_save.clicked.connect(_save_mask)
    btn_mark_done.clicked.connect(_mark_done)
    dataset_cb.currentTextChanged.connect(lambda _: _on_dataset_change())
    btn_load_prefix.clicked.connect(_load_prefix_btn)

    v.window.add_dock_widget(panel, area="right", name="Annotation Tools")
    v.window.add_dock_widget(left_panel, area="left", name="Data Selector")

    # ── 键盘快捷键 ────────────────────────────────────────────────────────────
    @v.bind_key("q")
    def _hotkey_lock(viewer):
        """Q — Lock currently selected instance."""
        _lock_selected()

    # ── 鼠标回调 ─────────────────────────────────────────────────────────────

    def mouse_cb(viewer, event):
        if event.type == "mouse_press" and event.button == 1:
            if state["merge_active"]:
                world_yx = event.position[:2]
                r, c     = _world_to_label_rc(world_yx)
                lid      = int(lab.data[r, c])
                if lid == 0:
                    yield;  return
                cands = state["merge_candidates"]
                if lid in cands:
                    cands.discard(lid)
                else:
                    cands.add(lid)
                lbl_merge_status.setText(
                    f"Selected: {sorted(cands)}" if cands else "Click instances to select"
                )
                yield;  return

            # 普通模式：计算相似度（固定时跳过）
            if sam_boxes.mode != "pan_zoom":
                yield;  return
            if lab.mode not in ("pan_zoom", "pick"):
                yield;  return
            if state["emb_norm"] is None:
                yield;  return
            if state["sim_fixed"]:
                yield;  return
            world_yx = event.position[:2]
            r, c = world_to_emb_rc(world_yx, state["sy"], state["sx"], state["emb_h"], state["emb_w"])
            state["last_query"].update({"r": r, "c": c, "world_yx": world_yx})
            _render_similarity(r, c, world_yx)
        yield

    v.mouse_drag_callbacks.append(mouse_cb)

    # ── Lock 保护回调（防重入）──────────────────────────────────────────────

    def on_labels_data_change(event=None):
        if _lg["reverting"]:
            return
        prev   = state["prev_labels_snapshot"]
        cur    = lab.data
        locked = state["locked_ids"]
        if not locked:
            # 没有锁定 ID 时不需要维护 snapshot，跳过全图 copy
            state["prev_labels_snapshot"] = None
            return
        if prev is None or prev.shape != cur.shape:
            state["prev_labels_snapshot"] = cur.copy();  return

        changed = (cur != prev)
        bad = changed & (np.isin(prev, list(locked)) | np.isin(cur, list(locked)))
        if bad.any():
            _lg["reverting"] = True
            try:
                cur2 = cur.copy();  cur2[bad] = prev[bad]
                lab.data = cur2;  lab.refresh()
                cur = lab.data
                print("[Lock] Reverted edits to locked instances.")
            finally:
                _lg["reverting"] = False

        state["prev_labels_snapshot"] = cur.copy()
        _update_lock_overlay()

    try:
        lab.events.data.connect(on_labels_data_change)
    except Exception:
        pass

    # ── 初始化 prefix 列表 & 加载 ─────────────────────────────────────────────
    _refresh_prefix_cb()

    _raw_prefixes = discover_prefixes(state["input_dir"])

    # initial_prefix が discover_prefixes の外にある場合（例: "slide"）も直接ロード可能
    if initial_prefix is not None and initial_prefix not in _raw_prefixes:
        from io_utils import paths_for_prefix
        _p = paths_for_prefix(state["input_dir"], initial_prefix)
        if os.path.exists(_p["rgb_png_path"]) and os.path.exists(_p["pca_path"]):
            load_prefix(initial_prefix)
        else:
            print(f"[Init] initial_prefix '{initial_prefix}' files not found, falling back.")
            if _raw_prefixes:
                load_prefix(_raw_prefixes[0])
    elif _raw_prefixes:
        _raw_default = initial_prefix if initial_prefix in _raw_prefixes else _raw_prefixes[0]
        _display_val = f"{_raw_default} ✓" if _raw_default in state["done_prefixes"] else _raw_default
        _idx2 = prefix_cb.findText(_display_val)
        if _idx2 >= 0:
            prefix_cb.setCurrentIndex(_idx2)
        load_prefix(_raw_default)
    else:
        print(f"[Init] No valid prefixes found in: {state['input_dir']}")

    return v
