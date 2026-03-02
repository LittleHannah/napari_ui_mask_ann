"""
viewer.py
---------
napari viewer 构建器。
改动：
  1. Lock 可视化：专用 RGBA overlay 图层（黄色高亮锁定区域）+ 回调防重入守卫
  2. Merge 交互：点击模式选择 → 彩色 overlay 预览 → 确认/取消
  3. UI：单一 Qt 面板，QGroupBox 分组，深色主题样式
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import napari
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from magicgui import magicgui
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel,
)

from io_utils import (
    load_he_rgb, load_rgb_png, normalize_rows,
    discover_prefixes, ensure_edited_mask_exists,
)
from mask_utils import rebuild_instance_table_from_labels, merge_instance_table
from similarity import world_to_emb_rc, compute_similarity


# ── Qt 样式表 ─────────────────────────────────────────────────────────────────
PANEL_STYLE = """
QWidget {
    background-color: #2b2b2b;
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
"""

# 每个 merge 候选 instance 的高亮颜色（循环使用）
_MERGE_COLORS = [
    [0.0, 0.85, 1.0, 1.0],   # cyan
    [1.0, 0.55, 0.0, 1.0],   # orange
    [0.85, 0.0, 1.0, 1.0],   # purple
    [0.0, 1.0, 0.50, 1.0],   # green
    [1.0, 0.20, 0.55, 1.0],  # pink
]


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

    # RGBA overlay：lock（黄色）& merge 候选（彩色）
    lock_overlay = v.add_image(
        np.zeros((10, 10, 4), dtype=np.float32),
        name="LockOverlay", rgb=True, opacity=0.6,
    )
    merge_overlay = v.add_image(
        np.zeros((10, 10, 4), dtype=np.float32),
        name="MergeOverlay", rgb=True, opacity=0.72, visible=False,
    )

    # Query 标记
    query_circle = v.add_points(np.zeros((0, 2), dtype=np.float32), name="QueryCircle", size=80)
    try:    query_circle.face_color = "transparent"
    except Exception: query_circle.face_color = [0, 0, 0, 0]
    try:    query_circle.edge_color = "cyan"; query_circle.edge_width = 4
    except Exception: pass
    query_cross = v.add_shapes([], shape_type="line", name="QueryCross", edge_color="cyan", edge_width=3)

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
        # merge
        "merge_active": False, "merge_candidates": set(),
    }

    params = {
        "sigma": 0.6, "gamma": 0.7,
        "q_low": 0.05, "q_high": 0.995,
        "colormap": "turbo", "auto_show_sim": False,
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
        for lid in locked:
            ov[lab.data == lid] = [1.0, 0.88, 0.0, 1.0]   # bright yellow, full alpha
        lock_overlay.data  = ov
        lock_overlay.scale = lab.scale

    def _update_merge_overlay():
        """用不同颜色 RGBA 高亮每个 merge 候选 instance。"""
        candidates = state["merge_candidates"]
        shape = lab.data.shape[:2]
        if not candidates:
            merge_overlay.data    = np.zeros((*shape, 4), dtype=np.float32)
            merge_overlay.visible = False
            return
        ov = np.zeros((*shape, 4), dtype=np.float32)
        for i, lid in enumerate(sorted(candidates)):
            ov[lab.data == lid] = _MERGE_COLORS[i % len(_MERGE_COLORS)]
        merge_overlay.data    = ov
        merge_overlay.scale   = lab.scale
        merge_overlay.visible = True

    def _render_similarity(r, c, world_yx):
        sim01, lo, hi = compute_similarity(
            state["emb_norm"], r, c, state["emb_h"], state["emb_w"], params,
        )
        sim.data = sim01
        try: sim.contrast_limits = (lo, hi)
        except Exception: pass

        wy, wx = float(world_yx[0]), float(world_yx[1])
        query_circle.data = np.asarray([[wy, wx]], dtype=np.float32)
        L = 70
        query_cross.data = [
            np.array([[wy - L, wx], [wy + L, wx]], dtype=np.float32),
            np.array([[wy, wx - L], [wy, wx + L]], dtype=np.float32),
        ]
        if params["auto_show_sim"]:
            sim.visible = True

    # ── Prefix 加载 ─────────────────────────────────────────────────────────
    def load_prefix(prefix: str):
        p = ensure_edited_mask_exists(input_dir, prefix)

        he_rgb  = load_he_rgb(p["he_path"])
        st_rgb  = load_rgb_png(p["rgb_png_path"])
        emb_pca = np.load(p["pca_path"])

        he_h, he_w   = he_rgb.shape[:2]
        st_h, st_w   = st_rgb.shape[:2]
        emb_h, emb_w, emb_d = emb_pca.shape
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

        he.data = he_rgb
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
            locked_ids=set(),
            merge_active=False, merge_candidates=set(),
        ))

        emb = emb_pca.astype(np.float32, copy=False).reshape(-1, emb_d)
        state["emb_norm"] = normalize_rows(emb)

        query_circle.data = np.zeros((0, 2), dtype=np.float32)
        query_cross.data  = []
        state["prev_labels_snapshot"] = lab.data.copy()

        _update_lock_overlay()
        _update_merge_overlay()
        print(f"[Load] {prefix} | HE={he_rgb.shape} ST={st_rgb.shape} EMB={emb_pca.shape} MASK={label_img.shape}")

    # ── Similarity controls widget (magicgui) ───────────────────────────────
    @magicgui(
        sigma={"min": 0.0, "max": 3.0, "step": 0.05},
        gamma={"min": 0.1, "max": 3.0, "step": 0.05},
        q_low={"min": 0.0, "max": 0.5, "step": 0.005},
        q_high={"min": 0.5, "max": 1.0, "step": 0.005},
        colormap={"choices": CMAPS},
        auto_show_sim={"widget_type": "CheckBox"},
    )
    def controls(
        sigma: float = 0.6, gamma: float = 0.7,
        q_low: float = 0.05, q_high: float = 0.995,
        colormap: str = "turbo", auto_show_sim: bool = False,
    ):
        params.update(dict(
            sigma=float(sigma), gamma=float(gamma),
            q_low=float(q_low), q_high=float(q_high),
            colormap=str(colormap), auto_show_sim=bool(auto_show_sim),
        ))
        _set_colormap_safe(params["colormap"])
        if state["last_query"]["r"] is not None:
            _render_similarity(
                state["last_query"]["r"], state["last_query"]["c"], state["last_query"]["world_yx"],
            )

    v.window.add_dock_widget(controls, area="right", name="Similarity Controls")

    # ── Qt Annotation Panel ──────────────────────────────────────────────────
    # 先创建所有 Qt 控件（action functions 需要引用它们）

    panel = QWidget()
    panel.setObjectName("AnnotationPanel")
    panel_layout = QVBoxLayout(panel)
    panel_layout.setSpacing(10)
    panel_layout.setContentsMargins(8, 8, 8, 8)
    panel.setStyleSheet(PANEL_STYLE)

    # ---- Mask group ----
    mask_grp    = QGroupBox("Mask")
    mask_layout = QHBoxLayout();  mask_layout.setSpacing(6)
    btn_add = QPushButton("＋ Add Instance")
    btn_del = QPushButton("✕ Delete");  btn_del.setObjectName("deleteBtn")
    mask_layout.addWidget(btn_add);  mask_layout.addWidget(btn_del)
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

    # ---- Save group ----
    save_grp    = QGroupBox("Save")
    save_layout = QHBoxLayout()
    btn_save = QPushButton("💾  Save Mask  (npy + csv)");  btn_save.setObjectName("saveBtn")
    save_layout.addWidget(btn_save)
    save_grp.setLayout(save_layout)

    panel_layout.addWidget(mask_grp)
    panel_layout.addWidget(lock_grp)
    panel_layout.addWidget(merge_grp)
    panel_layout.addWidget(save_grp)
    panel_layout.addStretch()

    # ── Action functions（引用上面的 Qt 控件）──────────────────────────────

    def _refresh_btn_styles():
        """objectName 变更后强制重新应用样式。"""
        ss = panel.styleSheet()
        panel.setStyleSheet("")
        panel.setStyleSheet(ss)

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

    # -- Lock --
    def _lock_selected():
        sel = int(lab.selected_label)
        if sel == 0:
            print("[Lock] Cannot lock background.");  return
        state["locked_ids"].add(sel)
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
        state["merge_active"]    = True
        state["merge_candidates"] = set()
        _update_merge_overlay()
        try:  lab.mode = "pan_zoom"   # 切换到 pan 防止误刷 label
        except Exception: pass
        btn_merge_start.setText("● Selecting...")
        btn_merge_start.setObjectName("mergeActiveBtn")
        btn_merge_start.setEnabled(False)
        btn_merge_confirm.setEnabled(True)
        btn_merge_cancel.setEnabled(True)
        lbl_merge_status.setText("Click instances in viewer to select")
        _refresh_btn_styles()

    def _cancel_merge():
        state["merge_active"]    = False
        state["merge_candidates"] = set()
        _update_merge_overlay()
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

    # ── 连接按钮 ─────────────────────────────────────────────────────────────
    btn_add.clicked.connect(_add_instance)
    btn_del.clicked.connect(_delete_instance)
    btn_lock.clicked.connect(_lock_selected)
    btn_unlock.clicked.connect(_unlock_selected)
    btn_unlock_a.clicked.connect(_unlock_all)
    btn_merge_start.clicked.connect(_start_merge)
    btn_merge_confirm.clicked.connect(_confirm_merge)
    btn_merge_cancel.clicked.connect(_cancel_merge)
    btn_save.clicked.connect(_save_mask)

    v.window.add_dock_widget(panel, area="right", name="Annotation Tools")

    # ── 鼠标回调（引用 Qt 控件，必须在控件创建后定义）────────────────────────

    def mouse_cb(viewer, event):
        if event.type == "mouse_press" and event.button == 1:
            if state["merge_active"]:
                # Merge 选择模式：点击 → 切换候选集合
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
                _update_merge_overlay()
                lbl_merge_status.setText(
                    f"Selected: {sorted(cands)}" if cands else "Click instances to select"
                )
                yield;  return

            # 普通模式：计算相似度
            if state["emb_norm"] is None:
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
        if prev is None or prev.shape != cur.shape:
            state["prev_labels_snapshot"] = cur.copy();  return
        locked = state["locked_ids"]
        if not locked:
            state["prev_labels_snapshot"] = cur.copy();  return

        changed     = (cur != prev)
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

    # ── Data selector (magicgui) ────────────────────────────────────────────
    prefixes = discover_prefixes(input_dir)
    default_prefix = (
        initial_prefix if (initial_prefix in prefixes)
        else (prefixes[0] if prefixes else None)
    )

    @magicgui(
        call_button="Load Selected Prefix",
        prefix={"choices": lambda w: discover_prefixes(input_dir)},
    )
    def selector(prefix: Optional[str] = default_prefix):
        if not prefix:
            print("[Load] No prefix selected.");  return
        load_prefix(prefix)

    v.window.add_dock_widget(selector, area="right", name="Data Selector")

    # ── 初始加载 ────────────────────────────────────────────────────────────
    prefixes = discover_prefixes(input_dir)
    if initial_prefix is None:
        initial_prefix = prefixes[0] if prefixes else None
    if initial_prefix is not None:
        load_prefix(initial_prefix)
    else:
        print(f"[Init] No valid prefixes found in: {input_dir}")

    return v
