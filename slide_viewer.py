"""
slide_viewer.py
---------------
Slide-level napari viewer：输入合并后的 slide mask + 拼合的 ST_RGB + 降采样 embeddings。

无 HE 图层（tile HE 太大），无 Data Selector（单张 slide）。
功能：Click 相似度查询 / Lock / Merge / Save。
"""

import json
import os
import re
import glob

import numpy as np
import pandas as pd
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QComboBox,
)
from napari.utils.colormaps import AVAILABLE_COLORMAPS

from io_utils import load_rgb_png, normalize_rows
from mask_utils import rebuild_instance_table_from_labels, merge_instance_table
from similarity import world_to_emb_rc, compute_similarity, region_query_vec


# ── 样式 ──────────────────────────────────────────────────────────────────────
PANEL_STYLE = """
QWidget { background-color: #2b2b2b; }
QLabel  { color: #cccccc; font-size: 12px; }
QGroupBox {
    color: #cccccc; font-weight: bold; font-size: 12px;
    border: 1px solid #4a4a4a; border-radius: 5px;
    margin-top: 12px; padding: 10px 6px 6px 6px;
}
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; padding: 0 4px; }
QPushButton {
    background-color: #3c3f41; color: #cccccc;
    border: 1px solid #555555; border-radius: 4px;
    padding: 5px 10px; font-size: 12px; min-height: 24px;
}
QPushButton:hover    { background-color: #4c5052; border-color: #777; }
QPushButton:pressed  { background-color: #2a2a2a; }
QPushButton:disabled { background-color: #282828; color: #555; border-color: #383838; }
QPushButton#lockBtn   { color: #ffdd44; border-color: #7a6a1a; }
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
QLabel#statusLabel { color: #888888; font-size: 11px; padding: 2px 4px; }
QLabel#mergeLabel  { color: #66aaff; font-size: 11px; padding: 2px 4px; font-style: italic; }
QComboBox { background-color: #3c3f41; color: #cccccc; border: 1px solid #555555; border-radius: 4px; padding: 3px 6px; }
QComboBox::drop-down { border: none; }
"""


# ── 数据组装 ──────────────────────────────────────────────────────────────────

def _discover_tiles(mask_dir: str) -> list[dict]:
    """找出所有 edited tile mask，返回 [{prefix, row, col, mask_path}]。"""
    files = sorted(glob.glob(os.path.join(mask_dir, "*_merged_mask_edited.npy")))
    tiles = []
    for f in files:
        m = re.search(r"r(\d+)_c(\d+)", os.path.basename(f))
        if m:
            tiles.append(dict(prefix=f"r{m.group(1)}_c{m.group(2)}",
                               row=int(m.group(1)), col=int(m.group(2)), mask_path=f))
    return tiles


def build_global_emb_canvas(pca_path: str, coords_path: str,
                             canvas_H: int, canvas_W: int,
                             tile_row_range: tuple[int, int] = None,
                             tile_col_range: tuple[int, int] = None,
                             downsample: int = 4) -> np.ndarray:
    """
    用全局 PCA embedding + 坐标文件构建 slide-level embedding canvas。

    pca_path    : (N, D) float32 — whole-slide PCA 向量
    coords_path : (N, 2) float32 — 每个向量对应的 array 坐标
                  coords[:,0] = col（x 方向）
                  coords[:,1] = row（y 方向）
                  坐标系与 tile 文件名 rXXX_cYYY 完全一致，直接使用无需归一化。
    返回 (canvas_H//ds, canvas_W//ds, D) float32。
    """
    print("[GlobalEmb] Loading PCA and coords (may take a moment)...")
    pca    = np.load(pca_path).astype(np.float32)    # (N, D)
    coords = np.load(coords_path).astype(np.float32) # (N, 2)
    N, D   = pca.shape

    # coords 已经是图像 array 坐标，直接映射：row=coords[:,0], col=coords[:,1]
    mask_rows = np.clip(coords[:, 0].astype(np.int32), 0, canvas_H - 1)
    mask_cols = np.clip(coords[:, 1].astype(np.int32), 0, canvas_W - 1)

    emb_H = canvas_H // downsample
    emb_W = canvas_W // downsample
    emb_rows = (mask_rows // downsample).clip(0, emb_H - 1)
    emb_cols = (mask_cols // downsample).clip(0, emb_W - 1)
    flat_idx = (emb_rows * emb_W + emb_cols).astype(np.int32)

    print(f"[GlobalEmb] Scatter {N:,} spots → ({emb_H}×{emb_W}) canvas (D={D})...")
    n_pixels = emb_H * emb_W
    cnt      = np.bincount(flat_idx, minlength=n_pixels).astype(np.float32)

    emb_flat = np.zeros((n_pixels, D), dtype=np.float32)
    for d in range(D):
        emb_flat[:, d] = np.bincount(flat_idx, weights=pca[:, d], minlength=n_pixels)

    nonzero = cnt > 0
    emb_flat[nonzero] /= cnt[nonzero, np.newaxis]

    coverage = nonzero.sum()
    print(f"[GlobalEmb] Done — coverage {coverage:,}/{n_pixels:,} pixels ({100*coverage/n_pixels:.1f}%)")
    return emb_flat.reshape(emb_H, emb_W, D)


def assemble_st_rgb(input_dir: str, tiles: list[dict], canvas_H: int, canvas_W: int,
                    tile_size: int = 512) -> np.ndarray:
    """把所有 tile 的 ST_RGB PNG 拼成 canvas_H×canvas_W×3 uint8 图像。"""
    canvas = np.zeros((canvas_H, canvas_W, 3), dtype=np.uint8)
    rgb_dir = os.path.join(input_dir, "pca_rgb_no")
    for t in tiles:
        rgb_path = os.path.join(rgb_dir, f"{t['prefix']}_rgb.png")
        if not os.path.exists(rgb_path):
            continue
        img = load_rgb_png(rgb_path)
        r0, c0 = t["row"], t["col"]
        canvas[r0 : r0 + tile_size, c0 : c0 + tile_size] = img[:tile_size, :tile_size]
    return canvas


def assemble_embeddings(input_dir: str, tiles: list[dict],
                        canvas_H: int, canvas_W: int,
                        tile_size: int = 512,
                        downsample: int = 4) -> np.ndarray:
    """
    把所有 tile 的 PCA embedding 拼成 (canvas_H//ds, canvas_W//ds, D) float32。

    每个 tile 在空间上按 stride=downsample 采样（降采样）。

    PCA 符号对齐：PCA 特征向量方向在各 tile 间可能任意翻转（sign ambiguity）。
    以第一个 tile 的全局均值向量为参考，若某 tile 的均值向量与参考向量点积为负，
    则将该 tile 的所有 embedding 乘以 -1，强制方向对齐。
    """
    emb_H = canvas_H // downsample
    emb_W = canvas_W // downsample
    step  = tile_size // downsample

    # 找到实际存放 tile embedding 的目录和后缀
    prefix0 = tiles[0]["prefix"]
    emb_dir, emb_suffix = None, None
    for d, suf in [("emb", "_emb.npy"), ("pca50", "_pca.npy"),
                   ("emb", "_pca.npy"), ("pca50", "_emb.npy")]:
        candidate = os.path.join(input_dir, d, f"{prefix0}{suf}")
        if os.path.exists(candidate):
            emb_dir, emb_suffix = os.path.join(input_dir, d), suf
            break
    if emb_dir is None:
        raise FileNotFoundError(f"Cannot find embedding files for prefix {prefix0} under {input_dir}")

    sample_path = os.path.join(emb_dir, f"{prefix0}{emb_suffix}")
    D = np.load(sample_path).shape[2]

    canvas   = np.zeros((emb_H, emb_W, D), dtype=np.float32)
    ref_vec  = None   # 参考方向（第一个成功加载的 tile 的均值向量）

    for t in tiles:
        emb_path = os.path.join(emb_dir, f"{t['prefix']}{emb_suffix}")
        if not os.path.exists(emb_path):
            continue

        emb_tile = np.load(emb_path).astype(np.float32)   # (tile_size, tile_size, D)

        # ── PCA 符号对齐 ──────────────────────────────────────────────────
        tile_mean = emb_tile.reshape(-1, D).mean(axis=0)   # (D,)
        norm = np.linalg.norm(tile_mean)
        if norm > 1e-8:
            tile_mean_n = tile_mean / norm
            if ref_vec is None:
                ref_vec = tile_mean_n
            elif float(ref_vec @ tile_mean_n) < 0:
                emb_tile = -emb_tile   # 翻转该 tile 所有 embedding
                print(f"[Emb] Sign-flipped tile {t['prefix']}")

        # spatial downsample
        sub = emb_tile[::downsample, ::downsample]
        r0e = t["row"] // downsample
        c0e = t["col"] // downsample
        canvas[r0e : r0e + step, c0e : c0e + step] = sub[:step, :step]

    return canvas


# ── Viewer ────────────────────────────────────────────────────────────────────

def build_slide_viewer(mask_dir: str, emb_downsample: int = 4):
    """
    构建 slide-level napari viewer。

    mask_dir    : .../maskann_size-512/sam2_merged_masks
    emb_downsample : embedding 空间降采样倍数（内存节省；4 → ~128MB emb）
    """
    TILE_SIZE  = 512
    SLIDE_MASK = os.path.join(mask_dir, "slide_merged_mask.npy")
    SLIDE_CSV  = os.path.join(mask_dir, "slide_merged_mask_info.csv")
    LOCKED_IDS = os.path.join(mask_dir, "slide_locked_ids.json")

    input_dir = os.path.dirname(mask_dir)   # .../maskann_size-512

    # ── 组装数据 ──────────────────────────────────────────────────────────
    print("[Slide] Discovering tiles...")
    tiles    = _discover_tiles(mask_dir)
    all_rows = sorted(set(t["row"] for t in tiles))
    all_cols = sorted(set(t["col"] for t in tiles))
    canvas_H = max(all_rows) + TILE_SIZE
    canvas_W = max(all_cols) + TILE_SIZE
    print(f"[Slide] {len(tiles)} tiles, canvas {canvas_H}×{canvas_W}")

    print("[Slide] Assembling ST_RGB...")
    st_rgb = assemble_st_rgb(input_dir, tiles, canvas_H, canvas_W, TILE_SIZE)
    print(f"[Slide] ST_RGB: {st_rgb.shape}, {st_rgb.nbytes/1e6:.0f} MB")

    print(f"[Slide] Assembling embeddings (downsample={emb_downsample})...")
    emb_canvas = assemble_embeddings(input_dir, tiles, canvas_H, canvas_W,
                                     TILE_SIZE, emb_downsample)
    emb_H, emb_W, emb_D = emb_canvas.shape
    print(f"[Slide] Emb: {emb_canvas.shape}, {emb_canvas.nbytes/1e6:.0f} MB")

    print("[Slide] Normalizing embeddings...")
    emb_norm = normalize_rows(emb_canvas.reshape(-1, emb_D))

    print("[Slide] Loading mask...")
    slide_mask = np.load(SLIDE_MASK).astype(np.uint32)
    print(f"[Slide] Mask: {slide_mask.shape}, instances: {slide_mask.max()}")

    # sim_scale: each emb pixel covers `emb_downsample` world pixels
    ds = float(emb_downsample)
    sim_scale = (ds, ds)

    # ── napari viewer ─────────────────────────────────────────────────────
    v = napari.Viewer(title="Slide Viewer")

    st  = v.add_image(st_rgb, name="ST_RGB", rgb=True, opacity=0.9)
    sim = v.add_image(
        np.zeros((emb_H, emb_W), dtype=np.float32),
        name="Similarity", colormap="turbo", opacity=0.75,
        contrast_limits=(0.0, 1.0), scale=sim_scale,
    )
    lab = v.add_labels(slide_mask, name="Mask", opacity=0.5)
    lock_overlay = v.add_image(
        np.zeros((canvas_H, canvas_W, 4), dtype=np.float32),
        name="LockOverlay", rgb=True, opacity=0.6,
    )
    query_cross = v.add_shapes(
        [], shape_type="line", name="QueryCross", edge_color="cyan", edge_width=3,
    )

    # ── 状态 ──────────────────────────────────────────────────────────────
    state = {
        "locked_ids": set(),
        "prev_labels_snapshot": slide_mask.copy(),
        "merge_active": False,
        "merge_candidates": set(),
        "sim_fixed": False,
    }

    params = {
        "sigma": 1.5, "gamma": 0.7,
        "q_low": 0.05, "q_high": 0.995,
        "colormap": "turbo",
        "query_radius": 6,
        "auto_show_sim": False,
    }

    # 从文件恢复 locked IDs
    if os.path.exists(LOCKED_IDS):
        with open(LOCKED_IDS) as f:
            state["locked_ids"] = set(json.load(f))
        print(f"[Lock] Restored {len(state['locked_ids'])} locked ID(s)")

    _lg = {"reverting": False}   # 防重入守卫

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _update_lock_overlay():
        ov = np.zeros((canvas_H, canvas_W, 4), dtype=np.float32)
        for lid in state["locked_ids"]:
            ov[lab.data == lid] = [1.0, 0.88, 0.0, 1.0]
        lock_overlay.data = ov

    def _world_to_label_rc(world_yx) -> tuple[int, int]:
        wy, wx = float(world_yx[0]), float(world_yx[1])
        h, w   = lab.data.shape[:2]
        r = int(np.clip(np.floor(wy), 0, h - 1))
        c = int(np.clip(np.floor(wx), 0, w - 1))
        return r, c

    def _render_similarity(r, c, world_yx):
        radius = int(params["query_radius"])
        qvec = None
        if radius > 0:
            qvec = region_query_vec(emb_norm, r, c, radius, emb_H, emb_W)
        sim01, lo, hi = compute_similarity(emb_norm, r, c, emb_H, emb_W, params, query_vec=qvec)
        sim.data = sim01
        try: sim.contrast_limits = (lo, hi)
        except Exception: pass
        wy, wx = float(world_yx[0]), float(world_yx[1])
        L = 120
        query_cross.data = [
            np.array([[wy - L, wx], [wy + L, wx]], dtype=np.float32),
            np.array([[wy, wx - L], [wy, wx + L]], dtype=np.float32),
        ]
        if params["auto_show_sim"]:
            sim.visible = True

    # ── Callbacks ─────────────────────────────────────────────────────────

    @v.mouse_drag_callbacks.append
    def on_click(viewer, event):
        if event.type != "mouse_press" or event.button != 1:
            return
        world_yx = event.position[:2]
        r, c = world_to_emb_rc(world_yx, sy=ds, sx=ds, emb_h=emb_H, emb_w=emb_W)

        if state["merge_active"]:
            lr, lc = _world_to_label_rc(world_yx)
            clicked_id = int(lab.data[lr, lc])
            if clicked_id == 0:
                return
            state["merge_candidates"].add(clicked_id)
            _update_merge_label()
            return

        if not state["sim_fixed"]:
            _render_similarity(r, c, world_yx)

    def on_labels_change(event):
        if _lg["reverting"]:
            return
        locked = state["locked_ids"]
        if not locked:
            state["prev_labels_snapshot"] = lab.data.copy()
            return
        cur  = lab.data
        prev = state["prev_labels_snapshot"]
        changed = np.where(cur != prev)
        if changed[0].size == 0:
            state["prev_labels_snapshot"] = lab.data.copy()
            return
        prev_locked = np.isin(prev[changed], list(locked))
        cur_locked  = np.isin(cur[changed],  list(locked))
        if (prev_locked | cur_locked).any():
            _lg["reverting"] = True
            lab.data = prev.copy()
            lab.refresh()
            _lg["reverting"] = False
            print("[Lock] Reverted: attempt to modify locked instance(s)")
            return
        state["prev_labels_snapshot"] = lab.data.copy()

    lab.events.data.connect(on_labels_change)

    # ── Qt Panel ──────────────────────────────────────────────────────────

    panel = QWidget(); panel.setStyleSheet(PANEL_STYLE)
    layout = QVBoxLayout(panel); layout.setSpacing(6)

    # ── Similarity Controls ──
    sim_box = QGroupBox("Similarity Controls"); sim_layout = QVBoxLayout(sim_box)

    fix_sim_btn = QPushButton("Fix Similarity (off)")
    fix_sim_btn.setObjectName("fixSimBtn")
    def toggle_fix_sim():
        state["sim_fixed"] = not state["sim_fixed"]
        fix_sim_btn.setText("Fix Similarity (on)" if state["sim_fixed"] else "Fix Similarity (off)")
        fix_sim_btn.setObjectName("fixSimActiveBtn" if state["sim_fixed"] else "fixSimBtn")
        fix_sim_btn.setStyleSheet("")
    fix_sim_btn.clicked.connect(toggle_fix_sim)

    cmap_combo = QComboBox()
    cmap_combo.addItems(sorted(AVAILABLE_COLORMAPS.keys()))
    cmap_combo.setCurrentText("turbo")
    def on_cmap(text):
        params["colormap"] = text
        try: sim.colormap = text
        except Exception: pass
    cmap_combo.currentTextChanged.connect(on_cmap)

    sim_layout.addWidget(QLabel("Colormap:")); sim_layout.addWidget(cmap_combo)
    sim_layout.addWidget(fix_sim_btn)
    layout.addWidget(sim_box)

    # ── Annotation Tools ──
    ann_box = QGroupBox("Annotation Tools"); ann_layout = QVBoxLayout(ann_box)

    # Mask section
    mask_group = QGroupBox("Mask"); mask_layout = QVBoxLayout(mask_group)
    del_btn         = QPushButton("✕ Delete Selected");     del_btn.setObjectName("deleteBtn")
    del_all_btn     = QPushButton("🗑 Delete All Unlocked"); del_all_btn.setObjectName("deleteBtn")

    def delete_selected():
        sel = int(lab.selected_label)
        if sel == 0:
            print("[Mask] Background cannot be deleted."); return
        if sel in state["locked_ids"]:
            print(f"[Mask] Instance {sel} is locked."); return
        data = lab.data.copy()
        if not (data == sel).any():
            print(f"[Mask] Id {sel} not found."); return
        data[data == sel] = 0
        lab.data = data; lab.refresh()
        state["prev_labels_snapshot"] = lab.data.copy()
        _update_lock_overlay()
        print(f"[Mask] Deleted id={sel}")

    def delete_all_unlocked():
        locked   = state["locked_ids"]
        all_ids  = set(np.unique(lab.data).tolist()) - {0}
        unlocked = all_ids - locked
        if not unlocked:
            print("[Mask] No unlocked instances to delete."); return
        data = lab.data.copy()
        data[np.isin(data, list(unlocked))] = 0
        lab.data = data; lab.refresh()
        state["prev_labels_snapshot"] = lab.data.copy()
        _update_lock_overlay()
        print(f"[Mask] Deleted {len(unlocked)} unlocked instance(s)")

    del_btn.clicked.connect(delete_selected)
    del_all_btn.clicked.connect(delete_all_unlocked)
    mask_layout.addWidget(del_btn); mask_layout.addWidget(del_all_btn)
    ann_layout.addWidget(mask_group)

    # Lock section
    lock_group = QGroupBox("Lock"); lock_layout = QVBoxLayout(lock_group)
    lock_btn   = QPushButton("Lock Selected Instance"); lock_btn.setObjectName("lockBtn")
    unlock_btn = QPushButton("Unlock All")
    lock_status = QLabel("Locked: 0"); lock_status.setObjectName("statusLabel")

    def _refresh_lock_status():
        lock_status.setText(f"Locked: {len(state['locked_ids'])}")

    def lock_selected():
        selected = list(lab.selected_label)
        for sid in selected:
            if sid != 0:
                state["locked_ids"].add(sid)
        _update_lock_overlay()
        _refresh_lock_status()
        state["prev_labels_snapshot"] = lab.data.copy()
        print(f"[Lock] Locked: {selected}")

    def unlock_all():
        state["locked_ids"].clear()
        _update_lock_overlay()
        _refresh_lock_status()
        state["prev_labels_snapshot"] = lab.data.copy()
        print("[Lock] Unlocked all")

    lock_btn.clicked.connect(lock_selected)
    unlock_btn.clicked.connect(unlock_all)
    lock_layout.addWidget(lock_btn); lock_layout.addWidget(unlock_btn); lock_layout.addWidget(lock_status)
    ann_layout.addWidget(lock_group)

    # Merge section
    merge_group = QGroupBox("Merge"); merge_layout = QVBoxLayout(merge_group)
    merge_start_btn  = QPushButton("Start Select"); merge_start_btn.setObjectName("mergeStartBtn")
    merge_confirm_btn = QPushButton("Confirm Merge"); merge_confirm_btn.setObjectName("confirmBtn")
    merge_cancel_btn  = QPushButton("Cancel")
    merge_label = QLabel(""); merge_label.setObjectName("mergeLabel")
    merge_confirm_btn.setEnabled(False); merge_cancel_btn.setEnabled(False)

    def _update_merge_label():
        n = len(state["merge_candidates"])
        merge_label.setText(f"Selected: {n} instance(s)" if n > 0 else "")

    def start_merge():
        state["merge_active"] = True
        state["merge_candidates"].clear()
        lab.mode = "pan_zoom"
        merge_start_btn.setObjectName("mergeActiveBtn"); merge_start_btn.setStyleSheet("")
        merge_confirm_btn.setEnabled(True); merge_cancel_btn.setEnabled(True)
        merge_start_btn.setText("Selecting…")
        _update_merge_label()

    def cancel_merge():
        state["merge_active"] = False; state["merge_candidates"].clear()
        merge_start_btn.setObjectName("mergeStartBtn"); merge_start_btn.setStyleSheet("")
        merge_start_btn.setText("Start Select")
        merge_confirm_btn.setEnabled(False); merge_cancel_btn.setEnabled(False)
        merge_label.setText("")

    def confirm_merge():
        cands = state["merge_candidates"]
        if len(cands) < 2:
            cancel_merge(); return
        locked = state["locked_ids"]
        if cands & locked:
            print("[Merge] Aborted: candidate overlaps locked instance"); cancel_merge(); return
        keep_id = min(cands)
        data = lab.data.copy()
        for cid in cands:
            if cid != keep_id:
                data[data == cid] = keep_id
        lab.data = data; lab.refresh()
        state["prev_labels_snapshot"] = lab.data.copy()
        print(f"[Merge] Merged {sorted(cands)} → {keep_id}")
        cancel_merge()

    merge_start_btn.clicked.connect(start_merge)
    merge_confirm_btn.clicked.connect(confirm_merge)
    merge_cancel_btn.clicked.connect(cancel_merge)
    for w in [merge_start_btn, merge_confirm_btn, merge_cancel_btn, merge_label]:
        merge_layout.addWidget(w)
    ann_layout.addWidget(merge_group)

    # Save section
    save_group  = QGroupBox("Save"); save_layout = QVBoxLayout(save_group)
    save_btn    = QPushButton("Save Slide Mask"); save_btn.setObjectName("saveBtn")
    save_status = QLabel(""); save_status.setObjectName("statusLabel")

    def save_slide():
        save_status.setText("Saving…")
        try:
            mask_data = lab.data.astype(np.uint32)
            np.save(SLIDE_MASK, mask_data)

            # rebuild CSV
            df = rebuild_instance_table_from_labels(mask_data)
            df.to_csv(SLIDE_CSV, index=False)

            # save locked IDs
            with open(LOCKED_IDS, "w") as f:
                json.dump(sorted(state["locked_ids"]), f)

            state["prev_labels_snapshot"] = mask_data.copy()
            n_inst = int(mask_data.max())
            save_status.setText(f"Saved — {n_inst} instances")
            print(f"[Save] {SLIDE_MASK}  ({n_inst} instances)")
        except Exception as e:
            save_status.setText(f"Error: {e}")
            print(f"[Save] Error: {e}")

    save_btn.clicked.connect(save_slide)
    save_layout.addWidget(save_btn); save_layout.addWidget(save_status)
    ann_layout.addWidget(save_group)

    layout.addWidget(ann_box)
    layout.addStretch()

    v.window.add_dock_widget(panel, area="right", name="Slide Controls")
    _refresh_lock_status()

    print("[Slide] Viewer ready.")
    return v
