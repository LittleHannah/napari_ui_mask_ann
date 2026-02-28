import os
import glob
import shutil
import re
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from typing import Optional
import napari
from napari.utils.colormaps import AVAILABLE_COLORMAPS
from magicgui import magicgui


# -----------------------------
# IO utils
# -----------------------------
def _load_rgb_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _load_he_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def _parse_prefix_from_filename(fname: str) -> str | None:
    # try to find rXXX_cYYY inside filename
    m = re.search(r"(r\d+_c\d+)", os.path.basename(fname))
    return m.group(1) if m else None


def discover_prefixes(input_dir: str) -> list[str]:
    """
    Return prefixes rXXX_cYYY that have at least HE + PCA + ST_RGB available.
    """
    pca_dir = os.path.join(input_dir, "pca50")
    he_dir = os.path.join(input_dir, "HE")
    rgb_dir = os.path.join(input_dir, "pca_rgb_no")

    pca_files = sorted(glob.glob(os.path.join(pca_dir, "r*_c*_pca.npy")))
    prefixes = []
    for p in pca_files:
        prefix = _parse_prefix_from_filename(p)
        if prefix is None:
            continue
        he_path = os.path.join(he_dir, f"{prefix}_he.png")
        rgb_path = os.path.join(rgb_dir, f"{prefix}_rgb.png")
        if os.path.exists(he_path) and os.path.exists(rgb_path):
            prefixes.append(prefix)
    return prefixes


def paths_for_prefix(input_dir: str, prefix: str) -> dict[str, str]:
    """
    Central place to define how files are named for a given prefix.
    """
    he_path = os.path.join(input_dir, "HE", f"{prefix}_he.png")
    rgb_png_path = os.path.join(input_dir, "pca_rgb_no", f"{prefix}_rgb.png")
    pca_path = os.path.join(input_dir, "pca50", f"{prefix}_pca.npy")

    # masks
    mask_dir = os.path.join(input_dir, "sam2_merged_masks")
    mask_path = os.path.join(mask_dir, f"{prefix}_merged_mask.npy")
    mask_label_csv = os.path.join(mask_dir, f"{prefix}_merged_mask_info.csv")

    # edited copies
    edited_mask_path = mask_path.replace(".npy", "_edited.npy")
    edited_csv_path = mask_label_csv.replace(".csv", "_edited.csv")

    return dict(
        he_path=he_path,
        rgb_png_path=rgb_png_path,
        pca_path=pca_path,
        mask_path=mask_path,
        mask_label_csv=mask_label_csv,
        edited_mask_path=edited_mask_path,
        edited_csv_path=edited_csv_path,
    )


# -----------------------------
# Instance table updater
# -----------------------------
def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return float(y0), float(x0), float(y1 - y0), float(x1 - x0)


def rebuild_instance_table_from_labels(label_img: np.ndarray, keep_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Minimal rebuild: id, area, bbox_y,bbox_x,bbox_h,bbox_w
    (你原表里 predicted_iou/stability_score 这种我们无法从编辑后恢复，所以这里仅重建基础几列)
    """
    ids = np.unique(label_img)
    ids = ids[ids != 0]
    rows = []
    for i in ids:
        m = (label_img == i)
        area = int(m.sum())
        bb = _bbox_from_mask(m)
        if bb is None:
            continue
        by, bx, bh, bw = bb
        rows.append(dict(id=int(i), area=area, bbox_y=by, bbox_x=bx, bbox_h=bh, bbox_w=bw))
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    if keep_cols is not None:
        # keep only requested cols if exist
        cols = [c for c in keep_cols if c in df.columns]
        df = df[cols]
    return df


# -----------------------------
# Viewer builder (with state)
# -----------------------------
def build_viewer(input_dir: str, initial_prefix: str | None = None):
    v = napari.Viewer()

    # layers placeholders
    he = v.add_image(np.zeros((10, 10, 3), dtype=np.uint8), name="HE", rgb=True)
    st = v.add_image(np.zeros((10, 10, 3), dtype=np.uint8), name="ST_RGB", rgb=True, opacity=0.7)
    sim = v.add_image(
        np.zeros((10, 10), dtype=np.float32),
        name="Similarity",
        colormap="turbo",
        opacity=0.75,
        contrast_limits=(0.0, 1.0),
    )
    lab = v.add_labels(np.zeros((10, 10), dtype=np.uint32), name="Mask", opacity=0.5)

    # Query marker layers
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
        pass

    query_cross = v.add_shapes(
        [],
        shape_type="line",
        name="QueryCross",
        edge_color="cyan",
        edge_width=3,
    )

    # ----------------------------
    # state
    # ----------------------------
    state = {
        "input_dir": input_dir,
        "prefix": None,

        "he_h": None, "he_w": None,
        "emb_h": None, "emb_w": None, "emb_d": None,
        "st_h": None, "st_w": None,

        "st_scale": (1.0, 1.0),
        "sim_scale": (1.0, 1.0),
        "sy": 1.0,
        "sx": 1.0,

        "emb_norm": None,  # (H*W, D)
        "last_query": {"r": None, "c": None, "world_yx": None},

        # lock
        "locked_ids": set(),
        "locked_color_rgba": (1.0, 1.0, 0.0, 1.0),  # yellow
        "prev_labels_snapshot": None,

        # paths
        "edited_mask_path": None,
        "edited_csv_path": None,
        "raw_csv_path": None,
    }

    params = {
        "sigma": 0.6,
        "gamma": 0.7,
        "q_low": 0.05,
        "q_high": 0.995,
        "colormap": "turbo",
        "auto_show_sim": False,
    }
    CMAPS = sorted(AVAILABLE_COLORMAPS.keys())

    def set_colormap_safe(cmap_name: str):
        try:
            sim.colormap = cmap_name
        except Exception:
            sim.colormap = "viridis"

    def world_to_emb_rc(world_yx):
        wy, wx = float(world_yx[0]), float(world_yx[1])
        r = int(np.clip(np.floor(wy / state["sy"]), 0, state["emb_h"] - 1))
        c = int(np.clip(np.floor(wx / state["sx"]), 0, state["emb_w"] - 1))
        return r, c

    def compute_and_render_similarity(r, c, world_yx):
        emb_norm = state["emb_norm"]
        emb_h, emb_w = state["emb_h"], state["emb_w"]

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

    def apply_locked_colors():
        """
        Force locked ids to a fixed color so you can visually tell they're finalized.
        """
        locked = state["locked_ids"]
        if not locked:
            # leave default
            return
        # switch to direct color mapping
        try:
            lab.color_mode = "direct"
        except Exception:
            pass

        # build mapping for locked ids only
        color_map = {}
        rgba = state["locked_color_rgba"]
        for i in locked:
            color_map[int(i)] = rgba
        try:
            lab.color = color_map
        except Exception:
            # if napari version doesn't like partial mapping, ignore
            pass

    def ensure_edited_mask_exists(prefix: str):
        p = paths_for_prefix(input_dir, prefix)
        # create edited mask if missing
        if not os.path.exists(p["edited_mask_path"]):
            if os.path.exists(p["mask_path"]):
                shutil.copy(p["mask_path"], p["edited_mask_path"])
            else:
                # no raw mask: create blank later after HE load
                pass
        return p

    def load_prefix(prefix: str):
        p = ensure_edited_mask_exists(prefix)

        he_rgb = _load_he_rgb(p["he_path"])
        st_rgb = _load_rgb_png(p["rgb_png_path"])
        emb_pca = np.load(p["pca_path"])

        he_h, he_w = he_rgb.shape[:2]
        st_h, st_w = st_rgb.shape[:2]
        emb_h, emb_w, emb_d = emb_pca.shape

        # scales (HE as world)
        st_scale = (he_h / st_h, he_w / st_w)
        sim_scale = (he_h / emb_h, he_w / emb_w)

        # load mask: prefer edited; fallback raw; else blank
        label_img = None
        if os.path.exists(p["edited_mask_path"]):
            label_img = np.load(p["edited_mask_path"]).astype(np.uint32, copy=False)
        elif os.path.exists(p["mask_path"]):
            label_img = np.load(p["mask_path"]).astype(np.uint32, copy=False)

        if label_img is None:
            label_img = np.zeros((emb_h, emb_w), dtype=np.uint32)

        # if label_img not matching, allow (HE) or (EMB) shapes
        if label_img.shape == (he_h, he_w):
            lab.scale = (1.0, 1.0)
        elif label_img.shape == (emb_h, emb_w):
            lab.scale = sim_scale
        else:
            raise ValueError(
                f"mask shape {label_img.shape} not match HE {(he_h, he_w)} or EMB {(emb_h, emb_w)}"
            )

        # update layers
        he.data = he_rgb
        st.data = st_rgb
        st.scale = st_scale

        # similarity layer resize/reset
        sim.data = np.zeros((emb_h, emb_w), dtype=np.float32)
        sim.scale = sim_scale
        set_colormap_safe(params["colormap"])
        try:
            sim.contrast_limits = (0.0, 1.0)
        except Exception:
            pass

        lab.data = label_img
        lab.refresh()

        # instance table
        df = None
        if os.path.exists(p["edited_csv_path"]):
            df = pd.read_csv(p["edited_csv_path"])
        elif os.path.exists(p["mask_label_csv"]):
            df = pd.read_csv(p["mask_label_csv"])
        if df is not None:
            lab.metadata["instance_table"] = df
            state["raw_csv_path"] = p["mask_label_csv"]
        else:
            lab.metadata.pop("instance_table", None)
            state["raw_csv_path"] = p["mask_label_csv"]

        # update state
        state["prefix"] = prefix
        state["he_h"], state["he_w"] = he_h, he_w
        state["st_h"], state["st_w"] = st_h, st_w
        state["emb_h"], state["emb_w"], state["emb_d"] = emb_h, emb_w, emb_d
        state["st_scale"] = st_scale
        state["sim_scale"] = sim_scale
        state["sy"], state["sx"] = sim_scale
        state["edited_mask_path"] = p["edited_mask_path"]
        state["edited_csv_path"] = p["edited_csv_path"]

        # reset lock state per-prefix (你也可以改成跨 prefix 继承)
        state["locked_ids"] = set()
        apply_locked_colors()

        # precompute emb_norm
        emb = emb_pca.astype(np.float32, copy=False).reshape(-1, emb_d)
        state["emb_norm"] = _normalize_rows(emb)

        # clear query marks
        query_circle.data = np.zeros((0, 2), dtype=np.float32)
        query_cross.data = []

        # snapshot for lock revert
        state["prev_labels_snapshot"] = lab.data.copy()

        print(f"[Load] {prefix} | HE={he_rgb.shape} ST={st_rgb.shape} EMB={emb_pca.shape} MASK={label_img.shape}")

    # ----------------------------
    # Similarity controls
    # ----------------------------
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

        if state["last_query"]["r"] is not None:
            compute_and_render_similarity(
                state["last_query"]["r"],
                state["last_query"]["c"],
                state["last_query"]["world_yx"],
            )

    v.window.add_dock_widget(controls, area="right", name="Similarity Controls")

    # ----------------------------
    # Mouse callback for similarity query
    # ----------------------------
    def mouse_cb(viewer, event):
        if event.type == "mouse_press" and event.button == 1:
            if state["emb_norm"] is None:
                return
            world_yx = event.position[:2]
            r, c = world_to_emb_rc(world_yx)
            state["last_query"]["r"], state["last_query"]["c"], state["last_query"]["world_yx"] = r, c, world_yx
            compute_and_render_similarity(r, c, world_yx)
        yield

    v.mouse_drag_callbacks.append(mouse_cb)

    # ----------------------------
    # Lock enforcement: revert edits touching locked ids
    # ----------------------------
    def on_labels_data_change(event=None):
        prev = state["prev_labels_snapshot"]
        cur = lab.data
        if prev is None or prev.shape != cur.shape:
            state["prev_labels_snapshot"] = cur.copy()
            return

        locked = state["locked_ids"]
        if not locked:
            state["prev_labels_snapshot"] = cur.copy()
            return

        changed = (cur != prev)
        if not changed.any():
            return

        # rule A: pixels that WERE locked cannot change
        prev_locked = np.isin(prev, list(locked))
        bad_a = changed & prev_locked

        # rule B: pixels cannot be changed INTO a locked id (avoid painting locked id elsewhere)
        cur_locked = np.isin(cur, list(locked))
        bad_b = changed & cur_locked

        bad = bad_a | bad_b
        if bad.any():
            # revert only bad pixels
            cur2 = cur.copy()
            cur2[bad] = prev[bad]
            lab.data = cur2
            lab.refresh()
            cur = lab.data  # update after revert

        state["prev_labels_snapshot"] = cur.copy()

    # listen to changes
    try:
        lab.events.data.connect(on_labels_data_change)
    except Exception:
        # older napari versions might not have events.data
        pass

    # ----------------------------
    # Mask tools
    # ----------------------------
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
        if sel in state["locked_ids"]:
            print(f"[Mask] Instance id {sel} is LOCKED, cannot delete.")
            return

        mask = (lab.data == sel)
        if mask.any():
            lab.data[mask] = 0
            lab.refresh()
            print(f"[Mask] Deleted instance id = {sel}")
        else:
            print(f"[Mask] Instance id {sel} not found.")

    @magicgui(call_button="Lock Selected")
    def lock_selected():
        sel = int(lab.selected_label)
        if sel == 0:
            print("[Lock] Cannot lock background (0).")
            return
        state["locked_ids"].add(sel)
        apply_locked_colors()
        print(f"[Lock] Locked instance id = {sel}")

    @magicgui(call_button="Unlock Selected")
    def unlock_selected():
        sel = int(lab.selected_label)
        if sel in state["locked_ids"]:
            state["locked_ids"].remove(sel)
            apply_locked_colors()
            print(f"[Lock] Unlocked instance id = {sel}")
        else:
            print(f"[Lock] Instance id {sel} not locked.")

    @magicgui(call_button="Unlock All")
    def unlock_all():
        state["locked_ids"] = set()
        # clear direct colors (napari will fall back to default)
        try:
            lab.color = {}
        except Exception:
            pass
        print("[Lock] Unlocked all instances.")

    @magicgui(
        call_button="Merge",
        ids_to_merge={"label": "ids_to_merge (e.g. 12,15,20)"},
        target_id={"label": "target_id (0=auto)"},
    )
    def merge_instances(ids_to_merge: str = "", target_id: int = 0):
        s = ids_to_merge.strip()
        if not s:
            print("[Merge] ids_to_merge is empty.")
            return
        ids = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            ids.append(int(part))
        ids = sorted(set(ids))
        ids = [i for i in ids if i != 0]
        if len(ids) < 2:
            print("[Merge] Need >=2 ids.")
            return

        # locked check
        locked_hit = [i for i in ids if i in state["locked_ids"]]
        if locked_hit:
            print(f"[Merge] These ids are LOCKED, cannot merge: {locked_hit}")
            return

        if target_id is None or int(target_id) == 0:
            tgt = min(ids)
        else:
            tgt = int(target_id)
            if tgt in state["locked_ids"]:
                print(f"[Merge] target_id {tgt} is LOCKED, cannot use as target.")
                return

        # merge in label map
        data = lab.data
        for i in ids:
            if i == tgt:
                continue
            data[data == i] = tgt
        lab.data = data
        lab.refresh()
        print(f"[Merge] Merged {ids} -> {tgt}")

        # update instance table (best-effort)
        df = lab.metadata.get("instance_table", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            keep_cols = list(df.columns)
            new_df = rebuild_instance_table_from_labels(lab.data, keep_cols=None)
            # try to preserve original columns if possible
            # we will left-join onto new_df by id to keep extra cols where available
            new_df2 = new_df.merge(df, on="id", how="left", suffixes=("", "_old"))
            # keep preference: new basic cols + existing extras
            # if original had area/bbox columns, use newly computed
            for col in ["area", "bbox_y", "bbox_x", "bbox_h", "bbox_w"]:
                if col in new_df.columns:
                    new_df2[col] = new_df[col].values
            # reorder to original columns if possible, else use merged
            cols = [c for c in keep_cols if c in new_df2.columns]
            if cols:
                new_df2 = new_df2[cols]
            lab.metadata["instance_table"] = new_df2
            print("[Merge] instance_table updated (best-effort).")

    @magicgui(call_button="Save Mask (npy + csv)")
    def save_mask():
        if state["edited_mask_path"] is None:
            print("[Save] No edited_mask_path in state (did you load a prefix?).")
            return
        np.save(state["edited_mask_path"], lab.data.astype(np.uint32))
        print(f"[Mask] Saved npy -> {state['edited_mask_path']}")

        # also save edited csv (rebuild minimal if needed)
        df = lab.metadata.get("instance_table", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(state["edited_csv_path"], index=False)
            print(f"[Mask] Saved csv -> {state['edited_csv_path']}")
        else:
            df2 = rebuild_instance_table_from_labels(lab.data)
            df2.to_csv(state["edited_csv_path"], index=False)
            print(f"[Mask] Saved rebuilt csv -> {state['edited_csv_path']}")

    v.window.add_dock_widget(add_instance, area="right", name="Mask Tools")
    v.window.add_dock_widget(delete_instance, area="right")
    v.window.add_dock_widget(lock_selected, area="right")
    v.window.add_dock_widget(unlock_selected, area="right")
    v.window.add_dock_widget(unlock_all, area="right")
    v.window.add_dock_widget(merge_instances, area="right")
    v.window.add_dock_widget(save_mask, area="right")

    prefixes = discover_prefixes(input_dir)
    default_prefix = initial_prefix if (initial_prefix in prefixes) else (prefixes[0] if prefixes else None)
    # ----------------------------
    # Prefix selector widget
    # ----------------------------
    @magicgui(
        call_button="Load Selected Prefix",
        prefix={"choices": lambda w: discover_prefixes(input_dir)},
    )
    def selector(prefix: Optional[str] = default_prefix):
        if prefix is None or prefix == "":
            print("[Load] No prefix selected.")
            return
        load_prefix(prefix)

    v.window.add_dock_widget(selector, area="right", name="Data Selector")

    # ----------------------------
    # Load initial prefix
    # ----------------------------
    prefixes = discover_prefixes(input_dir)
    if initial_prefix is None:
        initial_prefix = prefixes[0] if prefixes else None
    if initial_prefix is not None:
        load_prefix(initial_prefix)
    else:
        print(f"[Init] No valid prefixes found in: {input_dir}")

    return v


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    input_dir = "/Users/xiaohanzhao/Projects/Dinov3_ST/napari/data/Visium_HD_Human_Kidney_FFPE/maskann_size-512"
    # 你也可以手动指定初始 prefix，比如 "r0_c512"
    initial_prefix = "r0_c512"

    v = build_viewer(input_dir=input_dir, initial_prefix=initial_prefix)
    napari.run()