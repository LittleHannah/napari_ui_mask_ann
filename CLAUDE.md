# Project: Dinov3_ST napari annotation tool

## What this project does
Interactive napari viewer for spatial transcriptomics data.
- Overlays H&E histology images with DINOv3 PCA embeddings and SAM2 segmentation masks
- Click any pixel → compute cosine similarity across the tissue (via patch embeddings)
- Manually annotate / edit / merge / lock SAM2 instance masks
- Data: Visium HD Human Kidney FFPE, tile-based (`rXXX_cYYY` prefix format)

## File Structure
```
serve.py        — entry point (run this to launch the viewer)
viewer.py       — build_viewer(): all napari layers, Qt UI, callbacks
similarity.py   — pure numpy: world_to_emb_rc(), compute_similarity()
mask_utils.py   — pure pandas/numpy: bbox, instance table rebuild, merge
io_utils.py     — pure IO: image loading, path management, prefix discovery
utils.py        — original monolithic version (kept for reference, do not edit)
data/           — Visium_HD_Human_Kidney_FFPE/maskann_size-512/
test.ipynb      — scratch notebook
```

## Data Layout
```
data/.../maskann_size-512/
  HE/                   {prefix}_he.png              (H&E RGB)
  pca_rgb_no/           {prefix}_rgb.png             (ST RGB overlay)
  pca50/                {prefix}_pca.npy             (emb_h, emb_w, 50) float32
  sam2_merged_masks/    {prefix}_merged_mask.npy     (uint32 label image)
                        {prefix}_merged_mask_info.csv
                        {prefix}_merged_mask_edited.npy   ← written on save
                        {prefix}_merged_mask_info_edited.csv
```

## Key Architecture Notes
- `build_viewer()` uses closures for shared state (no class)
- Coordinate system: HE pixel space = napari world space
- `sim_scale = (he_h / emb_h, he_w / emb_w)` — each embedding patch covers this many HE pixels
- Lock overlay = separate RGBA image layer (yellow); more reliable than `lab.color` API
- Merge overlay = separate RGBA image layer (per-candidate colors)
- Re-entrancy guard `_lg["reverting"]` prevents infinite loop in lock protection callback

## Development Rules
- `similarity.py` and `mask_utils.py` must stay napari-free (importable without napari)
- All file path logic lives in `io_utils.paths_for_prefix()` — single source of truth
- Qt panel style defined in `PANEL_STYLE` constant at top of `viewer.py`
- Do not edit `utils.py`

## Running
```bash
python serve.py
```
Change `input_dir` and `initial_prefix` in `serve.py` to switch datasets.
