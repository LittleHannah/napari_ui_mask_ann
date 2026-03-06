# Cell Choir — Core Principles

## Definition
A choir is a spatially connected region where the ST foundation model embedding is locally homogeneous.
Boundaries are where the embedding changes rapidly across space.

## Key Principles

1. **Embedding is ground truth, H&E is auxiliary.**
   Choir boundaries are defined by transcriptional jumps in embedding space, not by morphology.

2. **Supra-cellular scale.**
   Choirs are defined at the neighborhood scale (~50–100 µm), not at the single-cell or sub-cellular scale.
   Sub-cellular ST resolution is an asset (richer signal), but choir identity is a regional property.

3. **Includes ECM.**
   ECM within a functional unit is part of that choir. Boundaries are regional, not cell-outline boundaries.

4. **Tissue-agnostic partition.**
   Every tissue is partitioned into choirs — discrete anatomical units (glomerulus, crypt, islet) and
   continuous territories (stroma subtypes, tumor zones) are both valid choirs.
   The difference is boundary sharpness, not choir validity.

5. **Discovery tool.**
   Choir boundaries are learned bottom-up from foundation model embeddings, without prior anatomical
   knowledge. The goal is to discover functionally coherent regions that existing tools cannot see.

## Annotation Principles

**Quality over coverage.**
Label accurate, stable instances. Do not force labels on ambiguous regions.
Unlabeled pixels are treated as "ignore" during training — they are neutral, not wrong.

**Never trace the raw similarity map boundary.**
Use Query Radius ≥ 5 (≈ 40 µm) to smooth sub-cellular noise before deciding where a boundary is.
The smoothed map shows the macroscopic choir territory; the raw map shows sub-cellular noise.

**Internal holes follow the H&E.**
If an internal low-similarity patch corresponds to a lumen or acellular space in H&E → fill it, annotate as continuous.
If it corresponds to a distinct cell type → leave it out or label separately.

**Blood vessels:** annotate large vessels (wall clearly visible in H&E); skip capillaries.

**Three-zone rule for every region:**
```
[choir core — clearly label]  [transition zone — leave unlabeled]  [next choir core — label]
```

## Annotation Order

**Step 1 — Understand the tile first.**
Set Query Radius to 5–10, click representative spots of each tissue type, observe the similarity map.
Identify how many distinct choir types exist in this tile before drawing anything.

**Step 2 — Label in order of boundary clarity.**
1. Discrete anatomical units with sharp boundaries (glomerulus, crypt, islet) — highest quality signal
2. Large continuous territories with visible boundaries (solid tumor core, major stroma zone)
3. Sub-types within a territory if similarity map shows clear internal structure

**Step 3 — Skip difficult tiles.**
If a tile is dominated by a transition/infiltration zone with no clean cores visible → skip it.
Reserve these as hard-case evaluation tiles, not training tiles.

## Tissue-specific Notes

| Tissue | Good choir targets | Skip |
|--------|-------------------|------|
| Kidney | Glomerulus, proximal tubule zone, distal tubule zone | Ambiguous tubule-stroma border |
| Colon | Crypt (single unit), muscularis | Crypt-stroma transition |
| Pancreas | Islet, acinar region, ductal region | Individual acini (too small) |
| Lung cancer | Solid tumor core, lepidic zone, stroma zone | Tumor-stroma infiltration front |
