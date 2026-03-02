"""
sam2_server.py
--------------
FastAPI server wrapping SAM2 for remote interactive segmentation.
Deploy this on the GPU machine and run:
    python sam2_server.py [--host 0.0.0.0] [--port 8000]
                          [--config sam2_hiera_large.yaml]
                          [--checkpoint checkpoints/sam2_hiera_large.pt]

The napari client (viewer.py) sends POST /segment requests with a base64-encoded
image and a list of box prompts, and receives binary masks back.
"""

import argparse
import base64
import io
import sys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from pydantic import BaseModel
sys.path.insert(0, "/home/hmaixxz/condo/sam2")
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ── Request / Response models ─────────────────────────────────────────────────

class SegmentRequest(BaseModel):
    image_b64: str           # base64-encoded PNG (H×W×3 uint8)
    boxes: list[list[float]] # [[x1, y1, x2, y2], ...] in pixel coords


class SegmentResponse(BaseModel):
    masks: list[str]   # base64-encoded uint8 flat arrays (0/1), one per box
    shape: list[int]   # [H, W] of the returned masks


# ── Global predictor (loaded at startup) ─────────────────────────────────────

app = FastAPI(title="SAM2 Segmentation Server")
_predictor: SAM2ImagePredictor | None = None


@app.on_event("startup")
def _load_model():
    global _predictor
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM2 Server] Loading model on {device} ...")
    print(f"  config:     {args.config}")
    print(f"  checkpoint: {args.checkpoint}")
    sam2_model = build_sam2(args.config, args.checkpoint, device=device)
    _predictor = SAM2ImagePredictor(sam2_model)
    print("[SAM2 Server] Model ready.")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    if _predictor is None:
        return JSONResponse(status_code=503, content={"detail": "Model not loaded"})

    # 1. Decode image
    img_bytes = base64.b64decode(req.image_b64)
    pil_img   = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np    = np.array(pil_img, dtype=np.uint8)   # H×W×3

    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return JSONResponse(status_code=400, content={"detail": "Expected H×W×3 RGB image"})

    H, W = img_np.shape[:2]

    # 2. Set image in predictor (once per request)
    with torch.inference_mode():
        _predictor.set_image(img_np)

        masks_out = []
        for box in req.boxes:
            if len(box) != 4:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Each box must be [x1,y1,x2,y2], got {box}"},
                )
            box_arr = np.array(box, dtype=np.float32)
            # SAM2 predict returns (num_masks, H, W) booleans; multimask_output=False → 1 mask
            masks, _, _ = _predictor.predict(
                box=box_arr,
                multimask_output=False,
            )
            mask_hw = masks[0].astype(np.uint8)   # bool → 0/1 uint8, shape H×W
            # encode as base64
            mask_b64 = base64.b64encode(mask_hw.flatten().tobytes()).decode("ascii")
            masks_out.append(mask_b64)

    return SegmentResponse(masks=masks_out, shape=[H, W])


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Segmentation Server")
    parser.add_argument("--host",       default="0.0.0.0",                  help="Bind host")
    parser.add_argument("--port",       default=8000,       type=int,        help="Bind port")
    parser.add_argument("--config",     default="configs/sam2.1/sam2.1_hiera_l.yaml",     help="SAM2 config name (relative to sam2 package)")
    parser.add_argument("--checkpoint", default="/condo/wanglab/hmaixxz/sam2/checkpoints/sam2.1_hiera_large.pt", help="SAM2 checkpoint path")
    # parse_known_args so uvicorn's own args don't conflict
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run("sam2_server:app", host=args.host, port=args.port, reload=False)
