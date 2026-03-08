from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from src.inference.cityseg_infer import (
    InferConfig,
    load_keras_model,
    make_overlay,
    predict_mask,
    preprocess_image,
)
from src.inference.palette import colorize_mask

MODEL_PATH = Path("models/unet_baseline_best.keras")

app = FastAPI(title="CitySeg API", version="1.0.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = load_keras_model(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/model")
def health_model():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True, "model_path": str(MODEL_PATH)}
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "model_loaded": False,
                "detail": str(exc),
            },
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Query(default="overlay", pattern="^(overlay|mask)$"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img_rgb = np.array(img, dtype=np.uint8)

        cfg = InferConfig()
        x, img_resized = preprocess_image(img_rgb, cfg.target_hw)

        model = get_model()
        mask = predict_mask(model, x)

        if mode == "mask":
            out = colorize_mask(mask)
        else:
            out = make_overlay(
                img_resized,
                mask,
                alpha=cfg.alpha,
                void_transparent=cfg.void_transparent,
            )

        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=prediction.png"},
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc