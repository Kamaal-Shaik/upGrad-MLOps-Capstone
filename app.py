"""app.py — Stage 3: FastAPI inference service.

Implement /health (liveness + loaded model info) and POST /predict (multipart image upload
→ {label, prob_defect, confidence}). Load the model once at startup; log every prediction
to artifacts/predictions.log.   Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io, json, time
from contextlib import asynccontextmanager
from pathlib import Path

import torch, torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

import config
from src import data_prep
from src.model import load_model

_state = {"model": None, "tf": None, "meta": {}}


def _load():
    # TODO 3: if config.MODEL_PATH exists, load model + eval transforms + model_meta.json.
    # raise NotImplementedError
    if config.MODEL_PATH.exists():
        _state["model"] = load_model(config.MODEL_PATH).to(config.DEVICE)
        _state["model"].eval()
        _state["tf"] = data_prep.get_transforms(train=False)

        if config.MODEL_META_PATH.exists():
            _state["meta"] = json.loads(config.MODEL_META_PATH.read_text())
    else:
        _state["model"] = None
        _state["tf"] = None
        _state["meta"] = {}

    return _state


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(title="Casting Defect Detection API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    # TODO 3: return status + model_loaded + classes + positive_class + test_metrics.
    # raise NotImplementedError

    return {
        "status": "ok", "model_loaded": _state["model"] is not None, "classes": config.CLASS_TO_IDX,
        "positive_class": config.POSITIVE_CLASS,
        "test_metrics": _state.get("meta", {}).get("test_metrics", {})
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    # TODO 3: validate model loaded (503 else); read image (400 if invalid); preprocess;
    #         softmax; return {label, is_defective, prob_defect, confidence}; log prediction.
    # raise NotImplementedError

    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Process
    tf = _state["tf"]
    xb = tf(img).unsqueeze(0).to(config.DEVICE)

    # Predict
    logits = _state["model"](xb)
    probs = F.softmax(logits, dim=1)
    prob_defect = float(probs[0, config.POSITIVE_IDX].cpu().item())
    pred_idx = int(torch.argmax(probs, dim=1).cpu().item())
    label = config.IDX_TO_CLASS[pred_idx]

    result = {
        "label": label, "is_defective": bool(pred_idx == config.POSITIVE_IDX),
        "prob_defect": prob_defect, "confidence": float(probs.max().cpu().item())
    }

    # Log prediction
    log_path = config.ARTIFACT_DIR / "predictions.log"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "time": time.time(), "file": file.filename, **result
        }) + "\n")

    return result