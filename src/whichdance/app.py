"""FastAPI inference service.

Loads the trained model once at startup and exposes it over HTTP. Meant to
run as an internal service — the WordPress plugin (see wordpress-plugin/)
proxies to it server-side, so it does not need to be exposed publicly.

Usage:
    uvicorn whichdance.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from whichdance import config
from whichdance.predict import Prediction, load_model, predict_with_model

app = FastAPI(title="whichdance", description="Predict traditional folk dance from a tune")

# Permissive by default since this is meant to sit behind a same-origin
# proxy (the WordPress plugin). Tighten allow_origins if this service is
# ever exposed directly to browsers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

_model = None
_label_encoder = None
_device = None


class PredictionResponse(BaseModel):
    filename: str
    fingerprint: str | None
    guessed_dances: list[dict]
    bpm: float
    duration_seconds: float

    @classmethod
    def from_prediction(cls, p: Prediction) -> "PredictionResponse":
        return cls(**p.to_dict())


@app.on_event("startup")
def _load_model() -> None:
    global _model, _label_encoder, _device
    checkpoint_path = config.CHECKPOINT_DIR / "best.pt"
    if not checkpoint_path.exists():
        # Don't crash the process — /health reports this, /predict returns
        # a clear 503 rather than the service failing to start entirely.
        return
    _model, _label_encoder, _device = load_model(str(config.CHECKPOINT_DIR))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "num_classes": len(_label_encoder) if _label_encoder is not None else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...), top_k: int = 3) -> PredictionResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available yet (checkpoints/best.pt missing). "
            "Run the training pipeline first.",
        )

    suffix = Path(file.filename or "").suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        size = 0
        while chunk := await file.read(1 << 16):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File too large")
            tmp.write(chunk)
        tmp.flush()

        try:
            result = predict_with_model(tmp.name, _model, _label_encoder, _device, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - surface as a clean 400, not a 500 traceback
            raise HTTPException(status_code=400, detail=f"Could not analyze audio: {exc}") from exc

        # predict_with_model saw the temp file's random name; report the
        # original uploaded filename instead, which is what the caller
        # actually cares about.
        result.filename = file.filename or result.filename

    return PredictionResponse.from_prediction(result)


_static_dir = config.REPO_ROOT / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
