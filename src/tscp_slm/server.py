from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .model import TSCPMiniModel
from .schemas import ClassifyRequest


MODEL_PATH = Path("artifacts/tscp_slm_model.json")
app = FastAPI(title="TSCP Small Language Model", version="0.1.0")
RESULTS: dict[str, dict[str, Any]] = {}
MODEL: TSCPMiniModel | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def get_model() -> TSCPMiniModel:
    global MODEL
    if MODEL is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")
        MODEL = TSCPMiniModel.load(MODEL_PATH)
    return MODEL


@app.post("/v1/classify")
def classify(request: ClassifyRequest) -> dict[str, Any]:
    response = get_model().classify_request(request).to_dict()
    RESULTS[response["interaction_id"]] = response
    return response


@app.get("/v1/classify/{interaction_id}")
def get_classification(interaction_id: str) -> dict[str, Any]:
    if interaction_id not in RESULTS:
        raise HTTPException(status_code=404, detail="interaction_id not found")
    return RESULTS[interaction_id]
