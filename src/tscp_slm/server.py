from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import TSCPMiniModel


MODEL_PATH = Path("artifacts/tscp_slm_model.json")
app = FastAPI(title="TSCP Small Language Model", version="0.1.0")


class ClassifyRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=32000)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify")
def classify(request: ClassifyRequest) -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact not found. Train the model first.")
    model = TSCPMiniModel.load(MODEL_PATH)
    return model.predict(request.content).to_dict()
