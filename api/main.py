"""FastAPI for the on-device summarization project.

Endpoints:
    GET  /health
    POST /summarize    - returns a summary + estimated cost + Pareto flag

Vendor-neutral: production stack runs a quantized SLM via llama.cpp on-device.
The notebook + this API use a deterministic extractive summarizer + analytical
latency model so the harness runs anywhere (incl. Dataiku DSS).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="On-Device SLM Summarization", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "Latency and size are estimates from the device-class model — measure on your "
    "target device before shipping."
)


class SummarizeIn(BaseModel):
    doc: str = Field(..., min_length=1, max_length=20000)
    budget_ms: int = Field(default=100, ge=10, le=2000)
    max_size_mb: int = Field(default=100, ge=10, le=1000)


class SummarizeOut(BaseModel):
    summary: str
    budget_chosen: str
    latency_ms_estimated: float
    model_size_mb_estimated: int
    on_pareto_frontier: bool
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeOut)
def summarize(req: SummarizeIn) -> SummarizeOut:
    try:
        from src.slm_quant.serve import summarize as _summarize  # type: ignore
    except Exception:
        from slm_quant.serve import summarize as _summarize
    out = _summarize(req.doc, budget_ms=req.budget_ms, max_size_mb=req.max_size_mb)
    return SummarizeOut(**out)
