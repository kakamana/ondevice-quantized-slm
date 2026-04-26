"""Inference helper used by the FastAPI layer."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

from .features import BUDGETS, estimate_latency, estimate_size_mb
from .models import textrank_summarize


MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


@lru_cache(maxsize=1)
def _load_pareto() -> list[dict]:
    p = MODEL_DIR / "summarizer.pkl"
    if not p.exists():
        # Default approximate Pareto so the API is useful before training is run
        return [
            {"budget": "tight",  "rouge_l_mean": 0.22, "latency_p95":  35.0, "size_mb":  35, "on_pareto_frontier": True},
            {"budget": "small",  "rouge_l_mean": 0.28, "latency_p95":  60.0, "size_mb":  60, "on_pareto_frontier": True},
            {"budget": "medium", "rouge_l_mean": 0.34, "latency_p95":  95.0, "size_mb": 110, "on_pareto_frontier": True},
            {"budget": "large",  "rouge_l_mean": 0.36, "latency_p95": 180.0, "size_mb": 220, "on_pareto_frontier": False},
        ]
    return joblib.load(p)["pareto"]


def _select_budget(budget_ms: int, max_size_mb: int) -> str:
    """Greedy: smallest tier that fits both budgets; else fall back to tight."""
    candidates = []
    for name, b in BUDGETS.items():
        # use mid-document latency estimate to pick
        approx = b.alpha_ms + b.beta_ms_per_token * 200
        if approx <= budget_ms and b.size_mb <= max_size_mb:
            candidates.append((name, b))
    if not candidates:
        return "tight"
    # pick the largest that fits — best quality
    candidates.sort(key=lambda x: x[1].size_mb)
    return candidates[-1][0]


def summarize(doc: str, budget_ms: int = 100, max_size_mb: int = 100) -> dict:
    chosen = _select_budget(budget_ms, max_size_mb)
    summary = textrank_summarize(doc, budget=chosen)
    n_tokens = len(doc.split())
    latency = estimate_latency(n_tokens, budget=chosen, noise_sd_pct=0.0)
    size_mb = estimate_size_mb(chosen)
    pareto = _load_pareto()
    on_pareto = next((r["on_pareto_frontier"] for r in pareto if r["budget"] == chosen), True)
    return dict(
        summary=summary,
        budget_chosen=chosen,
        latency_ms_estimated=float(latency),
        model_size_mb_estimated=int(size_mb),
        on_pareto_frontier=bool(on_pareto),
    )
