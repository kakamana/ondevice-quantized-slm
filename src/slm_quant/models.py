"""Extractive (TextRank-style) summarizer + Pareto evaluator.

Run as:
    python -m slm_quant.models
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from .data import PROCESSED, write_all
from .features import (
    BUDGETS,
    Budget,
    estimate_latency,
    estimate_size_mb,
    rouge_l,
    sentence_similarity,
    split_sentences,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def textrank_summarize(text: str, budget: str = "medium") -> str:
    """Pick top-k sentences by PageRank; truncate to budget token cap."""
    sents = split_sentences(text)
    if not sents:
        return ""
    b: Budget = BUDGETS[budget]
    if len(sents) <= b.max_sentences:
        out = sents
    else:
        sim = sentence_similarity(sents)
        # Avoid self-loops dominating
        np.fill_diagonal(sim, 0.0)
        graph = nx.from_numpy_array(sim)
        try:
            scores = nx.pagerank(graph, alpha=0.85, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            scores = {i: 1.0 for i in range(len(sents))}
        ranked = sorted(scores, key=scores.get, reverse=True)
        keep = sorted(ranked[: b.max_sentences])  # preserve original order
        out = [sents[i] for i in keep]
    summary = " ".join(out)
    toks = summary.split()
    if len(toks) > b.max_tokens:
        summary = " ".join(toks[: b.max_tokens])
    return summary


def evaluate_budget(test: pd.DataFrame, budget: str, n_eval: int = 1000, seed: int = 0) -> dict:
    sub = test.sample(n=min(n_eval, len(test)), random_state=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in sub.iterrows():
        pred = textrank_summarize(r["doc_text"], budget=budget)
        rows.append(dict(
            doc_id=r["doc_id"],
            domain=r["domain"],
            rouge_l=rouge_l(pred, r["ref_summary"]),
            latency_ms=estimate_latency(int(r["doc_tokens"]), budget=budget, rng=rng),
        ))
    df = pd.DataFrame(rows)
    return dict(
        budget=budget,
        size_mb=estimate_size_mb(budget),
        n=len(df),
        rouge_l_mean=float(df["rouge_l"].mean()),
        rouge_l_news=float(df[df["domain"]=="news"]["rouge_l"].mean()) if (df["domain"]=="news").any() else float("nan"),
        rouge_l_technical=float(df[df["domain"]=="technical"]["rouge_l"].mean()) if (df["domain"]=="technical").any() else float("nan"),
        rouge_l_narrative=float(df[df["domain"]=="narrative"]["rouge_l"].mean()) if (df["domain"]=="narrative").any() else float("nan"),
        latency_p50=float(df["latency_ms"].median()),
        latency_p95=float(df["latency_ms"].quantile(0.95)),
    )


def pareto_frontier(rows: Iterable[dict]) -> list[dict]:
    """Mark each (latency, rouge) point as on the Pareto frontier (min latency, max rouge)."""
    rows = list(rows)
    rows.sort(key=lambda r: r["latency_p95"])
    best_rouge = -1.0
    for r in rows:
        on = r["rouge_l_mean"] > best_rouge
        r["on_pareto_frontier"] = bool(on)
        if on:
            best_rouge = r["rouge_l_mean"]
    return rows


def run_full_evaluation(n_eval: int = 1000) -> dict:
    counts = write_all()
    test = pd.read_parquet(PROCESSED / "test.parquet")
    rows = [evaluate_budget(test, b, n_eval=n_eval) for b in BUDGETS.keys()]
    rows = pareto_frontier(rows)
    summary = pd.DataFrame(rows)
    summary.to_parquet(PROCESSED / "pareto.parquet", index=False)
    summary.to_csv(PROCESSED / "pareto.csv", index=False)
    # Persist the budget table too for the API to consume cheaply
    joblib.dump({"budgets": BUDGETS, "pareto": rows}, MODEL_DIR / "summarizer.pkl")
    return dict(counts=counts, pareto=rows)


if __name__ == "__main__":
    out = run_full_evaluation(n_eval=500)
    for r in out["pareto"]:
        print(
            f"{r['budget']:>6}  R-L={r['rouge_l_mean']:.3f}  "
            f"p95={r['latency_p95']:.1f}ms  size={r['size_mb']}MB  "
            f"{'PARETO' if r['on_pareto_frontier'] else ''}"
        )
