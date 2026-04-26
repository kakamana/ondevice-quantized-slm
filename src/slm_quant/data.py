"""Synthetic short-document corpus generator for the on-device summarization project.

Outputs (under data/processed/):
    docs.parquet   10,000 short docs + reference summaries
    test.parquet   2,000 held-out test docs

Run as a module:
    python -m slm_quant.data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED = DATA_DIR / "processed"

DOMAINS = ["news", "technical", "narrative"]
DOMAIN_COUNTS = {"news": 4000, "technical": 3000, "narrative": 3000}

# Vocabulary per domain.
SUBJECTS = {
    "news": [
        "the central bank", "the regulator", "the council", "the consortium",
        "Dubai's Roads Authority", "the airport operator", "the labour ministry",
        "a Federal court", "the energy regulator",
    ],
    "technical": [
        "the new release", "the deployment", "the kafka cluster", "the data warehouse",
        "the inference service", "the postgres replica", "the training pipeline",
        "the schema migration", "the autoscaler",
    ],
    "narrative": [
        "Mariam", "Khalid", "the old engineer", "the new intern",
        "the team lead", "the night-shift technician", "the visiting consultant",
    ],
}
ACTIONS = {
    "news": [
        "announced a revised policy", "introduced new compliance requirements",
        "warned of seasonal demand spikes", "approved a major infrastructure project",
        "issued updated safety guidance", "released the quarterly review",
    ],
    "technical": [
        "rolled out an experimental feature flag", "moved to a new storage backend",
        "introduced an asynchronous queue", "switched to columnar parquet writes",
        "added retries with exponential backoff", "swapped a sync call for streaming",
    ],
    "narrative": [
        "noticed something unusual on the morning report",
        "stayed late to triage the alert",
        "called a quick standup to review the incident",
        "wrote a calm post-mortem",
        "shared a small win with the team",
    ],
}
DETAILS = {
    "news": [
        "The change takes effect next quarter and applies to all licensed entities.",
        "Officials cited the need for greater transparency and consumer protection.",
        "Industry groups have requested a transition period to update their systems.",
        "Implementation guidance will be published on the official portal.",
    ],
    "technical": [
        "Throughput improved by roughly 18% in the staging benchmark.",
        "Tail latency dropped from 320ms to 110ms at p95.",
        "The migration ran with zero downtime thanks to a careful cutover.",
        "Error budget consumption fell back below the SLO threshold.",
    ],
    "narrative": [
        "The room was quiet except for the soft hum of the air conditioner.",
        "She made another coffee, opened the laptop, and started reading the logs.",
        "He thanked the team and reminded them that incidents are how we learn.",
        "Outside, the city was waking up; inside, the dashboards were turning green.",
    ],
}
CLOSERS = {
    "news": [
        "The story remains developing.",
        "A follow-up briefing is expected next week.",
        "Further details will be shared in the official communication.",
    ],
    "technical": [
        "The runbook has been updated with the new procedure.",
        "Monitoring dashboards now expose the new metrics by default.",
        "A short knowledge-share session is scheduled for Friday.",
    ],
    "narrative": [
        "Tomorrow, the routine begins again.",
        "She closed the laptop and smiled.",
        "It was a small thing, but it mattered.",
    ],
}


def _make_doc(domain: str, rng: np.random.Generator) -> tuple[str, str]:
    """Return (doc_text, ref_summary) — ref is the first + last sentence."""
    subj = rng.choice(SUBJECTS[domain])
    act = rng.choice(ACTIONS[domain])
    s1 = f"{subj.capitalize()} {act}."
    n_details = int(rng.integers(2, 6))
    body = " ".join(rng.choice(DETAILS[domain], size=n_details, replace=True).tolist())
    closer = rng.choice(CLOSERS[domain])
    doc = f"{s1} {body} {closer}"
    summary = f"{s1} {closer}"
    return doc, summary


def make_docs(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    did = 0
    for domain, n in DOMAIN_COUNTS.items():
        for _ in range(n):
            doc, summary = _make_doc(domain, rng)
            rows.append(dict(
                doc_id=f"D{did:06d}",
                domain=domain,
                doc_text=doc,
                ref_summary=summary,
                doc_tokens=len(doc.split()),
                summary_tokens=len(summary.split()),
            ))
            did += 1
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def carve_test(df: pd.DataFrame, n_per_domain: dict | None = None, seed: int = 43) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_per_domain = n_per_domain or {"news": 1000, "technical": 500, "narrative": 500}
    rng = np.random.default_rng(seed)
    test_idx = []
    for domain, n in n_per_domain.items():
        idx = df.index[df["domain"] == domain].to_numpy()
        chosen = rng.choice(idx, size=n, replace=False)
        test_idx.extend(chosen.tolist())
    test = df.loc[test_idx].reset_index(drop=True)
    train = df.drop(index=test_idx).reset_index(drop=True)
    return train, test


def write_all() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = make_docs()
    train, test = carve_test(df)
    df.to_parquet(PROCESSED / "docs.parquet", index=False)
    train.to_parquet(PROCESSED / "train.parquet", index=False)
    test.to_parquet(PROCESSED / "test.parquet", index=False)
    df.to_csv(PROCESSED / "docs.csv", index=False)
    return dict(full=len(df), train=len(train), test=len(test))


if __name__ == "__main__":
    counts = write_all()
    print(f"wrote rows: {counts}")
