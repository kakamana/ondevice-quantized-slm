"""Sentence segmentation, similarity graph, ROUGE-L, and budget table."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return parts


def sentence_similarity(sentences: list[str]) -> np.ndarray:
    """Cosine similarity over TF-IDF sentence vectors."""
    if len(sentences) < 2:
        return np.zeros((max(1, len(sentences)), max(1, len(sentences))))
    vec = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    return cosine_similarity(vec)


# ---------- ROUGE-L ----------
def _lcs_length(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l(pred: str, ref: str, beta: float = 1.0) -> float:
    """F-beta on LCS at the token level."""
    p_toks = pred.lower().split()
    r_toks = ref.lower().split()
    if not p_toks or not r_toks:
        return 0.0
    lcs = _lcs_length(p_toks, r_toks)
    if lcs == 0:
        return 0.0
    P = lcs / len(p_toks)
    R = lcs / len(r_toks)
    if P + R == 0:
        return 0.0
    return (1 + beta * beta) * P * R / (R + beta * beta * P)


# ---------- Budget table ----------
@dataclass(frozen=True)
class Budget:
    name: str
    max_sentences: int
    max_tokens: int
    quant: str
    size_mb: int
    alpha_ms: float
    beta_ms_per_token: float


BUDGETS: dict[str, Budget] = {
    "tight":  Budget("tight",  1, 24,  "int4", 35,  20.0, 0.05),
    "small":  Budget("small",  2, 40,  "int4", 60,  30.0, 0.10),
    "medium": Budget("medium", 3, 60,  "int8", 110, 45.0, 0.20),
    "large":  Budget("large",  4, 100, "fp16", 220, 70.0, 0.40),
}


def estimate_latency(doc_tokens: int, budget: str = "medium", noise_sd_pct: float = 0.05, rng: np.random.Generator | None = None) -> float:
    """Analytical latency model with optional Gaussian device noise."""
    b = BUDGETS[budget]
    base = b.alpha_ms + b.beta_ms_per_token * doc_tokens
    if noise_sd_pct > 0:
        rng = rng or np.random.default_rng(0)
        return float(base * (1.0 + rng.normal(0, noise_sd_pct)))
    return float(base)


def estimate_size_mb(budget: str = "medium") -> int:
    return BUDGETS[budget].size_mb
