# Methodology — On-Device Quantized SLM Summarization

Three components:
1. **Summarizer** — turns a document into a short summary.
2. **Latency / size model** — predicts on-device cost as a function of input length and budget tier.
3. **Pareto evaluator** — sweeps budget tiers and emits a (latency, size, quality) frontier.

> Production summarizer = quantized SLM (GGUF int4/int8) via llama.cpp.
> Notebook fallback = extractive TextRank-style summarizer (TF-IDF + networkx PageRank).

---

## 1. Synthetic dataset
- 10,000 short documents across `news`, `technical`, `narrative` domains.
- Each document is built from domain-specific vocabulary templates.
- Reference summary is the first + last sentences of the templated document — a deterministic oracle.
- Schema: `doc_id, domain, doc_text, ref_summary, doc_tokens, summary_tokens`.

## 2. Extractive summarizer (notebook stand-in)
1. Split document into sentences.
2. Vectorize sentences with TF-IDF.
3. Build a sentence-similarity graph (cosine on TF-IDF vectors).
4. Run PageRank (networkx) to score sentences.
5. Take the top-$k$ sentences by score, where $k$ depends on the budget.
6. Truncate to a max-token cap.

The number of sentences kept and the token cap are functions of the budget tier:

| Budget tier | Max sentences | Max tokens | Quant tier (sim) | Size (MB, sim) |
|---|---|---|---|---|
| `tight`  | 1 | 24  | int4 | 35 |
| `small`  | 2 | 40  | int4 | 60 |
| `medium` | 3 | 60  | int8 | 110 |
| `large`  | 4 | 100 | fp16 | 220 |

## 3. Latency model
$$ L(d, b) = \alpha_b + \beta_b \cdot d_{\text{tokens}} $$
with per-budget coefficients:
- `tight`: $\alpha=20, \beta=0.05$
- `small`: $\alpha=30, \beta=0.10$
- `medium`: $\alpha=45, \beta=0.20$
- `large`: $\alpha=70, \beta=0.40$

Latencies in milliseconds. The simulator adds Gaussian noise (sd = 5% of mean) to mimic real device variance.

## 4. Quality metric — ROUGE-L
Implemented manually (no external rouge-score dependency) — longest common subsequence between predicted and reference, normalized:
$$ \text{R-L} = \frac{(1+\beta^2)\,P\,R}{R + \beta^2 P}, \quad P = \frac{\text{LCS}}{|\text{pred}|}, \; R = \frac{\text{LCS}}{|\text{ref}|}, \; \beta = 1 $$

## 5. Pareto frontier
Evaluator runs the full sweep (4 budget tiers × 10k docs) and emits per-tier means; the Pareto knee is the tier maximizing $R\text{-L} / (\alpha\,L + \beta\,S)$ for chosen weights $\alpha, \beta$.

## 6. References
- Mihalcea & Tarau, *TextRank*, 2004.
- Lin, *ROUGE: A Package for Automatic Evaluation of Summaries*, 2004.
- Dettmers et al., *LLM.int8()*, 2022 — quantization theory.
