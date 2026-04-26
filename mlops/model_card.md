# Model Card — On-Device Summarizer

## Intended use
On-device summarization for privacy-sensitive workloads. Returns a summary plus the (latency, size, Pareto-flag) triple so the calling app knows where on the trade-off curve it landed.

## Training data
10,000 synthetic short documents across `news`, `technical`, `narrative` domains with deterministic reference summaries (first + last sentence). See `data/data_card.md`.

## Model family
- **Production:** quantized SLM (e.g. ~1B params) in **GGUF int4 / int8**, loaded via llama.cpp.
- **Notebook fallback:** extractive TextRank-style summarizer (TF-IDF sentence vectors → cosine graph → networkx PageRank → top-k sentences → token-budget truncation).

## Metrics (held-out test, to be filled)
| Tier | ROUGE-L (avg) | p95 latency | size |
|---|---|---|---|
| tight | – | – | 35 MB |
| small | – | – | 60 MB |
| medium | – | – | 110 MB |
| large | – | – | 220 MB |

Targets:
- `medium` ROUGE-L ≥ 0.30, p95 ≤ 100 ms.
- News domain at `medium` ROUGE-L ≥ 0.35.

## Limitations
- Extractive can't paraphrase — best for news / structured docs.
- Latency is simulated in the notebook; production must measure on the target device.

## Ethical considerations
- On-device inference keeps user content local.
- All training/eval data is synthetic — no PII.

## Retraining
- Quarterly: refresh the SLM and re-profile each tier.

## Ownership
- DS lead: Asad
