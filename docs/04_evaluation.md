# Evaluation Plan — On-Device Quantized SLM Summarization

## 1. Held-out benchmark
- 2,000 documents (label-balanced across the 3 domains) carved from the 10k corpus before any tuning.

## 2. Primary scorecard
| Tier | ROUGE-L (avg) | p95 latency (ms) | model size (MB) |
|---|---|---|---|
| tight | – | – | – |
| small | – | – | – |
| medium | – | – | – |
| large | – | – | – |

Targets:
- `medium` tier: ROUGE-L ≥ 0.30, latency ≤ 100 ms.
- News domain at `medium`: ROUGE-L ≥ 0.35.

## 3. Per-domain breakdown
Per-tier × per-domain ROUGE-L; bar chart in the model card.

## 4. Pareto frontier
Plot latency (x) vs ROUGE-L (y); annotate each point with its tier + size. Pareto-frontier flag attached to each tier; the API returns it on every `/summarize` call.

## 5. Latency variance
For each tier, run 200 simulated calls and report mean + p95.

## 6. Robustness
- Sub-sample doc length to 50 / 200 / 400 tokens; measure ROUGE-L decay.
- Inject 10% noise tokens; measure ROUGE-L decay.

## 7. Operational metrics
- p95 API latency for `POST /summarize`
- Memory footprint of the extractive summarizer (notebook only)

## 8. Deployment readiness checklist
- [ ] All four tiers profiled
- [ ] Pareto-frontier chart in `mlops/model_card.md`
- [ ] News-domain ROUGE-L ≥ 0.35
