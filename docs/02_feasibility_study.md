# Feasibility Study — On-Device Quantized SLM Summarization

## 1. Data feasibility
- **10,000 short documents + reference summaries**, template-generated across three domains: `news`, `technical`, `narrative`.
- Schema: `doc_id, domain, doc_text, ref_summary, doc_tokens, summary_tokens`.
- Document length: 80–512 tokens; reference summary length: 12–60 tokens.
- No PII; deterministic with seed=42.

## 2. Technical feasibility
- **Production:**
  - SLM: open-weights small model (e.g. ~1B params) in **GGUF int4 / int8** via llama.cpp.
  - Latency-quality Pareto: sweep quantization × prompt-template × max-output-tokens.
- **Notebook fallback (Dataiku-compatible):**
  - Summarizer: TextRank-style extractive on TF-IDF sentence vectors via networkx.
  - Latency: analytical function $L(d, b) = \alpha + \beta \cdot d_{\text{tokens}} + \gamma / b$ where $d$ is document length and $b$ is the budget tier.
  - Model size simulator: maps quantization tier → MB.
- **Compute:** notebook runs on a single CPU; no GPU required.

## 3. Economic feasibility
| Line item | Monthly cost |
|-----------|--------------|
| 1× small container (API + UI) | ~$8 |
| Hosted-endpoint quality reference (optional) | metered |
| **Total** | **~$8 / mo** |

Value: every on-device summary is a saved hosted-endpoint call — at scale this dwarfs the development cost.

## 4. Operational feasibility
- **Retraining:** the extractive summarizer is non-parametric — no training. The production SLM is updated on quarterly cadence.
- **Monitoring:** ROUGE-L vs reference on a fixed canary set; latency p95 by device class.
- **Human-in-the-loop:** quarterly human-eval batch (5×100 samples) calibrates ROUGE-L to perceived quality.

## 5. Ethical / legal feasibility
- All synthetic data; no PII.
- On-device inference protects user content.

## 6. Recommendation
**Go.** The harness ships measurable Pareto numbers, the fallback runs in Dataiku, and the API answers the only product question that matters: "given a budget, which point on the curve do I get?".
