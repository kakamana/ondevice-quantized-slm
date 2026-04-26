# Business Requirements — On-Device Quantized SLM Summarization

## 1. Problem Statement
Mobile and edge applications increasingly want **summarization** features without round-tripping every document to a hosted endpoint (e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct). On-device inference solves three things simultaneously: privacy (PII never leaves the device), cost (zero per-call fee), and offline use. The constraint is the device budget — RAM, latency, and binary size. We need a **Pareto-frontier evaluation harness** so the product team can pick the right (latency, model size, quality) trade-off for their device class.

## 2. Stakeholders
| Role | Interest | Success criterion |
|------|----------|-------------------|
| Mobile engineer | Fits the binary budget | Model ≤ 100 MB on disk |
| Product manager | Acceptable summary quality | ROUGE-L ≥ 0.30 on benchmark |
| Privacy lead | Data never leaves device | 100% on-device inference |
| Battery/thermal team | Low joules per summary | p95 latency ≤ 100 ms on small budget |

## 3. Business Objectives
1. **ROUGE-L ≥ 0.30** on the held-out 10k-doc benchmark (averaged across domains).
2. **p95 latency ≤ 100 ms** at the smallest budget setting.
3. **Model size ≤ 100 MB** at the chosen Pareto knee.
4. **Pareto frontier** documented and consumed by the API + UI.

## 4. KPIs
| KPI | Definition | Target | Baseline |
|-----|-----------|--------|----------|
| ROUGE-L (avg) | mean across domains | ≥ 0.30 | 0.21 |
| ROUGE-L (news) | per-domain | ≥ 0.35 | 0.24 |
| p95 latency (small) | ms per summary | ≤ 100 | – |
| Model size (knee) | MB | ≤ 100 | – |

## 5. Scope
**In scope:** English short documents (≤ ~512 tokens) across 3 domains (news, technical, narrative). Extractive summarization in the notebook fallback; production swap-in is a quantized SLM via llama.cpp.
**Out of scope:** abstractive long-form generation, multilingual summarization, instruction-tuned freeform Q&A.

## 6. Constraints & Assumptions
- **On-device only** in production.
- **Notebook fallback** must run in Dataiku DSS — extractive summarizer + analytical latency model only.
- **Vendor-neutral baseline** — quality reference is any hosted endpoint, but the production stack does not call one.

## 7. Risks
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Extractive cannot match abstractive quality | High | Medium | Document the Pareto gap; production SLM closes part of it |
| Latency varies by device | High | Medium | Latency simulator parameterized by device class |
| ROUGE-L not aligned with user perception | Medium | High | Pair with a human-eval batch (5×100 samples) |
| GGUF format churn | Medium | Low | Pin model + quantization scheme in model card |
