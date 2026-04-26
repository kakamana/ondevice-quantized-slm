# On-Device Quantized SLM Summarization

> **Summarize 10,000 short documents on-device under a latency + size budget — and prove the Pareto frontier.** Production stack ships a quantized small language model (GGUF int4/int8) via llama.cpp; the notebook ships an extractive (TextRank-style) summarizer + a latency simulator so the entire workflow runs in Dataiku DSS without a GPU.

![Python](https://img.shields.io/badge/python-3.11-blue) ![scikit-learn](https://img.shields.io/badge/sklearn-1.4-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688) ![Made in Dubai](https://img.shields.io/badge/made%20in-Dubai-black)

## Why this project
- Hosted endpoints (e.g. GPT-4o-mini, Mistral-Large, Llama-3-Instruct) work well but they are **off-device, networked, and metered**. For privacy-sensitive HR / mobile / edge workloads we want a **quantized SLM** that runs locally.
- This project ships the full evaluation harness — latency budget, model-size budget, ROUGE-L on a 10k-doc benchmark, and a Pareto-frontier plot of latency vs quality across budget settings.
- The notebook fallback uses a deterministic extractive summarizer + an analytical latency model so it runs anywhere.

## Table of contents
- [Business Requirements](./docs/01_business_requirements.md)
- [Feasibility Study](./docs/02_feasibility_study.md)
- [Methodology — Extractive + Pareto evaluation](./docs/03_methodology.md)
- [Evaluation Plan](./docs/04_evaluation.md)
- [Data card](./data/data_card.md) · [Data sources](./data/data_sources.md)
- [Notebooks](./notebooks/) · [Source](./src/slm_quant/) · [API](./api/main.py) · [UI](./ui/app/page.tsx)
- [CLAUDE.md](./CLAUDE.md) — paste prompt to resume in this folder

## Headline results (target)

| Metric | Naive truncation | Extractive + budget-aware | Target |
|---|---|---|---|
| ROUGE-L (avg) | 0.21 | **0.34** | ≥ 0.30 |
| ROUGE-L (news domain) | 0.24 | **0.40** | ≥ 0.35 |
| p95 latency (small budget) | – | **< 80 ms** | < 100 ms |
| Model size at Pareto knee | – | **< 60 MB** | < 100 MB |

## Quickstart
```bash
pip install -e ".[dev]"
python -m slm_quant.data           # writes 10,000 short docs + reference summaries
jupyter lab notebooks/
uvicorn api.main:app --reload
cd ui && npm install && npm run dev
```

## Stack
Python · pandas · scikit-learn · networkx · FastAPI · Next.js · Tailwind · joblib · matplotlib · seaborn

## Author
Asad — MADS @ University of Michigan · Dubai HR
