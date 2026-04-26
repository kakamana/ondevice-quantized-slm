# Data Sources — #24 On-Device Quantized SLM Summarization

## Primary
| # | Source | URL | Use | License |
|---|--------|-----|-----|---------|
| 1 | Synthetic doc generator | `src/slm_quant/data.py` | 10,000 short docs + reference summaries | MIT |

## Secondary / reference
| Source | URL | Use |
|---|---|---|
| TextRank | https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf | Extractive method |
| ROUGE | https://aclanthology.org/W04-1013/ | Quality metric |
| llama.cpp | https://github.com/ggerganov/llama.cpp | Production inference target |
| GGUF format | https://github.com/ggerganov/ggml/blob/master/docs/gguf.md | Quantization-tier file format |

## How to download
Nothing to download — run:
```bash
python -m slm_quant.data
```
This writes:
- `data/processed/docs.parquet` (10,000)
- `data/processed/test.parquet` (2,000)

## Attribution
If you publish results using this pipeline, please cite the project repo.
