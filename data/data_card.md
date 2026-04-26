# Data Card — #24 On-Device Quantized SLM Summarization

## Dataset composition
| Layer | Source | Rows | Schema |
|---|---|---|---|
| Document corpus | template generator (`src/slm_quant/data.py`) | 10,000 | `doc_id, domain, doc_text, ref_summary, doc_tokens, summary_tokens` |
| Held-out test | carved from corpus | 2,000 | same |

## Domain mix
- `news` — 4,000
- `technical` — 3,000
- `narrative` — 3,000

## Field semantics
- `doc_text` — UTF-8, sentence-segmentable, 80–512 tokens
- `ref_summary` — first + last sentence of templated document; 12–60 tokens
- `doc_tokens`, `summary_tokens` — pre-computed for latency simulation

## Known biases
- Vocabulary-controlled — does not capture every real-world style
- English only

## PII
None. Fully synthetic.

## Splits
- Train pool: 8,000 (not strictly used in the extractive setup; useful for any learned ranker)
- Test: 2,000 (1,000 news / 500 technical / 500 narrative)

## Reproducing
```bash
python -m slm_quant.data
```
Deterministic with seed=42.

## Licensing
- All synthetic content is MIT (this repo).
