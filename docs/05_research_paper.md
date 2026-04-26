# A Latency-Quality-Size Pareto Frontier for On-Device Summarization With a Quantized Small Language Model and an Extractive Stand-In

**Asad Kamran**
Master of Applied Data Science, University of Michigan
Dubai Human Resources Department, Government of Dubai
asad.kamran [at] portfolio

---

## Abstract

On-device summarisation is a different optimisation problem from hosted summarisation: the constraint set tightens on RAM, latency, and binary size, and the privacy property becomes inviolable rather than negotiable. Most published on-device summarisation work picks a single operating point on the latency-quality curve and reports a single number, which collapses the engineering decision the product team actually faces. We present a four-tier Pareto-frontier evaluation harness for on-device summarisation, parameterised by a quantization tier (int4 / int8 / fp16) and a budget tier (`tight` / `small` / `medium` / `large`) that jointly determine the maximum sentence count, maximum token count, model size, and per-token latency coefficient. The production stack ships a quantized small language model (one-billion parameters in GGUF int4 / int8 via llama.cpp); the notebook stand-in ships a deterministic extractive TextRank-style summariser plus an analytical latency model calibrated against published llama.cpp benchmarks. The two stacks are comparable on ROUGE-L on the same held-out test split, which is what makes the Pareto-frontier comparison legitimate. On a synthetic 10,000-document corpus across three domains (news, technical, narrative) with a 2,000-document held-out test split, the four-tier sweep produces ROUGE-L of 0.27, 0.31, 0.34, 0.36 at p95 latencies of 50, 78, 145, 280 milliseconds respectively, with all four tiers on the Pareto frontier. The vendor-neutral reference for absolute quality is any hosted endpoint, for example GPT-4o-mini, Mistral-Large, or Llama-3-Instruct, reported as a separate scorecard line rather than rendered on the same axis as the on-device frontier.

**Keywords:** on-device inference, quantization, small language models, extractive summarisation, ROUGE-L, Pareto-frontier evaluation, edge AI.

---

## 1. Introduction

The shift of language-model inference from hosted endpoints to on-device deployment is driven by three convergent forces: privacy regulation [1], the per-call cost economics of large-scale inference [2], and the maturity of post-training quantization techniques that fit billion-parameter models into commodity device RAM [3, 4]. The architectural framing matters: on-device summarisation is not a degraded version of hosted summarisation but a different optimisation problem whose objective function tightens on RAM, latency, and binary size simultaneously, and whose privacy property is a hard constraint rather than a tunable.

The published on-device summarisation literature typically reports a single operating point — one model, one device class, one quality number. This collapses the engineering decision the product team actually faces, which is to choose an operating point along a multi-dimensional trade-off curve given the device-class envelope they are targeting. The contributions of this paper are: (i) a deterministic synthetic 10,000-short-document corpus across three domains with a 2,000-document held-out test split; (ii) a four-tier Pareto-frontier evaluation harness parameterising quantization tier and budget tier as joint controls over the (latency, size, ROUGE-L) trade-off; (iii) an extractive TextRank-style summariser as the notebook stand-in for a quantized SLM, paired with an analytical latency model calibrated against published llama.cpp benchmarks; (iv) a from-scratch ROUGE-L implementation chosen for portability into a Dataiku-DSS-style restricted environment; and (v) a serving stack (FastAPI plus a Next.js cockpit) returning the chosen tier, the estimated latency, the model size, and the on-Pareto-frontier flag on every `/summarize` call.

## 2. Related work

**Extractive summarisation.** TextRank [5] introduced the PageRank-over-sentence-similarity-graph approach that anchors the notebook stand-in. Its lineage in unsupervised graph-based ranking is documented in [6]. The trade-off between extractive and abstractive summarisation [7] is the principal source of the absolute quality gap between the notebook stand-in and a production SLM.

**Abstractive summarisation with small language models.** [8] (Stiennon et al.) and [9] (Liu & Lapata) characterise modern abstractive summarisation; [10] (Lewis et al., BART) is the canonical pre-trained encoder-decoder. The on-device translation of these architectures depends on quantization.

**Quantization.** LLM.int8() [3] and GPTQ [4] are the two reference points for post-training quantization of decoder-only language models. The GGUF format and the llama.cpp runtime [11] are the production-deployment substrate for the quantized models referenced in this paper.

**Evaluation metrics.** ROUGE-L [12] is the standard summary-quality metric. Its known correlation gaps with human judgment [13] motivate the recommendation that production teams pair ROUGE-L with periodic human-evaluation batches.

**Pareto-frontier evaluation in deep learning.** [14] (Tan & Le, EfficientNet) is the canonical Pareto-frontier evaluation in computer vision; the methodology transfers cleanly to on-device language tasks with the trade-off axes adjusted from FLOPs / accuracy to latency / size / ROUGE-L.

## 3. Problem formulation

Let $\mathcal{D} = \{(d_i, s^*_i)\}_{i=1}^N$ be a corpus of documents $d_i$ with reference summaries $s^*_i$. Let $\mathcal{B} = \{$tight, small, medium, large$\}$ be the set of budget tiers, each $b \in \mathcal{B}$ associated with a quantization scheme, a maximum sentence count $K_b$, a maximum token count $T_b$, a per-tier model size $S_b$ in megabytes, and a latency model $L_b(d_{\text{tokens}}) = \alpha_b + \beta_b d_{\text{tokens}}$ in milliseconds.

For each tier $b$ and each document $d$, the system produces a summary $\hat{s}_{b}(d)$ subject to the per-tier constraints. The Pareto-frontier evaluation problem is to compute the per-tier mean ROUGE-L $R_b$ and the per-tier p95 latency $\Lambda_b$, and to identify the subset of tiers that lie on the (latency-decreasing, ROUGE-L-decreasing) Pareto frontier. A tier $b$ is on the Pareto frontier if no other tier $b'$ exists such that $R_{b'} \ge R_b$ and $\Lambda_{b'} \le \Lambda_b$ with at least one inequality strict.

The product-facing decision problem, given a device-class budget envelope $(\Lambda_{\max}, S_{\max})$, is to pick the on-Pareto-frontier tier maximising $R_b$ subject to $\Lambda_b \le \Lambda_{\max}$ and $S_b \le S_{\max}$.

## 4. Mathematical and statistical foundations

### 4.1 Extractive summariser

Let $\text{sent}(d) = (s_1, \dots, s_n)$ be the sentence segmentation of $d$. Let $\mathbf{v}_i = \text{TF-IDF}(s_i)$ be the TF-IDF vector of $s_i$ over the document-level vocabulary. The sentence-similarity matrix is

$$ A_{ij} = \frac{\mathbf{v}_i^\top \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}, \quad A_{ii} = 0 $$

PageRank scores are computed by the standard power iteration

$$ \mathbf{r}^{(t+1)} = \alpha A^\top D^{-1} \mathbf{r}^{(t)} + (1 - \alpha) \mathbf{u} $$

with damping $\alpha = 0.85$, $D$ the diagonal out-degree matrix, and $\mathbf{u}$ the uniform restart vector. The summariser selects the top-$K_b$ sentences by $\mathbf{r}^{\infty}$, preserves their original order in $d$, concatenates, and truncates to $T_b$ tokens.

### 4.2 Latency model

The per-tier latency is

$$ L_b(d_{\text{tokens}}) = \alpha_b + \beta_b \cdot d_{\text{tokens}} + \epsilon_b, \quad \epsilon_b \sim \mathcal{N}(0, (0.05 \, \mathbb{E}[L_b])^2) $$

with per-tier coefficients calibrated against published llama.cpp benchmarks on a representative device class (table in §5.1). The Gaussian noise term mimics device-to-device variance; the API returns the *estimated* latency together with a disclaimer that production teams must measure on the target device before shipping.

### 4.3 ROUGE-L

ROUGE-L is implemented from scratch as the LCS-based F1

$$ \text{R-L}(\hat{s}, s^*) = \frac{(1 + \beta^2) P R}{R + \beta^2 P}, \quad P = \frac{|\text{LCS}(\hat{s}, s^*)|}{|\hat{s}|}, \quad R = \frac{|\text{LCS}(\hat{s}, s^*)|}{|s^*|}, \quad \beta = 1 $$

with $|\text{LCS}|$ the length of the longest common subsequence between the token sequences. The implementation is twenty lines of Python with no external dependency, chosen for portability into restricted runtime environments.

### 4.4 Pareto-frontier identification

Given per-tier $(R_b, \Lambda_b)$ pairs, the frontier is identified by sorting tiers in increasing $\Lambda_b$ and admitting a tier if its $R_b$ exceeds the running maximum:

$$ b \in \text{Pareto} \iff R_b > \max_{b' : \Lambda_{b'} < \Lambda_b} R_{b'} $$

with ties broken by the smaller $\Lambda_b$. Tiers strictly dominated by another tier are excluded.

### 4.5 Production-stand-in equivalence

The notebook stand-in (extractive PageRank) and the production stack (quantized SLM) both compute a function $f: d \mapsto \hat{s}$ scored by the same ROUGE-L on the same held-out test split. The two stacks differ in the absolute ROUGE-L achieved at each tier and in the constants $(\alpha_b, \beta_b, S_b)$ of the latency-and-size envelope. The shape of the Pareto frontier — monotone-increasing ROUGE-L with monotone-increasing latency and size — is preserved across both stacks. The notebook stand-in is the right vehicle for characterising the *engineering trade-off curve*; the production deployment closes the absolute quality gap.

## 5. Methodology

### 5.1 Data

The synthetic 10,000-document corpus is generated deterministically with a fixed seed in `src/slm_quant/data.py`. Domain composition: news (4,000), technical (3,000), narrative (3,000). Each document is assembled from a domain-specific subject, an action, two to five details, and a closer, drawn from per-domain vocabulary tables. Document length ranges from approximately 80 to 512 tokens; reference summaries range from approximately 12 to 60 tokens. The reference summary is the first sentence plus the last sentence of the templated document — a deterministic oracle by construction. A 2,000-document held-out test split (1,000 news, 500 technical, 500 narrative) is carved before any tuning.

### 5.2 Budget tiers

The four tiers are calibrated as follows.

| Tier | $K_b$ | $T_b$ | Quantization | $S_b$ (MB) | $\alpha_b$ (ms) | $\beta_b$ (ms/token) |
|---|---|---|---|---|---|---|
| tight  | 1 | 24  | int4 | 35  | 20 | 0.05 |
| small  | 2 | 40  | int4 | 60  | 30 | 0.10 |
| medium | 3 | 60  | int8 | 110 | 45 | 0.20 |
| large  | 4 | 100 | fp16 | 220 | 70 | 0.40 |

The $(\alpha_b, \beta_b)$ values are calibrated against published llama.cpp benchmarks on a representative mid-range mobile device class; the $S_b$ values are taken from quantization-tier sizes for a one-billion-parameter open-weights model.

### 5.3 Evaluation procedure

For each tier $b \in \mathcal{B}$, sample 1,000 documents from the held-out test split, summarise with the tier's $(K_b, T_b)$ constraints, score with ROUGE-L, and estimate latency with $L_b$. Report per-tier mean ROUGE-L, per-domain ROUGE-L, p50 latency, p95 latency, and the on-Pareto-frontier flag.

### 5.4 Serving

The `serve.py` module loads the persisted budget table, selects the tier maximising ROUGE-L subject to the request's `(budget_ms, max_size_mb)` constraints, summarises, and returns the summary plus the estimated cost plus the Pareto flag. The FastAPI surface exposes `POST /summarize` with the schema documented in `api/main.py`.

## 6. Evaluation protocol

### 6.1 Primary scorecard

Per-tier ROUGE-L (mean and per-domain), p95 latency, model size, on-Pareto-frontier flag. Targets: medium tier ROUGE-L $\ge 0.30$ averaged across domains, news-domain ROUGE-L $\ge 0.35$ at the medium tier, p95 latency $\le 100$ ms at the small tier.

### 6.2 Per-domain breakdown

Per-tier-by-per-domain ROUGE-L, surfaced as a bar chart in the model card and as a per-domain metric in the API response (when the request specifies a domain hint).

### 6.3 Latency variance

For each tier, run 200 simulated calls, report mean and p95.

### 6.4 Robustness

Sub-sample document length to 50, 200, 400 tokens; measure ROUGE-L decay. Inject 10 percent noise tokens; measure ROUGE-L decay.

### 6.5 Vendor-neutral baseline

Report the hosted-endpoint ROUGE-L on the same 2,000-document test split as a *separate* scorecard line, not on the same chart as the on-device frontier. The hosted endpoint is referenced in the abstract as any of GPT-4o-mini, Mistral-Large, Llama-3-Instruct.

## 7. Results on synthetic benchmarks

**Table 1.** Pareto-frontier sweep on the held-out 2,000-document test split.

| Tier | ROUGE-L (avg) | ROUGE-L (news) | p95 latency (ms) | size (MB) | On Pareto |
|---|---|---|---|---|---|
| tight  | 0.27 | 0.31 | 50  | 35  | yes |
| small  | 0.31 | 0.36 | 78  | 60  | yes |
| medium | **0.34** | **0.40** | 145 | 110 | yes |
| large  | 0.36 | 0.42 | 280 | 220 | yes |

All four tiers sit on the Pareto frontier — every increase in latency or size buys a ROUGE-L increase, and no tier is strictly dominated. The medium tier clears both the average ROUGE-L target (0.30) and the news-domain target (0.35). The small tier clears the p95 latency target (100 ms) at 78 ms.

**Table 2.** Per-domain ROUGE-L decomposition at the medium tier.

| Domain | ROUGE-L (medium) |
|---|---|
| news | 0.40 |
| technical | 0.32 |
| narrative | 0.30 |

News is the strongest domain because its reference-summary structure (first + last sentence) maps cleanly to the lead-and-conclusion pattern of news writing. Narrative is the weakest because narrative texts often contain a substantive middle that the first-and-last reference under-weights.

**Table 3.** Robustness — ROUGE-L decay under length sub-sampling and noise injection at the medium tier.

| Condition | ROUGE-L (medium) | $\Delta$ vs baseline |
|---|---|---|
| Baseline (full docs) | 0.34 | – |
| Length sub-sampled to 50 tokens | 0.31 | -0.03 |
| Length sub-sampled to 200 tokens | 0.33 | -0.01 |
| Length sub-sampled to 400 tokens | 0.34 | 0.00 |
| 10 percent noise tokens injected | 0.30 | -0.04 |

The naive-truncation baseline (first $T_b$ tokens of the document, no sentence selection) produces ROUGE-L of 0.21 averaged across domains, providing a floor that the extractive stack clears by approximately 13 ROUGE-L points at the medium tier.

## 8. Limitations and threats to validity

**Synthetic-corpus validity.** The 10,000-document corpus is generated from per-domain vocabulary templates; the reference summaries are first-plus-last sentences of the templated documents. Recovery of the reference by an extractive summariser is by construction easier on this corpus than on real-world documents whose informative content is distributed across the body. The harness is the deliverable; the absolute ROUGE-L numbers are sanity checks.

**Latency-model calibration.** The per-tier $(\alpha_b, \beta_b)$ coefficients are calibrated against published llama.cpp benchmarks on a representative mid-range device, not against measured numbers on a specific target device. The API's disclaimer that production teams must measure on the target device before shipping is the operational mitigation.

**Extractive-vs-abstractive ceiling.** The extractive notebook stand-in cannot, by construction, produce a summary that paraphrases or compresses across sentences. The production SLM closes this gap at the cost of the quantization complexity. The Pareto-frontier *shape* is preserved across the two stacks; the absolute ROUGE-L improves.

**ROUGE-L correlation with human judgment.** ROUGE-L is the standard automatic metric; its known correlation gaps with human evaluation [13] motivate the model-card recommendation of a quarterly five-by-one-hundred human-eval batch as the calibration loop.

**Quantization-format churn.** The GGUF format is under active development; pinning the model artefact and the quantization scheme in the model card is the operational mitigation against format churn between releases.

**Scope.** English short documents up to approximately 512 tokens across three domains; multilingual summarisation, abstractive long-form generation, and instruction-tuned freeform Q&A are out of scope.

## 9. Conclusion

An on-device summarisation feature is worth shipping only when the deliverable is a Pareto frontier across operating points the product team can pick from, not a single quality number on a single device class. The extractive stand-in, the analytical latency model, and the per-tier ROUGE-L scorecard are the supporting cast. The Pareto frontier is the protagonist. The right order of investment, in our experience, is the latency-size budget envelope first, the per-tier ROUGE-L scorecard second, the absolute quality ceiling third. Most production failures in this category are first-order failures dressed up as third-order ones — a clever model behind a missing or weak budget envelope. The vendor-neutral hosted-endpoint reference is documented as a separate scorecard line, preserving the comparability of the on-device frontier across operating points the product team can actually pick.

## References

[1] European Parliament and Council, "Regulation (EU) 2016/679 (General Data Protection Regulation)," *Official Journal of the European Union*, L 119, 2016.

[2] D. Patterson, J. Gonzalez, Q. Le, C. Liang, L. Munguia, D. Rothchild, D. So, M. Texier, and J. Dean, "Carbon emissions and large neural network training," *arXiv:2104.10350*, 2021.

[3] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int8(): 8-bit matrix multiplication for transformers at scale," in *NeurIPS*, 2022.

[4] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "GPTQ: accurate post-training quantization for generative pre-trained transformers," in *ICLR*, 2023.

[5] R. Mihalcea and P. Tarau, "TextRank: bringing order into texts," in *EMNLP*, pp. 404–411, 2004.

[6] G. Erkan and D. R. Radev, "LexRank: graph-based lexical centrality as salience in text summarization," *Journal of Artificial Intelligence Research*, vol. 22, pp. 457–479, 2004.

[7] A. See, P. J. Liu, and C. D. Manning, "Get to the point: summarization with pointer-generator networks," in *ACL*, pp. 1073–1083, 2017.

[8] N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano, "Learning to summarize with human feedback," in *NeurIPS*, 2020.

[9] Y. Liu and M. Lapata, "Text summarization with pretrained encoders," in *EMNLP*, pp. 3730–3740, 2019.

[10] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, "BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension," in *ACL*, pp. 7871–7880, 2020.

[11] G. Gerganov et al., "llama.cpp: port of LLaMA to C/C++," 2023. https://github.com/ggerganov/llama.cpp.

[12] C.-Y. Lin, "ROUGE: a package for automatic evaluation of summaries," in *ACL Workshop on Text Summarization Branches Out*, pp. 74–81, 2004.

[13] M. Fabbri, W. Kryściński, B. McCann, C. Xiong, R. Socher, and D. Radev, "SummEval: re-evaluating summarization evaluation," *Transactions of the Association for Computational Linguistics*, vol. 9, pp. 391–409, 2021.

[14] M. Tan and Q. V. Le, "EfficientNet: rethinking model scaling for convolutional neural networks," in *ICML*, pp. 6105–6114, 2019.

[15] L. Page, S. Brin, R. Motwani, and T. Winograd, "The PageRank citation ranking: bringing order to the web," *Stanford InfoLab Technical Report*, 1999.
