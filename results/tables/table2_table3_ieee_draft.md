# Draft Persamaan (1)–(4), TABLE II, & TABLE III (Standar IEEE — Hari 3)

Seluruh angka desimal berawalan nol (*leading zero*), tanpa garis vertikal, dan 100% tervalidasi silang dengan `pytrec_eval` (`max_abs_diff = 0.00000000`).

---

## 1. Persamaan Matematika (Section III-C & III-D)

- **Persamaan (1) — Cosine Similarity Ter-normalisasi L2:**
  $$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q}^{\top}\mathbf{d}}{\|\mathbf{q}\|_{2}\,\|\mathbf{d}\|_{2}}$$

- **Persamaan (2) — Normalized Discounted Cumulative Gain (`nDCG@10`):**
  $$\text{nDCG}@10(q) = \frac{\text{DCG}@10(q)}{\text{IDCG}@10(q)}, \quad \text{dengan } \text{DCG}@10(q) = \sum_{i=1}^{10} \frac{\text{rel}(q, d_i)}{\log_{2}(i + 1)}$$

- **Persamaan (3) — Mean Reciprocal Rank (`MRR@10`):**
  $$\text{MRR}@10 = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(r_q \le 10) \cdot \frac{1}{r_q}$$
  di mana $r_q$ adalah peringkat pertama dokumen relevan ($\text{rel} > 0$) untuk kueri $q$.

- **Persamaan (4) — Recall pada Pemotongan 5 (`Recall@5`):**
  $$\text{Recall}@5(q) = \frac{|\{d \in \text{Top-5}(q) : \text{rel}(q, d) > 0\}|}{|\{d \in \mathcal{D} : \text{rel}(q, d) > 0\}|}$$

---

## 2. Draf TABLE II (Section IV-A — Retrieval Effectiveness)

**TABLE II.** ZERO-SHOT RETRIEVAL EFFECTIVENESS ON SCIFACT TEST SPLIT ($N = 300$ QUERIES, $5,183$ DOCUMENTS)

| Model / Document Configuration | Paradigm | `nDCG@10` | `MRR@10` | `Recall@5` | $\Delta\text{nDCG@10}$ vs. A |
| :--- | :---: | :---: | :---: | :---: | :---: |
| BM25 Baseline (Thakur *et al.*, 2021) | Lexical | 0.6650 | — | — | — |
| DPR (Thakur *et al.*, 2021) | Dense (MS MARCO) | 0.3180 | — | — | — |
| ANCE (Thakur *et al.*, 2021) | Dense (MS MARCO) | 0.5070 | — | — | — |
| TAS-B (Thakur *et al.*, 2021) | Dense (MS MARCO) | 0.6430 | — | — | — |
| **SPECTER2 — Config A (`Title`)** | Dense (SciRepEval) | 0.5705 | 0.5352 | 0.6345 | 0.0000 |
| **SPECTER2 — Config B (`Abstract`)** | Dense (SciRepEval) | 0.6209 | 0.5828 | 0.6953 | +0.0504 |
| **SPECTER2 — Config C (`Title + [SEP] + Abstract`)** | Dense (SciRepEval) | **0.6455** | **0.6095** | **0.7155** | **+0.0750** |

---

## 3. Draf TABLE III (Section IV-B & IV-C — Wilcoxon Test & Efficiency Trade-Off)

**TABLE III.** PAIRED WILCOXON SIGNED-RANK TEST ON `nDCG@10` ($\alpha_{\text{bonf}} = 0.0167$) AND TESLA T4 ENCODING EFFICIENCY

| Comparison / Configuration | $\Delta\text{nDCG@10}$ | Wilcoxon $W$ | Raw $p$-value | Adj. $p_{\text{bonf}}$ | Sig. ($< 0.0167$) | Wins / Losses / Ties | GPU Time (s) | Speedup vs. C |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A vs. Config C** | +0.0750 | 2,642.5 | $2.94 \times 10^{-5}$ | $8.81 \times 10^{-5}$ | **Yes** | 88 / 46 / 166 | 12.24 vs. 160.66 | 13.12x (A) |
| **Config B vs. Config C** | +0.0246 | 1,677.0 | 0.0361 | 0.1082 | No* | 55 / 39 / 206 | 153.17 vs. 160.66 | 1.05x (B) |
| **Config A vs. Config B** | +0.0504 | 4,524.5 | 0.0325 | 0.0976 | No* | 82 / 68 / 150 | 12.24 vs. 153.17 | 12.51x (A vs. B) |

*\*Nominally significant at unadjusted $\alpha = 0.05$, but does not pass strict Bonferroni correction ($\alpha_{\text{bonf}} = 0.05 / 3 = 0.0167$).*
