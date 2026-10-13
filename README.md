# Hybrid-AI Log Diagnostics: DRAIN + LogBERT + SLM

Double-degree AI MSc thesis project — **Politehnica University of Bucharest / 3iL Engineers** — conducted with **Worldline** (Client Delivery 2, Financial Services division).

## Problem

Banking log streams (≈172M lines/month on our scope) bury incidents under alert cascades, and monolithic-LLM analysis is too costly and unreliable for production. Goal: cut MTTR at controlled cost, without moving data off the private network.

## Pipeline (evidence-first)

```text
DRAIN parsing → sessionisation → LogBERT detection → inference bundle → constrained Qwen SLM
```

- **LogBERT decides** (paper-style top-g scoring) and outputs evidence (score, out-of-top-g masked positions, Drain templates).
- A **deterministic policy** filters; the **SLM explains** without ever re-deciding (strict JSON, finite action catalogue, human-in-the-loop).
- No generative confidence scores — the only decision-strength signal is LogBERT's score-vs-threshold margin.

## Experiments & results (locked test, validation-only selection)

Protocol throughout: normal-only training, `(m, g, r)` selected on a separate validation split, test scored once after freezing. Official LogBERT repo pinned at `a845e61`.

| Setup | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| V2 HDFS `m=.75,g=8,r=.04` (1 seed, 506k sessions) | 82.67% | 83.62% | 83.14% | 0.9235 |
| V3 BGL `m=.50,g=15,r=.01` (mean ± std, n=3, 11k sessions) | 90.45 ± 0.66% | 91.66 ± 0.07% | 91.05 ± 0.35% | 0.9596 ± 0.0001 |

- vs LogBERT paper: **partial exceedance on HDFS** (recall +5.5 pts, F1 +0.8 pt; precision −4.35 pts), **match on BGL** (F1 +0.22 < inter-seed std).
- Key finding: the **optimal mask ratio follows sequence length** (0.75 @ ~19 tokens on HDFS, official 0.50 confirmed @ ~562 tokens on BGL).
- FinOps: **÷43 inference cost** vs all-LLM (≈€76–147/month), sovereign local hosting (GDPR/PCI-DSS).

## Repository layout

```text
logbert_v3_bgl.ipynb                  # V3 protocol notebook (BGL ablation + multi-seed + optional HDFS Phase C)
logbert_v3_bgl-result.ipynb           # Executed BGL run (Phase A/B, bundle, figures)
logbert_hdfs_v2_resultats.ipynb       # V2 HDFS run (m-ablation, locked test, SLM section)
logbert_hdfs_full_baseline-resultat_run.ipynb  # V1 official baseline run
gen_v3.py / add_phasec.py             # Notebook generators (reproducible protocol source)
academicResearch.tex / partie2.tex    # Thesis report (EN / FR)
figs/                                 # Figures extracted from runs + title-page logos
```

## Reproduce (Kaggle GPU)

1. Open `logbert_v3_bgl.ipynb`, enable GPU + internet.
2. Run top to bottom: BGL auto-downloads from LogHub (MD5-checked), official Drain parsing, training (`m ∈ {0.35, 0.50, 0.65}`), validation selection, locked test, multi-seed repeat.
3. Set `RUN_PHASE_C=True` for the extra HDFS seeds (needs `HDFS.log`; attach via *Add Input* if Zenodo throttles).
4. Train/checkpoint files resume automatically — a cut session picks up where it stopped.
5. Download `logbert_v3_bgl_results.json` / `logbert_v3_hdfs_extraseeds.json` at the end (Kaggle wipes `/kaggle/working` on reset).

## References

- Guo et al., *LogBERT: Log Anomaly Detection via BERT*, IJCNN 2021 ([code](https://github.com/HelenGuohx/LogBERT))
- He et al., *Drain: An Online Log Parsing Approach*, ICWS 2017
- Xu et al., *Detecting Large-Scale System Problems by Mining Console Logs*, SOSP 2009 (HDFS)
- Zhu et al., *LogHub: A Large Collection of System Log Datasets*, ISSRE 2020 ([data](https://github.com/logpai/loghub))
