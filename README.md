# LLM Hallucination Detection with Mondrian Conformal Risk Control

Domain-stratified conformal risk control (CRC) for detecting hallucinations in LLM
outputs, with distribution-free, per-domain risk-control guarantees. Two detector
backbones are compared: a TF-IDF + logistic-regression baseline (Phase 1) and BERT
sentence embeddings (Phase 2).

## Overview

Each candidate answer is scored for hallucination risk, and Mondrian CRC fits a
per-domain threshold so the conformal risk guarantee holds *within* each of the six
HaluBench domains, not just on average. The tables below report the downstream
classifier metrics this produces — hallucination recall, false-alarm rate, precision —
which are distinct from the risk that CRC actually controls (see `TECHNICAL_REPORT.md`,
§1.2). Dataset: HaluBench (Patronus AI), 14,900 samples across RAGTruth, HaluEval,
PubMedQA, DROP, CovidQA, and FinanceBench, split into train / calibration / test.

## Quick Start

```bash
# Phase 1 — TF-IDF baseline
python python/01_load_eda.py
python python/02_train_detector.py
python python/03_fit_crc.py
python python/04_final_evaluation.py
python python/05_baseline_comparison.py
python python/06_ablation_studies.py

# Phase 2 — BERT
python python/07_bert_detector.py
python python/08_bert_crc_comparison.py
```

## Results

Test set, Mondrian (per-domain) CRC at α = 0.10. "Recall" is the fraction of true
hallucinations flagged — the downstream classifier metric, not the conformal risk itself.

| Metric           | TF-IDF | BERT  |
|------------------|--------|-------|
| Recall           | 15.8%  | 19.2% |
| False-alarm rate | 5.0%   | 7.0%  |
| Precision        | 71.5%  | 58.0% |

BERT raises overall recall by ~3.4 points, but at the cost of more false alarms and
~13 points of precision. The gain is not uniform: it concentrates almost entirely in
RAGTruth (recall 39% → ~66%), while DROP (≈11% → 3%) and FinanceBench (16% → 12%)
regress. This tracks the base BERT detector, which is near chance overall (~54%
accuracy) and strong only on RAGTruth (~81%). The result is a trade-off rather than a
clean win — and a demonstration that Mondrian CRC delivers valid per-domain risk control
even on top of a weak detector, where the achievable recall/false-alarm operating point
is bounded by detector quality, not by the conformal layer.

## Files

- `TECHNICAL_REPORT.md` — full methodology and results
- `python/01-06_*.py` — Phase 1 pipeline (TF-IDF)
- `python/07_bert_detector.py` — BERT detector
- `python/08_bert_crc_comparison.py` — BERT vs TF-IDF under CRC
- `figures/` — eight visualizations across both phases
- `requirements.txt` — dependencies

## Citation

```bibtex
@software{HallucinationCRC2026,
  author = {Aldirawi, Tareq},
  title  = {LLM Hallucination Detection with Mondrian Conformal Risk Control},
  year   = {2026},
  url    = {https://github.com/TareqAldirawi94/llm-hallucination-crc}
}
```
