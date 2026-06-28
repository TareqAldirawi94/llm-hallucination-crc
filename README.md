# LLM Hallucination Detection with Mondrian Conformal Risk Control

Per-domain conformal false-negative-rate (FNR) control for hallucination detection on
HaluBench, with two base detectors compared: TF-IDF + logistic regression (Phase 1) and
BERT sentence embeddings (Phase 2).

## What this project shows

The conformal layer does exactly what it promises: it controls the false-negative rate
*within each domain* at a chosen level, validated on held-out test data. The interesting
part is what that valid guarantee reveals about the detector underneath it.

1. **The per-domain guarantee holds out-of-sample.** At a 90% recall target, held-out
   recall lands at 85.9%–95.0% in every one of the six domains (overall 89.6%).
2. **Mondrian's value is per-domain uniformity, not better detection.** Holding recall
   fixed at the same ~90% target, per-domain recall has standard deviation **~3 pp** under
   Mondrian versus **~27–30 pp** under a single pooled threshold. On RAGTruth the pooled
   threshold catches **17.5%** of hallucinations where Mondrian catches **95.0%** — same
   data, same detector, the only difference is stratification.
3. **Both detectors are near random *within* domain.** Mean per-domain ROC AUC is **0.537**
   (TF-IDF) and **0.506** (BERT). Semantic embeddings do not separate hallucinated from
   faithful answers any better than lexical features on this benchmark.
4. **The apparent signal is domain identity, not hallucination.** Pooled AUC (0.568 TF-IDF,
   0.601 BERT) sits well above per-domain AUC. The score is partly predicting *which
   domain* a sample is from, not whether it is a hallucination. Conditioning on domain —
   which Mondrian does — strips that away and exposes the ~0.5 within-domain signal. BERT's
   pooled-to-conditional gap is the larger of the two: the semantic model leaks more domain
   identity while detecting no better.

5. **The bottleneck is the detector, and that is robust.** Four detection strategies —
   lexical (TF-IDF), semantic (BERT), relational (passage/answer interaction), and zero-shot
   entailment (NLI cross-encoder) — all land at within-domain AUC ~0.5–0.56. The only
   above-chance results are a lexical-overlap shortcut tied to how HaluEval was built, or
   confined to surface-knowledge domains; entailment fails on exactly the retrieval-grounded
   domains where it should be strongest. Detecting these hallucinations needs a task-specific
   fine-tuned model, not an off-the-shelf detector.

The honest conclusion: **Mondrian CRC is a valid, finite-sample, domain-conditional
guarantee, and its cost (false alarms) is set entirely by the base detector — which is
near random here for both feature sets. The conformal machinery works; the detector is the
bottleneck, and off-the-shelf embeddings do not fix it.**

## Results (test set, alpha = 0.10)

Per-domain, TF-IDF detector under Mondrian FNR control:

| Domain | Recall | FAR | Precision | AUC |
|---|---|---|---|---|
| DROP | 91.1% | 94.5% | 44.1% | 0.668 |
| FinanceBench | 90.7% | 83.5% | 50.6% | 0.550 |
| RAGTruth | 95.0% | 93.6% | 22.5% | 0.548 |
| covidQA | 91.7% | 95.7% | 52.9% | 0.423 |
| halueval | 89.2% | 86.4% | 51.4% | 0.549 |
| pubmedQA | 85.9% | 83.2% | 50.3% | 0.482 |

TF-IDF vs BERT, identical Mondrian FNR control:

| | TF-IDF | BERT |
|---|---|---|
| Pooled ROC AUC | 0.568 | 0.601 |
| Mean per-domain AUC | 0.537 | 0.506 |
| Mean recall | 90.6% | 94.0% |
| Mean FAR | 89.5% | 92.4% |

Recall is controlled to target for both; the high FAR is the cost of forcing that recall
out of a near-random detector. Neither is deployable as-is — and that is a measured result,
not a guess.

Matched-recall comparison (both at ~90% recall):

| Method | Recall | Per-domain recall std | FAR |
|---|---|---|---|
| Pooled FAIL-quantile (one threshold) | 88.9% | 27.0 pp | 81.8% |
| Mondrian CRC (per-domain) | 89.6% | 2.7 pp | 87.8% |

## Quick start

```bash
python python/00_download_halubench.py   # one-time: HaluBench -> halubench_raw.csv
python python/01_load_eda.py             # EDA + stratified 60/20/20 splits
python python/02_train_detector.py       # TF-IDF + logistic regression, score = P(FAIL|x)
python python/03_fit_crc.py              # per-domain Mondrian FNR thresholds
python python/03b_alpha_sweep.py         # recall/FAR frontier + per-domain AUC (diagnostic)
python python/04_final_evaluation.py     # held-out test evaluation + FINAL_REPORT.md
python python/05_baseline_comparison.py  # matched-recall comparison (uniformity)
python python/06_ablation_studies.py     # alpha, calibration size, Mondrian vs global
python python/07_bert_detector.py        # BERT (MiniLM) detector + per-domain AUC
python python/08_bert_crc_comparison.py  # TF-IDF vs BERT under CRC
python python/09_faithfulness_detector.py  # relational passage/answer features
python python/09b_overlap_ablation.py      # isolates the lexical-overlap artifact
python python/10_nli_detector.py           # zero-shot NLI entailment detector
```

Scripts use relative paths; run from the project root (the folder containing `data/`).

## Method in one paragraph

The score is the detector's label-free hallucination probability, s(x) = P(FAIL | x),
computed identically on calibration and test. For each domain, the threshold tau_d is the
conformal lower-quantile of that domain's calibration FAIL scores at rank floor(alpha(m+1));
a test example is flagged when s(x) >= tau_d. By exchangeability this controls the
false-negative rate at alpha within each domain (Mondrian stratification). The reported
recall, false-alarm rate, and precision are downstream properties of the resulting
classifier; the conformal guarantee is on the controlled risk, not on those metrics.

## Files

- `TECHNICAL_REPORT.md` — full methodology and results
- `FINAL_REPORT.md` — auto-generated test-set results (Block 4)
- `python/00_download_halubench.py` — dataset download (CSV)
- `python/01–06_*.py` — TF-IDF pipeline (load, train, fit CRC, evaluate, compare, ablate)
- `python/03b_alpha_sweep.py` — alpha-sweep / per-domain AUC diagnostic
- `python/07–08_*.py` — BERT detector and TF-IDF-vs-BERT comparison
- `python/09–10_*.py` — detector investigation: relational features, overlap ablation, NLI
- `figures/` — per-block visualizations
- `requirements.txt` — dependencies

## Dataset

HaluBench (Patronus AI), 14,900 (context, question, answer) triples labeled PASS/FAIL,
across six domains (halueval, DROP, pubmedQA, FinanceBench, covidQA, RAGTruth). Downloaded
from `https://huggingface.co/datasets/PatronusAI/HaluBench`.

## Citation

```bibtex
@software{HallucinationCRC2026,
  author = {Aldirawi, Tareq},
  title  = {LLM Hallucination Detection with Mondrian Conformal Risk Control},
  year   = {2026},
  url    = {https://github.com/TareqAldirawi94/llm-hallucination-crc}
}
```

## References

- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal Risk
  Control. *ICLR*. arXiv:2208.02814.
- Sadinle, M., Lei, J., & Wasserman, L. (2019). Least Ambiguous Set-Valued Classifiers with
  Bounded Error Levels. *JASA*, 114(525), 223–234.
- Ravi, S. S., Mielczarek, B., Kannappan, A., Kiela, D., & Qian, R. (2024). Lynx: An Open
  Source Hallucination Evaluation Model. arXiv:2407.08488. (HaluBench dataset.)
