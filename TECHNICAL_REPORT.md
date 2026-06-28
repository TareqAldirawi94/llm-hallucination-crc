# LLM Hallucination Detection with Mondrian Conformal Risk Control

**Author:** Tareq Aldirawi
**Date:** June 2026
**Method:** Mondrian Conformal Risk Control (per-domain false-negative-rate control)

---

## Abstract

We apply Mondrian Conformal Risk Control (CRC) to control the per-domain false-negative
rate of a hallucination detector on HaluBench, and use the resulting valid guarantee as a
diagnostic of the detector itself. Two base detectors are compared: TF-IDF with logistic
regression, and BERT sentence embeddings. The conformal procedure is valid in the sense
that matters: at a 90% recall target it controls held-out recall to 85.9%–95.0% in every
one of six domains. The substantive findings are about what that guarantee costs and why.
First, the cost in false alarms is near-maximal (mean false-alarm rate ~88–92%), because
the detectors are statistically near random *within* each domain — mean per-domain ROC AUC
is 0.537 for TF-IDF and 0.506 for BERT. Second, the value of stratification is uniformity,
not detection: at matched ~90% recall, per-domain recall has standard deviation ~3 pp under
Mondrian against ~27–30 pp under a single pooled threshold. Third, the apparent pooled
signal (AUC 0.57–0.60) is largely domain identity rather than hallucination; conditioning
on domain removes it and exposes the ~0.5 within-domain signal, and the more semantic
detector (BERT) exhibits the larger pooled-to-conditional gap while detecting no better.
The conformal layer works; the base detector is the bottleneck. A four-approach
investigation — lexical, semantic, relational, and zero-shot entailment detectors — confirms
this is a property of the task and not of one feature choice: none separates hallucinations
within domain, and the only above-chance results are construction artifacts or confined to
surface-knowledge domains.

---

## 1. Introduction

Large language models hallucinate — they produce fluent statements unsupported by the
provided context or by fact. A practical detector should not only flag suspected
hallucinations but come with a reliability statement, and ideally one that holds separately
for each domain rather than only on average, since domains differ in difficulty and base
rate. Conformal risk control provides a distribution-free guarantee of this kind: for a
chosen level alpha and a bounded, monotone loss, it controls the expected loss at alpha
under exchangeability. The Mondrian variant fits a separate threshold per domain, so the
control holds conditionally on domain.

It is important to be precise about what such a guarantee does and does not deliver. The
control is on the risk in expectation, not a high-probability statement, and it is a
property of the procedure, not of any single downstream metric. Throughout this report the
controlled quantity is the per-domain false-negative rate (FNR); the recall, false-alarm
rate, and precision we tabulate are downstream properties of the resulting classifier. The
guarantee is what makes the per-domain comparison meaningful, and the cost of the guarantee
is what tells us about the detector.

---

## 2. Method

The base detector outputs a label-free hallucination score, s(x) = P(FAIL | x). This score
uses no true label, so it is computed identically on calibration and test data, and it
points in the natural direction — higher means more likely a hallucination. For each domain
d we collect the calibration scores of the true hallucinations (the FAIL examples) and set
the threshold tau_d to their conformal lower-quantile at rank k = floor(alpha (m+1)), where
m is the number of calibration FAILs in that domain. A test example is flagged when
s(x) >= tau_d. Because at most a fraction alpha of calibration FAILs fall below tau_d, and
the calibration and test FAILs are exchangeable, a new FAIL falls below the threshold with
probability at most alpha; equivalently, recall is at least 1 − alpha within each domain.
Lowering tau_d to catch more hallucinations also flags more faithful answers, so the
false-alarm rate rises with recall; how steeply depends on how well the score separates the
two classes in that domain, which is exactly the detector property the rest of the report
measures.

This corrects two errors that are easy to make and that the early version of this project
contained. The score must not be the negative log-likelihood of the *true* label, because
that requires the label at test time and inverts the flag direction; and the threshold must
be calibrated on the FAIL subpopulation to control FNR, rather than set at a fixed quantile
of all scores, which merely fixes the flag rate and cannot deliver the guarantee.

---

## 3. Experimental Setup

HaluBench (Ravi et al., 2024, "Lynx", Patronus AI) provides 14,900 (context, question,
answer) triples labeled PASS or FAIL, assembled from six existing QA datasets: halueval
(~10,000), and DROP, pubmedQA, FinanceBench, covidQA, and RAGTruth (roughly 900–1,000
each). The label balance is about 48% FAIL. We split 60/20/20 into train, calibration, and
test, stratified by domain (8,940 / 2,980 / 2,980), with a fixed seed so the split is
reproducible. The TF-IDF detector uses unigram and bigram features with an L2-penalized
logistic regression; the BERT detector replaces the representation with
all-MiniLM-L6-v2 sentence embeddings and the same logistic head. Both refit with a fixed
seed so test scores match the calibration model.

---

## 4. Results

### 4.1 The per-domain guarantee holds out-of-sample

Applying the per-domain thresholds to the held-out test set, recall lands close to the 90%
target in every domain: DROP 91.1%, FinanceBench 90.7%, RAGTruth 95.0%, covidQA 91.7%,
halueval 89.2%, pubmedQA 85.9%, for an overall pooled recall of 89.6%. The departures from
exactly 90% are finite-sample noise, not bias; recall does not collapse on test, which is
the meaningful check that the guarantee transfers from calibration to test.

### 4.2 The cost, and why it is high

The false-alarm rate that accompanies this recall is near-maximal — per domain it ranges
from 83.2% to 95.7%, overall 87.8%, with precision around 49%. The reason is visible in the
detector's separating power. Sweeping alpha traces each domain's recall/false-alarm
frontier, and that frontier is the ROC curve; the per-domain ROC AUC is the single number
that says whether any usable operating point exists. For TF-IDF the per-domain AUCs are
0.668 (DROP), 0.550 (FinanceBench), 0.548 (RAGTruth), 0.549 (halueval), 0.482 (pubmedQA),
and 0.423 (covidQA), with a mean of 0.537. These are close to 0.5, and one is below it. A
score that barely separates the classes leaves no threshold that achieves high recall at
low false alarms; the conformal procedure can only place the operating point honestly on a
near-diagonal curve.

### 4.3 Mondrian's value is uniformity, not detection

Because the detector caps detection power, the contribution of Mondrian stratification is
not a better aggregate number — at matched recall it cannot be, on a near-random detector.
It is per-domain validity. Holding recall fixed at the same ~90% target, a single pooled
FAIL-quantile threshold achieves 88.9% recall but with per-domain standard deviation
27.0 pp, while Mondrian achieves 89.6% with standard deviation 2.7 pp. The pooled false-
alarm rate is, if anything, slightly higher under Mondrian (87.8% versus 81.8%): there is
no free lunch, and Mondrian does not claim one. The clearest illustration is RAGTruth, whose
score scale differs from the pool: under the single pooled threshold it catches 17.5% of its
hallucinations, while under its own per-domain threshold it catches 95.0%. Same data, same
detector; stratification is the entire difference between a guarantee that holds for that
domain and one that abandons it.

### 4.4 Finite-sample reliability

The guarantee is exact only asymptotically, and its reliability depends on the number of
calibration FAILs available per domain. Subsampling the calibration FAILs and measuring how
far test recall departs from the target, the mean absolute deviation shrinks monotonically
from 6.4 pp at 10% of the calibration data to 2.2 pp at the full set, and the run-to-run
variance collapses as the sample grows. This is the finite-sample behavior of the
procedure, and it explains why the smallest domains — RAGTruth has only about 26
calibration FAILs — are the least stable.

### 4.5 TF-IDF versus BERT, and the domain-identity gap

Replacing TF-IDF with BERT embeddings does not change the picture, and clarifies it. Under
identical Mondrian control on the test set, BERT's mean per-domain AUC is 0.506 — at or
below TF-IDF's 0.537, and indistinguishable from random. Recall remains controlled (mean
94.0%) at near-maximal false alarms (mean 92.4%). The instructive contrast is between
pooled and per-domain AUC. TF-IDF's pooled AUC is 0.568 against a per-domain mean of 0.537;
BERT's pooled AUC is 0.601 against a per-domain mean of 0.506. In both cases the pooled
number overstates the within-domain signal, because a score that encodes which domain a
sample belongs to earns pooled AUC from the domains' differing base rates and styles
without detecting any hallucination. Conditioning on domain, as Mondrian does, removes that
artifact and exposes the true ~0.5 signal. BERT, being semantic, encodes domain identity
more strongly, which is precisely why its pooled-to-conditional gap is the larger of the
two while its within-domain detection is no better. The only thing either detector reliably
learns on HaluBench is which domain a sample comes from.

### 4.6 Why the detector is the bottleneck: four approaches all fail

The preceding sections attribute the high false-alarm cost to a near-random base detector.
To establish that this is a property of the task on HaluBench rather than an artifact of one
feature choice, we evaluated four qualitatively different detectors and measured each one's
within-domain separating power (per-domain test ROC AUC).

The lexical detector (TF-IDF with logistic regression) has mean per-domain AUC 0.537, and
the semantic detector (MiniLM sentence embeddings of passage and answer) 0.506 — both near
random, as already reported. A relational detector embeds passage and answer separately and
feeds the logistic head only their interaction (absolute difference, product, cosine) plus
the lexical overlap of answer tokens with the passage, deliberately excluding the raw
embeddings that carry domain identity. Its mean per-domain AUC is about 0.60, but an
ablation shows the lift is largely a construction artifact. Removing lexical overlap, the
embedding-only relational features fall to roughly chance on most domains; the apparent
improvement concentrates almost entirely on HaluEval, whose hallucinations were built by
editing the answer's lexical overlap with the passage. Overlap is therefore a near-giveaway
for that benchmark's construction method rather than a transferable faithfulness signal, and
HaluEval's FAIL and PASS overlap distributions are widely separated where other domains' are
not.

The fourth detector is a zero-shot natural-language-inference cross-encoder that scores
whether the passage entails the answer, with score s(x) = 1 − P(entailment). Because it has
never seen HaluBench, it cannot exploit the benchmark's construction. Its mean per-domain AUC
is about 0.56, and the per-domain pattern is the decisive part: it lifts only the
surface-knowledge domains (HaluEval ≈ 0.77, pubmedQA ≈ 0.64) and is at or below chance on the
genuinely retrieval-grounded domains where entailment should excel — RAGTruth, covidQA,
FinanceBench, and DROP all near 0.46–0.53. HaluBench's hard cases are fluent, on-topic
answers that are entailment-plausible yet factually wrong, which sentence-level entailment
does not catch.

The four approaches agree: no off-the-shelf detector — lexical, semantic, relational, or
entailment-based — separates HaluBench hallucinations within domain. Every above-chance
result is either a construction artifact or confined to surface-knowledge domains, and
neither would transfer to deployed retrieval-augmented outputs. Detecting these hallucinations
requires a model fine-tuned on the task, which is precisely the route the benchmark's own
authors take, fine-tuning a dedicated evaluator rather than relying on zero-shot or
feature-engineered detectors. This makes the central claim a measured property rather than an
assertion: the conformal guarantee's cost is governed by detector quality, and detector
quality on HaluBench is low robustly across four distinct detection strategies.

---

## 5. Discussion

The result separates two questions that are easy to conflate. Is the conformal procedure
valid? Yes: the per-domain false-negative rate is controlled, on held-out data, for both
detectors, and its reliability improves with calibration size in the expected way. Is the
detector useful? No: its within-domain AUC is near 0.5 for both lexical and semantic
features, so the price of any useful recall is a near-maximal false-alarm rate. A weak
detector wrapped in a valid guarantee is still a weak detector, and the contribution of
this project is to make that cost explicit and honest rather than to hide it behind an
aggregate number that benefits from domain leakage.

The practical implication is that progress depends on the base detector, not on the
conformal layer or the choice of alpha. Off-the-shelf sentence embeddings are not enough;
a deployable system would need a detector built for this task — fine-tuned on hallucination
supervision, or using features that actually distinguish faithful from fabricated content
within a domain. The conformal machinery is already doing its job and will continue to, on
whatever better detector replaces these.

That the detector is the bottleneck is not a conclusion from one feature set but a robust
one. Four distinct strategies — lexical, semantic, relational, and zero-shot entailment —
all land at within-domain AUC near 0.5 to 0.56, and the apparent exceptions are instructive
rather than encouraging. The relational detector's gain is mostly a lexical-overlap shortcut
tied to how HaluEval was constructed, and the entailment detector helps only on
surface-knowledge domains while failing on the retrieval-grounded ones where it should be
strongest. The hard hallucinations here are fluent and entailment-plausible but factually
wrong, and catching them requires task-specific fine-tuning — the same conclusion the
benchmark's authors reached. The value of running the conformal analysis on top of these
detectors is that it makes the bottleneck precise and honest instead of letting an
aggregate number obscure it.

A methodological note: the pooled-versus-conditional AUC gap is itself an argument for
per-domain analysis beyond fairness. The pooled metric is optimistic precisely because it
absorbs domain identity, so conditioning on domain is the only honest way to measure the
detector. Stratification here is not just a fairness device; it is the correct measurement.

---

## 6. Conclusion

Mondrian Conformal Risk Control delivers a valid, distribution-free, finite-sample
guarantee on the per-domain false-negative rate of a hallucination detector, demonstrated
on held-out HaluBench data and characterized as the calibration set grows. Its value is
per-domain uniformity — recall standard deviation of ~3 pp against ~27–30 pp for a single
threshold — rather than improved detection, which it cannot provide on a near-random
detector. Both TF-IDF and BERT detectors are near random within domain (mean per-domain AUC
0.537 and 0.506); their apparently higher pooled AUC reflects domain identity, not
hallucination, and conditioning on domain exposes the true signal. A four-approach
investigation — lexical, semantic, relational, and zero-shot entailment detectors — finds
the same near-random within-domain separation throughout, with the only above-chance results
explained by construction artifacts or limited to surface-knowledge domains; detecting these
hallucinations needs a task-specific fine-tuned model. The conformal layer is sound; the base
detector is the bottleneck, and the path forward is a better detector, not a different
guarantee.

---

## Reproducibility

```
00_download_halubench.py   download HaluBench -> halubench_raw.csv (once)
01_load_eda.py             EDA + stratified 60/20/20 splits
02_train_detector.py       TF-IDF + logistic regression; score = P(FAIL|x)
03_fit_crc.py              per-domain Mondrian FNR thresholds
03b_alpha_sweep.py         recall/FAR frontier + per-domain AUC
04_final_evaluation.py     held-out test evaluation + FINAL_REPORT.md
05_baseline_comparison.py  matched-recall comparison (uniformity)
06_ablation_studies.py     alpha, calibration size, Mondrian vs global
07_bert_detector.py        BERT (MiniLM) detector + per-domain AUC
08_bert_crc_comparison.py  TF-IDF vs BERT under CRC
09_faithfulness_detector.py   relational (passage/answer interaction) detector
09b_overlap_ablation.py       isolates the lexical-overlap construction artifact
10_nli_detector.py            zero-shot NLI cross-encoder (entailment) detector
```

All scores are the label-free s(x) = P(FAIL | x); all thresholds are calibrated on
calibration-set FAIL scores and evaluated on the held-out test set. Splits use a fixed seed.

## References

Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal Risk
Control. *International Conference on Learning Representations*. arXiv:2208.02814.

Ravi, S. S., Mielczarek, B., Kannappan, A., Kiela, D., & Qian, R. (2024). Lynx: An Open
Source Hallucination Evaluation Model. arXiv:2407.08488. (HaluBench dataset.)

Sadinle, M., Lei, J., & Wasserman, L. (2019). Least Ambiguous Set-Valued Classifiers with
Bounded Error Levels. *Journal of the American Statistical Association*, 114(525), 223–234.

Vovk, V., Gammerman, A., & Saunders, C. (1999). Machine-learning applications of
algorithmic randomness. *International Conference on Machine Learning*, 444–453.
