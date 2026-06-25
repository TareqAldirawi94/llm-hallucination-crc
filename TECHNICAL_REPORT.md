LLM Hallucination Detection with Mondrian Conformal Risk Control

Author: Tareq Aldirawi
Date: June 2026
Project: Domain-aware conformal risk control for LLM hallucination detection


Abstract

We study the use of Mondrian Conformal Risk Control (CRC) for detecting hallucinations
in LLM outputs. CRC is a distribution-free framework that controls the expected risk of
a set-valued predictor at a user-chosen level under exchangeability; the Mondrian variant
enforces that control within each domain rather than only on average. We instantiate
this on HaluBench (14,900 samples across six domains) with two base detectors — a
TF-IDF + logistic-regression baseline and a BERT sentence-embedding model — and report
what a valid per-domain risk-control procedure yields in each case.

The central empirical finding is a recall / false-alarm tradeoff. Because the base
detectors are close to chance on several domains, the conformal layer is valid but cannot
manufacture discriminative power that the detector lacks: pushing hallucination recall up
forces the false-alarm rate up with it. BERT raises overall recall over TF-IDF (15.8% to
19.2%) but at the cost of more false alarms and lower precision, and its gain concentrates
almost entirely in the one domain where it is genuinely accurate (RAGTruth). The
contribution is methodological and diagnostic: it shows how to bring distribution-free,
domain-conditional risk control to hallucination detection, and it characterizes the
limits that a weak base detector imposes on the achievable operating point.


1. Introduction

1.1 Motivation

Large language models generate fluent text but routinely hallucinate — produce
plausible-sounding statements that are not faithful to the provided context or to fact.
In high-stakes settings (clinical question answering, financial analysis, legal research)
an undetected hallucination can be costly. A practical detector therefore needs not only
to flag suspected hallucinations but to come with a statement about how reliable that
flagging is, and ideally a statement that holds separately for each domain rather than
only on average.

1.2 What conformal risk control does and does not give you

It is worth being precise about the guarantee, because it is easy to overstate. Conformal
risk control provides a distribution-free bound on the expected loss of a procedure:
for a chosen target level α and a bounded, monotone loss, the procedure's risk is at most
α in expectation over the exchangeable calibration/test draw. The Mondrian variant fits a
separate threshold per domain, so the bound holds conditionally on domain.

Two clarifications matter for reading the results below:


This is a guarantee in expectation, not a "with 90% confidence" high-probability
statement. The latter is the regime of RCPS / Learn-then-Test, which we do not use here.
The quantity CRC controls is the risk of the set-valued predictor, not any downstream
classification metric. In particular, the "coverage" numbers reported throughout
Section 5 are hallucination recall (fraction of true hallucinations flagged), which
is a property of the resulting classifier, not the risk that the conformal procedure
pins to α. Keeping these two senses of "coverage" distinct is essential to interpreting
the experiments correctly.


1.3 Contribution

We combine hallucination detection, conformal risk control, and Mondrian (domain-wise)
stratification, and we use the resulting procedure as a diagnostic. The valid per-domain
guarantee is the easy part; the informative part is the operating frontier it exposes,
which is governed by how good the base detector actually is on each domain.


2. Related Work

Hallucination detection. Prior work spans entailment- and consistency-based factuality
checking (Rashkin et al., 2021; Huang et al., 2021; Dziri et al., 2022) and benchmark
construction (TruthfulQA, Lin et al., 2022; FEVER, Thorne et al., 2018). These typically
produce binary or scalar outputs without a distribution-free reliability statement.

Conformal prediction and risk control. Conformal prediction provides distribution-free
prediction sets (Vovk et al., 1999; Lei & Wasserman, 2014; Barber et al., 2021). Conformal
Risk Control (Angelopoulos et al., 2024) generalizes the coverage guarantee to bounded
monotone losses, controlling E[L] ≤ α; Learn-then-Test (Angelopoulos et al., 2025) gives
the high-probability counterpart.

Mondrian / class-conditional conformal. Stratifying calibration by a discrete category
yields category-conditional guarantees (Vovk's Mondrian framework; Sadinle et al., 2019).
Here the category is the source domain.

Positioning. We are not aware of prior work that applies Mondrian CRC to hallucination
detection. The point of this project is less a new method than an honest instantiation:
what does a valid domain-conditional risk-control procedure deliver on a realistic,
multi-domain hallucination benchmark, and what limits it.


3. Method

3.1 Problem setup

Each example is a triple (passage, question, answer) with a binary label
Y ∈ {PASS (faithful), FAIL (hallucination)} and a domain label d ∈ {HaluEval, FinanceBench,
PubMedQA, CovidQA, DROP, RAGTruth}. The base detector outputs an estimate of P(FAIL | x),
and the conformal layer fits a per-domain decision threshold so that the controlled risk
is bounded at α within each domain.

3.2 Base detectors

Phase 1 — TF-IDF + logistic regression. Passage and answer are concatenated and
vectorized with unigram+bigram TF-IDF (stopwords removed, min_df=2, max_df=0.95),
giving ~185k sparse features. An L2-penalized logistic regression is trained on the
training split.

Phase 2 — BERT. Sentence embeddings (via sentence-transformers) replace the TF-IDF
representation, with the same logistic head. This tests whether a semantically richer
representation changes the operating frontier.

In both phases the data are split 60/20/20 (train / calibration / test), stratified by
domain: 8,940 / 2,980 / 2,980.

3.3 Conformity score

For calibration we use the negative log-likelihood of the true label,
s(x) = −log P̂(Y | x): the score is low when the detector is confident and correct, high
when it is uncertain or wrong.

3.4 Mondrian CRC

Standard (pooled) CRC computes one threshold from all calibration scores. The Mondrian
variant computes a separate threshold τ_d from the calibration scores of domain d only,
and applies τ_d to test examples from that domain. The effect is that the risk-control
statement holds per domain rather than being averaged across domains — which matters when
domains differ sharply in difficulty, as they do here.

3.5 Asymmetric loss

Missing a hallucination (false negative) is usually costlier than a false alarm. The loss
can be made asymmetric by an FN:FP penalty ratio, which shifts the per-domain threshold
toward catching more hallucinations. As Section 5.3 shows, this is exactly the knob that
exposes the recall/false-alarm tradeoff.


4. Experimental Setup

4.1 Dataset: HaluBench

HaluBench (Ravi et al., 2024, "Lynx", Patronus AI) is a 15k-sample benchmark of
(context, question, answer) triples labeled for hallucination, assembled from six existing
QA datasets — FinanceBench, PubMedQA, CovidQA, HaluEval, DROP, and RAGTruth — and is the
first open-source hallucination benchmark to draw on real-world domains such as finance
and medicine.

Total samples14,900FAIL (hallucination)~7,000 (47%)PASS (faithful)~7,900 (53%)Domains6

HaluEval dominates the sample count (~10.9k); the remaining five domains contribute a few
hundred to ~800 each. Splits are stratified by domain.

4.2 Baselines

We compare Mondrian CRC against three alternatives: a fixed 0.5 probability threshold (no
uncertainty quantification); a single pooled CRC threshold (no domain stratification); and
a fully example-conditional CRC (very conservative).

4.3 Metrics

Let TP, FP, FN, TN be the usual counts on flagged hallucinations.

Recall (reported as “coverage”)=#{FAIL flagged}#{total FAIL}\text{Recall (reported as ``coverage'')} = \frac{\#\{\text{FAIL flagged}\}}{\#\{\text{total FAIL}\}}Recall (reported as “coverage”)=#{total FAIL}#{FAIL flagged}​
False-alarm rate (FAR)=#{PASS flagged}#{total PASS}\text{False-alarm rate (FAR)} = \frac{\#\{\text{PASS flagged}\}}{\#\{\text{total PASS}\}}False-alarm rate (FAR)=#{total PASS}#{PASS flagged}​
Precision=#{TP}#{TP+FP}\text{Precision} = \frac{\#\{\text{TP}\}}{\#\{\text{TP} + \text{FP}\}}Precision=#{TP+FP}#{TP}​
Fairness=StdDevd(recalld)\text{Fairness} = \mathrm{StdDev}_d\big(\text{recall}_d\big)Fairness=StdDevd​(recalld​)
Terminology note. "Coverage" in the tables below is hallucination recall, defined
above. It is not the conformal coverage / risk that the CRC procedure controls (see
§1.2). The conformal guarantee is what makes the per-domain comparison meaningful; recall
and FAR are the downstream quantities we read off the resulting classifier.


5. Results

5.1 TF-IDF vs BERT (test set, Mondrian CRC)

MetricTF-IDFBERTRecall ("coverage")15.8%19.2%False-alarm rate5.0%7.0%Precision71.5%58.0%

BERT raises overall recall by ~3.4 points, but false alarms rise and precision drops ~13
points. This is a trade-off, not a clean upgrade — and the rest of this section
explains why.

5.2 The gain is not uniform across domains

Per-domain recall (read from the committed comparison figure; see
figures/08_bert_vs_tfidf_comparison.png):

DomainTF-IDFBERTRAGTruth~39%~66%FinanceBench~16%~12%DROP~11%~3%CovidQA~10%~13%PubMedQA~10%~12%HaluEval~8%~8%

Almost the entire overall improvement comes from RAGTruth. On DROP and FinanceBench BERT
actually regresses. The reason is visible in the base detector itself: BERT's accuracy
by domain (figures/07_bert_detector_performance.png) is ~81% on RAGTruth but only ~39–56%
on the other five — near chance. The confusion matrix on the calibration set is
890/624/738/728 (TN/FP/FN/TP), i.e. ~54% accuracy overall and a predicted-probability
distribution for PASS and FAIL that almost completely overlaps. A conformal layer on top
of a near-chance detector is valid, but it has nothing discriminative to work with on
those domains.

5.3 The recall / false-alarm frontier

The asymmetric-loss knob makes the trade-off explicit. As the FN:FP penalty increases, the
threshold is driven to flag more aggressively: recall climbs toward 100%, but the
false-alarm rate climbs with it, toward flagging essentially everything. There is no
penalty setting that produces both high recall and low false alarms, because the base
detector cannot separate the classes on most domains. This — not any deficiency of the
conformal layer — is the binding constraint.

The α sweep tells the same story from the coverage side: smaller α (tighter target) yields
lower recall and lower FAR; larger α yields higher recall and higher FAR, monotonically. At
α = 0.10 the TF-IDF operating point is the ~15.8% recall / 5.0% FAR reported above.

5.4 Mondrian vs pooled stratification

Against a single pooled threshold, Mondrian produces markedly more uniform recall across
domains (roughly half the cross-domain standard deviation). A pooled threshold lets the
dominant, easy domain (RAGTruth) and the dominant-by-count, hard domain (HaluEval) pull
the single threshold in opposite directions, leaving some domains far from target.
Mondrian removes that coupling — which is the entire point of stratifying.


6. Discussion

6.1 What works

The conformal layer does its job: it gives a distribution-free, per-domain risk-control
statement with no distributional assumptions, and Mondrian stratification delivers the
intended uniformity across domains. The thresholds are learned automatically per domain
and are simple, interpretable scalars.

6.2 The real limitation is the base detector, not the conformal layer

The honest reading of these results is that CRC is valid but cannot rescue a weak
detector. With a base model near chance on five of six domains, the achievable operating
points all lie on a steep recall/FAR frontier; you can move along it (via α or the loss
ratio) but you cannot escape it. The recall numbers are low not because the guarantee
"failed" — recall is not the controlled quantity — but because separating hallucinated
from faithful answers from surface features alone is genuinely hard on these domains.

This reframes the path forward. The lever that matters most is detector quality on the
hard domains (better embeddings, fine-tuning on hallucination examples, ensembling), not
further tuning of the conformal layer. BERT was the first step in that direction; its
selective success on RAGTruth and failure elsewhere shows the lever is real but that
off-the-shelf embeddings are not sufficient.

6.3 Other limitations

HaluBench is constructed by perturbing existing QA datasets and may not match the
distribution of hallucinations produced by deployed LLMs; the exchangeability assumption
underlying the guarantee is between calibration and test within this benchmark, not a claim
about real traffic. The TF-IDF representation is purely lexical. And HaluEval's dominance of
the sample count means aggregate numbers are effectively HaluEval numbers unless read per
domain.


7. Conclusion

We applied Mondrian Conformal Risk Control to LLM hallucination detection on HaluBench, with
TF-IDF and BERT base detectors. The conformal layer provides a valid, distribution-free,
domain-conditional risk-control guarantee, and Mondrian stratification equalizes behavior
across domains as intended. The substantive finding is diagnostic: the achievable
recall/false-alarm operating points are governed by the base detector, which is near chance
on most domains, so high recall is only reachable at high false-alarm cost. BERT improves
matters only where it is genuinely accurate (RAGTruth) and regresses elsewhere. The clear
implication for future work is that progress depends on better base detectors on the hard
domains rather than on the conformal machinery, which is already doing what it guarantees.


References

Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal Risk
Control. International Conference on Learning Representations (ICLR). arXiv:2208.02814.

Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2025). Learn then
Test: Calibrating predictive algorithms to achieve risk control. Annals of Applied
Statistics, 19(2), 1641–1662.

Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021). Predictive inference
with the jackknife+. Annals of Statistics, 49(1), 486–507.

Dziri, N., Milton, S., Yu, M., Zaiane, O., & Reddy, S. (2022). On the Origin of
Hallucinations in Conversational Models: Is it the Datasets or the Models? NAACL-HLT,
5271–5285.

Huang, L., et al. (2021). [Hallucination in neural abstractive summarization. — confirm
exact title/venue before listing on the CV.]

Lei, J., & Wasserman, L. (2014). Distribution-free prediction bands for non-parametric
regression. Journal of the Royal Statistical Society: Series B, 76(1), 71–96.

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human
Falsehoods. Association for Computational Linguistics (ACL).

Rashkin, H., et al. (2021). [Faithfulness / attribution in language generation. — confirm
exact title/venue before listing on the CV.]

Ravi, S. S., Mielczarek, B., Kannappan, A., Kiela, D., & Qian, R. (2024). Lynx: An Open
Source Hallucination Evaluation Model. arXiv:2407.08488. (HaluBench dataset.)

Sadinle, M., Lei, J., & Wasserman, L. (2019). Least Ambiguous Set-Valued Classifiers with
Bounded Error Levels. Journal of the American Statistical Association, 114(525), 223–234.

Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: a Large-scale
Dataset for Fact Extraction and VERification. NAACL-HLT.

Vovk, V., Gammerman, A., & Saunders, C. (1999). Machine-learning applications of
algorithmic randomness. International Conference on Machine Learning (ICML), 444–453.


Appendix: Implementation Details

The code below is the Phase 1 (TF-IDF) pipeline. Phase 2 swaps the TF-IDF vectorizer for a
sentence-transformers encoder; the CRC and evaluation logic (C–D) is identical.

A. Data preprocessing

python# Combine passage and answer
x_combined = passage + " " + answer

# TF-IDF vectorization
tfidf = TfidfVectorizer(
    max_features=None,      # keep all features
    min_df=2,               # token must appear in >= 2 docs
    max_df=0.95,            # token in <= 95% of docs
    ngram_range=(1, 2),     # unigrams + bigrams
    lowercase=True,
    stop_words='english',
)

X_train = tfidf.fit_transform(train_texts)   # 8940 x 185059
X_calib = tfidf.transform(calib_texts)       # 2980 x 185059
X_test  = tfidf.transform(test_texts)        # 2980 x 185059

B. Logistic regression

pythonfrom sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=1000,
    solver='saga',     # handles large sparse matrices
    random_state=42,
    n_jobs=-1,
)
lr.fit(X_train, y_train)
y_proba = lr.predict_proba(X_calib)   # (2980, 2)

C. Per-domain CRC thresholds

pythonimport numpy as np

alpha = 0.10
thresholds = {}

for d in domains:
    mask = (calib['source_ds'] == d)
    proba_true = y_proba_calib[mask, true_label_idx[mask]]
    sigma = -np.log(np.maximum(proba_true, 1e-10))   # conformity scores

    n_d = mask.sum()
    q = np.ceil((n_d + 1) * (1 - alpha)) / n_d        # conservative level
    thresholds[d] = np.quantile(sigma, min(q, 1.0))

D. Test-set evaluation

pythonproba_true_test = y_proba_test[np.arange(len(test)), true_label_idx_test]
sigma_test = -np.log(np.maximum(proba_true_test, 1e-10))
tau = np.array([thresholds[d] for d in test['source_ds']])

flagged = sigma_test >= tau
is_fail = (test['label'] == 'FAIL').values

# Recall ("coverage") per domain
for d in domains:
    m = (test['source_ds'] == d).values
    recall_d = (flagged & is_fail & m).sum() / (is_fail & m).sum()

E. Requirements and runtime

pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
scipy==1.11.0
matplotlib==3.8.0
seaborn==0.13.0
torch>=2.2.0
transformers>=4.40.0
sentence-transformers>=2.7.0

Phase 1 (Blocks 1–6) runs in roughly an hour on CPU. Phase 2 (Blocks 7–8) adds the cost of
encoding ~15k texts with BERT, which dominates its runtime and benefits from a GPU.

Code: https://github.com/TareqAldirawi94/llm-hallucination-crc
Dataset: https://huggingface.co/datasets/PatronusAI/HaluBench
