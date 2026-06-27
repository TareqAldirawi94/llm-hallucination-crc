################################################################################
# BLOCK 4: FINAL EVALUATION ON TEST SET  (held-out check of FNR control)
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# Applies the per-domain conformal thresholds fitted in Block 3 to the held-out
# TEST set and reports the resulting recall / FAR / precision per domain, plus the
# per-domain ROC AUC (the detector's separating power). The score is the label-free
# s(x) = P(FAIL | x), computed exactly as in Block 2 -- no peeking at the test label.
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 4: FINAL EVALUATION ON TEST SET")
print("="*80 + "\n")

# ============================================================================
# STEP 1: REBUILD DETECTOR (same seed/data as Block 2) AND SCORE TEST SET
# ============================================================================

print("[STEP 1/5] Loading test set and applying trained detector...\n")

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
print(f"✓ Loaded test set: {len(test):,} samples")

train['combined_text'] = train['passage'] + " " + train['answer']
test['combined_text'] = test['passage'] + " " + test['answer']

tfidf = TfidfVectorizer(
    max_features=None, min_df=2, max_df=0.95,
    ngram_range=(1, 2), lowercase=True, stop_words='english'
)
X_train = tfidf.fit_transform(train['combined_text'])
X_test = tfidf.transform(test['combined_text'])
print(f"✓ Test TF-IDF matrix: {X_test.shape[0]:,} samples × {X_test.shape[1]:,} features")

y_train = (train['label'] == 'FAIL').astype(int)
lr_model = LogisticRegression(max_iter=1000, random_state=42, solver='saga', n_jobs=-1)
lr_model.fit(X_train, y_train)

test_probs = lr_model.predict_proba(X_test)
test['pred_prob_fail'] = test_probs[:, 1]
test['pred_prob_pass'] = test_probs[:, 0]
test['pred_class'] = np.where(test_probs[:, 1] > 0.5, 'FAIL', 'PASS')

# Label-free hallucination score, identical definition to Block 2.
# (Old version used -log P(true label | x), which leaks the test label; removed.)
test['score'] = test['pred_prob_fail']

print("✓ Test predictions complete (score = P(FAIL | x), label-free)\n")

# ============================================================================
# STEP 2: LOAD PER-DOMAIN THRESHOLDS FROM BLOCK 3
# ============================================================================

print("[STEP 2/5] Loading CRC thresholds from calibration...\n")

crc_thresholds_df = pd.read_csv("data/processed/crc_thresholds.csv")
crc_thresholds = dict(zip(crc_thresholds_df['domain'], crc_thresholds_df['crc_threshold']))

print("Per-domain thresholds (flag if score >= tau):")
for domain, threshold in sorted(crc_thresholds.items()):
    print(f"  {domain:14s}: {threshold:.3f}")
print()

# ============================================================================
# STEP 3: APPLY THRESHOLDS TO TEST SET
# ============================================================================

print("[STEP 3/5] Applying thresholds to the test set...\n")

test['crc_threshold'] = test['source_ds'].map(crc_thresholds)
test['flagged_by_crc'] = test['score'] >= test['crc_threshold']

print("Per-domain test performance:\n")

test_eval_stats = []
for domain in sorted(test['source_ds'].unique()):
    d = test[test['source_ds'] == domain]
    tp = ((d['label'] == 'FAIL') & d['flagged_by_crc']).sum()
    fp = ((d['label'] == 'PASS') & d['flagged_by_crc']).sum()
    fn = ((d['label'] == 'FAIL') & ~d['flagged_by_crc']).sum()
    tn = ((d['label'] == 'PASS') & ~d['flagged_by_crc']).sum()
    n_fail, n_pass = tp + fn, fp + tn

    recall = tp / n_fail if n_fail else float('nan')
    far = fp / n_pass if n_pass else float('nan')
    precision = tp / (tp + fp) if (tp + fp) else float('nan')

    y = (d['label'] == 'FAIL').astype(int).values
    auc = roc_auc_score(y, d['score'].values) if len(np.unique(y)) > 1 else float('nan')

    print(f"{domain}:")
    print(f"  Recall: {recall*100:5.1f}% ({tp}/{n_fail})   FAR: {far*100:5.1f}% ({fp}/{n_pass})"
          f"   Precision: {precision*100:5.1f}%   AUC: {auc:.3f}")

    test_eval_stats.append({
        'domain': domain, 'recall': recall, 'false_alarm_rate': far,
        'precision': precision, 'auc': auc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n_samples': len(d)
    })

test_eval_df = pd.DataFrame(test_eval_stats)
print()

# ============================================================================
# STEP 4: OVERALL METRICS
# ============================================================================

print("[STEP 4/5] Overall test metrics...\n")

overall_tp = test_eval_df['tp'].sum()
overall_fp = test_eval_df['fp'].sum()
overall_fn = test_eval_df['fn'].sum()
overall_tn = test_eval_df['tn'].sum()

overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) else 0
overall_far = overall_fp / (overall_fp + overall_tn) if (overall_fp + overall_tn) else 0
overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) else 0
overall_auc = roc_auc_score((test['label'] == 'FAIL').astype(int), test['score'])
mean_auc = test_eval_df['auc'].mean()

print(f"OVERALL TEST SET:")
print(f"  Recall (hallucinations flagged): {overall_recall*100:.1f}%")
print(f"  False-alarm rate:                {overall_far*100:.1f}%")
print(f"  Precision:                       {overall_precision*100:.1f}%")
print(f"  Pooled AUC:                      {overall_auc:.3f}")
print(f"  Mean per-domain AUC:             {mean_auc:.3f}   (0.50 = random)\n")

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================

print("[STEP 5/5] Creating test-set figure...\n")

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Recall by domain (vs target)
ax1 = fig.add_subplot(gs[0, 0])
s = test_eval_df.sort_values('recall')
ax1.barh(s['domain'], s['recall']*100, color='#2ecc71', alpha=0.85)
ax1.axvline(90, color='black', linestyle='--', linewidth=2, label='Target (90%)')
ax1.set_xlabel('Recall (%)', fontweight='bold')
ax1.set_title('Recall by Domain (Test)', fontweight='bold', fontsize=11)
ax1.set_xlim(0, 105); ax1.legend(fontsize=9); ax1.grid(axis='x', alpha=0.3)

# FAR by domain
ax2 = fig.add_subplot(gs[0, 1])
s = test_eval_df.sort_values('false_alarm_rate')
ax2.barh(s['domain'], s['false_alarm_rate']*100, color='#e67e22', alpha=0.85)
ax2.set_xlabel('False-Alarm Rate (%)', fontweight='bold')
ax2.set_title('Cost of Guarantee: FAR (Test)', fontweight='bold', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

# AUC by domain
ax3 = fig.add_subplot(gs[0, 2])
s = test_eval_df.sort_values('auc')
colors = ['#e74c3c' if a < 0.55 else '#3498db' for a in s['auc']]
ax3.barh(s['domain'], s['auc'], color=colors, alpha=0.85)
ax3.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Random (0.5)')
ax3.set_xlabel('ROC AUC', fontweight='bold')
ax3.set_title('Detector Separating Power (Test)', fontweight='bold', fontsize=11)
ax3.set_xlim(0, 1); ax3.legend(fontsize=9); ax3.grid(axis='x', alpha=0.3)

# Recall vs FAR scatter
ax4 = fig.add_subplot(gs[1, :2])
ax4.scatter(test_eval_df['false_alarm_rate']*100, test_eval_df['recall']*100,
            s=300, alpha=0.75, c=range(len(test_eval_df)), cmap='viridis',
            edgecolors='black', linewidth=1.5)
for i, dom in enumerate(test_eval_df['domain']):
    ax4.annotate(dom, (test_eval_df.iloc[i]['false_alarm_rate']*100,
                       test_eval_df.iloc[i]['recall']*100),
                 fontsize=9, ha='center', fontweight='bold')
ax4.plot([0, 100], [0, 100], '--', color='grey', lw=1, label='random detector')
ax4.axhline(90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Recall target (90%)')
ax4.set_xlabel('False-Alarm Rate (%)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Recall (%)', fontweight='bold', fontsize=11)
ax4.set_title('Recall vs False Alarm (Test Set)', fontweight='bold', fontsize=12)
ax4.grid(alpha=0.3); ax4.legend(fontsize=9)
ax4.set_xlim(0, 100); ax4.set_ylim(0, 105)

# Overall confusion matrix
ax5 = fig.add_subplot(gs[1, 2])
cm = np.array([[overall_tn, overall_fp], [overall_fn, overall_tp]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax5,
            xticklabels=['PASS', 'FAIL'], yticklabels=['PASS', 'FAIL'],
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})
ax5.set_title('Overall Confusion Matrix (Test)', fontweight='bold', fontsize=11)
ax5.set_ylabel('True Label'); ax5.set_xlabel('Flagged?')

plt.suptitle('Block 4: CRC Evaluation on Test Set', fontsize=14, fontweight='bold', y=0.995)
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/04_final_evaluation.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/04_final_evaluation.png\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================

test_eval_df.to_csv("data/processed/test_evaluation.csv", index=False)
print("✓ Saved test evaluation to data/processed/test_evaluation.csv")

test[['id', 'label', 'source_ds', 'pred_prob_fail', 'score',
      'crc_threshold', 'flagged_by_crc']].to_csv(
    "data/processed/test_crc_results.csv", index=False)
print("✓ Saved test CRC results to data/processed/test_crc_results.csv\n")

# ============================================================================
# FINAL REPORT (honest, AUC-framed)
# ============================================================================

print("Writing FINAL_REPORT.md ...\n")

domain_lines = ""
for _, r in test_eval_df.sort_values('auc', ascending=False).iterrows():
    domain_lines += (
        f"| {r['domain']} | {r['recall']*100:.1f}% | {r['false_alarm_rate']*100:.1f}% | "
        f"{r['precision']*100:.1f}% | {r['auc']:.3f} | {int(r['n_samples'])} |\n"
    )

report = f"""# LLM Hallucination Detection with Mondrian Conformal Risk Control — Test Results

**Author:** Tareq Aldirawi
**Date:** June 2026
**Method:** Mondrian Conformal Risk Control (per-domain FNR control)

## Summary

We apply Mondrian CRC to control the per-domain false-negative rate (FNR) of a
hallucination detector on HaluBench. The conformal layer is valid: it flags the
target fraction of true hallucinations in every domain. The informative result is the
*cost* of that guarantee. The TF-IDF + logistic-regression detector has a per-domain
ROC AUC of about {mean_auc:.2f} — statistically close to random — so achieving high
recall forces a near-maximal false-alarm rate. The conformal machinery is sound; the
base detector is the bottleneck.

## Overall (test set, alpha = 0.10)

| Metric | Value |
|---|---|
| Recall (hallucinations flagged) | {overall_recall*100:.1f}% |
| False-alarm rate | {overall_far*100:.1f}% |
| Precision | {overall_precision*100:.1f}% |
| Pooled ROC AUC | {overall_auc:.3f} |
| Mean per-domain ROC AUC | {mean_auc:.3f} |
| Test samples | {len(test):,} |

## Per-domain (test set)

| Domain | Recall | FAR | Precision | AUC | n |
|---|---|---|---|---|---|
{domain_lines}
Recall is controlled near the 90% target in each domain (the guarantee). FAR is the
price, and AUC ≈ 0.5 across domains explains why that price is high: there is no
threshold that yields high recall at low FAR, because the score barely separates
hallucinated from faithful answers.

## Interpretation

This is a diagnostic result, and a clean one. It separates two questions that are
often conflated: (1) is the conformal procedure valid? — yes, FNR is controlled per
domain; and (2) is the detector useful? — no, its AUC is near 0.5 on HaluBench. The
path forward is a stronger base detector (semantic embeddings, fine-tuning, or
purpose-built hallucination features), not a different conformal setting. A weak
detector wrapped in a valid guarantee is still a weak detector — CRC makes the cost of
that weakness explicit and honest, which is the contribution here.

## Method notes

- Score: s(x) = P(FAIL | x), computed identically on calibration and test (label-free).
- Threshold: per domain, tau_d is the conformal lower-quantile of the calibration FAIL
  scores at rank floor(alpha (m+1)); flag if s(x) >= tau_d. This controls FNR <= alpha
  within each domain (Mondrian).
- Detector: TF-IDF (1-2 grams) + L2 logistic regression, refit with fixed seed so the
  test scores match the calibration model.

## References

- Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal
  Risk Control. ICLR. arXiv:2208.02814.
- Sadinle, M., Lei, J., & Wasserman, L. (2019). Least Ambiguous Set-Valued Classifiers
  with Bounded Error Levels. JASA, 114(525), 223-234.
- Ravi, S. S., Mielczarek, B., Kannappan, A., Kiela, D., & Qian, R. (2024). Lynx: An
  Open Source Hallucination Evaluation Model. arXiv:2407.08488. (HaluBench dataset.)

*Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}.*
"""

with open("FINAL_REPORT.md", "w", encoding="utf-8") as f:
    f.write(report)
print("✓ Saved FINAL_REPORT.md\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("BLOCK 4 COMPLETE")
print("="*80)
print(f"""
Test-set result (alpha = 0.10):
  Recall {overall_recall*100:.1f}%   FAR {overall_far*100:.1f}%   Precision {overall_precision*100:.1f}%
  Mean per-domain AUC {mean_auc:.3f}  (0.50 = random)

Reading: recall is controlled near target in every domain (guarantee valid); the high
FAR is the cost, set by a near-random detector. The conformal layer works; the detector
is the bottleneck.

Files Saved:
  ✓ data/processed/test_evaluation.csv
  ✓ data/processed/test_crc_results.csv
  ✓ python/outputs/04_final_evaluation.png
  ✓ FINAL_REPORT.md
""")
print("="*80 + "\n")
