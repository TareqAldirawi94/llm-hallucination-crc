################################################################################
# BLOCK 7: TRAIN HALLUCINATION DETECTOR WITH BERT EMBEDDINGS
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
# Semantic detector (vs TF-IDF in Block 2), same CRC machinery downstream.
################################################################################

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 7: TRAIN HALLUCINATION DETECTOR WITH BERT EMBEDDINGS")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("[STEP 1/5] Loading train and calibration data...\n")

train = pd.read_csv("data/processed/train.csv")
calib = pd.read_csv("data/processed/calibration.csv")
print(f"✓ Train: {len(train):,} samples")
print(f"✓ Calib: {len(calib):,} samples\n")

# ============================================================================
# STEP 2: GENERATE BERT EMBEDDINGS
# ============================================================================

print("[STEP 2/5] Generating BERT embeddings...\n")

print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("  ✓ Model loaded\n")

train['combined_text'] = train['passage'] + " " + train['answer']
calib['combined_text'] = calib['passage'] + " " + calib['answer']

print("  Encoding train set...")
train_embeddings = model.encode(train['combined_text'].tolist(),
                                show_progress_bar=True, batch_size=32)
print(f"  ✓ Train embeddings: {train_embeddings.shape}")

print("  Encoding calibration set...")
calib_embeddings = model.encode(calib['combined_text'].tolist(),
                                show_progress_bar=True, batch_size=32)
print(f"  ✓ Calib embeddings: {calib_embeddings.shape}\n")

y_train = (train['label'] == 'FAIL').astype(int)

# ============================================================================
# STEP 3: TRAIN LOGISTIC REGRESSION ON EMBEDDINGS
# ============================================================================

print("[STEP 3/5] Training logistic regression on BERT embeddings...\n")

lr_model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', n_jobs=-1)

print("  Running 5-fold cross-validation...")
cv_results = cross_validate(
    lr_model, train_embeddings, y_train, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=-1
)
cv_accuracy = cv_results['test_accuracy'].mean()
cv_precision = cv_results['test_precision'].mean()
cv_recall = cv_results['test_recall'].mean()
cv_f1 = cv_results['test_f1'].mean()

print(f"  CV Accuracy: {cv_accuracy*100:.1f}% ± {cv_results['test_accuracy'].std()*100:.1f}%")
print(f"  CV F1: {cv_f1*100:.1f}%\n")

print("  Training final model on full training set...")
lr_model.fit(train_embeddings, y_train)
print("  ✓ Final model ready\n")

# ============================================================================
# STEP 4: PREDICT ON CALIBRATION
# ============================================================================

print("[STEP 4/5] Predicting on calibration set...\n")

calib_probs = lr_model.predict_proba(calib_embeddings)
calib_results = calib.copy()
calib_results['pred_prob_fail'] = calib_probs[:, 1]
calib_results['pred_prob_pass'] = calib_probs[:, 0]
calib_results['pred_class'] = np.where(calib_probs[:, 1] > 0.5, 'FAIL', 'PASS')
calib_results['correct'] = (calib_results['pred_class'] == calib_results['label'])

print(f"  Accuracy on calibration set: {calib_results['correct'].mean()*100:.1f}%\n")

# ============================================================================
# STEP 5: HALLUCINATION SCORE (label-free) + SEPARATING POWER
# ============================================================================

print("[STEP 5/5] Computing label-free score and AUC...\n")

# Same definition as Block 2: s(x) = P(FAIL | x). No true label used.
calib_results['score'] = calib_results['pred_prob_fail']

y = (calib_results['label'] == 'FAIL').astype(int).values
overall_auc = roc_auc_score(y, calib_results['score'].values)
mean_fail = calib_results.loc[calib_results['label'] == 'FAIL', 'score'].mean()
mean_pass = calib_results.loc[calib_results['label'] == 'PASS', 'score'].mean()

print(f"  s(x) = P(FAIL|x):  mean|FAIL {mean_fail:.3f}  mean|PASS {mean_pass:.3f}"
      f"  (separation {mean_fail - mean_pass:+.3f})")
print(f"  Pooled ROC AUC: {overall_auc:.3f}   (0.50 = random)\n")

print("  Per-domain AUC (the answer: does BERT beat TF-IDF's ~0.5?):")
auc_by_domain = {}
for dom in sorted(calib_results['source_ds'].unique()):
    d = calib_results[calib_results['source_ds'] == dom]
    yy = (d['label'] == 'FAIL').astype(int).values
    auc = roc_auc_score(yy, d['score'].values) if len(np.unique(yy)) > 1 else float('nan')
    auc_by_domain[dom] = auc
    flag = "  <-- near random" if auc < 0.55 else ""
    print(f"    {dom:14s} AUC = {auc:.3f}{flag}")
mean_domain_auc = np.nanmean(list(auc_by_domain.values()))
print(f"    {'MEAN':14s} AUC = {mean_domain_auc:.3f}\n")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("Creating visualizations...\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

cm = confusion_matrix(calib_results['label'], calib_results['pred_class'], labels=['PASS', 'FAIL'])
ax = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
            xticklabels=['PASS', 'FAIL'], yticklabels=['PASS', 'FAIL'],
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})
ax.set_title(f'Confusion Matrix (BERT, acc={calib_results["correct"].mean()*100:.0f}%)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')

ax = axes[0, 1]
for label in ['PASS', 'FAIL']:
    sub = calib_results[calib_results['label'] == label]
    ax.hist(sub['score'], alpha=0.6, label=label, bins=50)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('s = P(FAIL|x)'); ax.set_ylabel('Count')
ax.set_title('Score Distribution (BERT)', fontweight='bold', fontsize=12)
ax.legend(); ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
calib_results.boxplot(column='score', by='label', ax=ax)
ax.set_title('Score s = P(FAIL|x) by Label (BERT)', fontweight='bold', fontsize=12)
ax.set_ylabel('Score'); ax.set_xlabel('True Label')
plt.suptitle('')

ax = axes[1, 1]
s = pd.Series(auc_by_domain).sort_values()
colors = ['#e74c3c' if a < 0.55 else '#3498db' for a in s.values]
s.plot(kind='barh', ax=ax, color=colors)
ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Random (0.5)')
ax.set_xlabel('ROC AUC'); ax.set_xlim(0, 1)
ax.set_title('BERT Separating Power by Domain', fontweight='bold', fontsize=12)
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/07_bert_detector_performance.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/07_bert_detector_performance.png")

# ============================================================================
# SAVE
# ============================================================================

print("\nSaving results...\n")

calib_results[[
    'id', 'passage', 'question', 'answer', 'label', 'source_ds',
    'pred_prob_fail', 'pred_prob_pass', 'pred_class', 'correct', 'score'
]].to_csv("data/processed/calib_predictions_bert.csv", index=False)
print("✓ Saved data/processed/calib_predictions_bert.csv  (carries 'score')")

np.save("data/processed/train_embeddings_bert.npy", train_embeddings)
np.save("data/processed/calib_embeddings_bert.npy", calib_embeddings)
print("✓ Saved embeddings to data/processed/\n")

# ============================================================================
# SUMMARY
# ============================================================================

tn = ((calib_results['pred_class'] == 'PASS') & (calib_results['label'] == 'PASS')).sum()
fp = ((calib_results['pred_class'] == 'FAIL') & (calib_results['label'] == 'PASS')).sum()
fn = ((calib_results['pred_class'] == 'PASS') & (calib_results['label'] == 'FAIL')).sum()
tp = ((calib_results['pred_class'] == 'FAIL') & (calib_results['label'] == 'FAIL')).sum()
overall_accuracy = (tp + tn) / (tp + tn + fp + fn)

print("="*80)
print("BLOCK 7 COMPLETE - BERT DETECTOR SUMMARY")
print("="*80)
print(f"""
Model: Logistic Regression on all-MiniLM-L6-v2 embeddings (384-dim)
Calibration accuracy: {overall_accuracy*100:.1f}%   CV accuracy: {cv_accuracy*100:.1f}%

Separating power (the question we set out to answer):
  Pooled ROC AUC:        {overall_auc:.3f}
  Mean per-domain AUC:   {mean_domain_auc:.3f}   (TF-IDF was ~0.53)
  Score separation:      mean|FAIL {mean_fail:.3f} - mean|PASS {mean_pass:.3f} = {mean_fail-mean_pass:+.3f}

Reading: if these AUCs are also ~0.5, the semantic detector is no more separable than
the lexical one on HaluBench -- the conformal guarantee will be just as valid and just
as costly. If they are meaningfully above ~0.6, BERT changes the operating frontier and
we re-run 8 to quantify it. The number above is the answer.

Files Saved:
  ✓ data/processed/calib_predictions_bert.csv
  ✓ data/processed/{{train,calib}}_embeddings_bert.npy
  ✓ python/outputs/07_bert_detector_performance.png

Next: Block 8 - apply Mondrian CRC on the BERT score and compare to TF-IDF.
""")
print("="*80 + "\n")
