################################################################################
# BLOCK 2: TRAIN BASE HALLUCINATION DETECTOR (TF-IDF + Logistic Regression)
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 2: TRAIN BASE HALLUCINATION DETECTOR")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("[STEP 1/5] Loading train and calibration data...")

train = pd.read_csv("data/processed/train.csv")
calib = pd.read_csv("data/processed/calibration.csv")

print(f"✓ Train: {len(train):,} samples")
print(f"✓ Calib: {len(calib):,} samples\n")

train_fail = (train['label'] == 'FAIL').sum()
train_pass = (train['label'] == 'PASS').sum()
calib_fail = (calib['label'] == 'FAIL').sum()
calib_pass = (calib['label'] == 'PASS').sum()

print(f"  Train: {train_fail:,} FAIL ({train_fail/len(train)*100:.1f}%), "
      f"{train_pass:,} PASS ({train_pass/len(train)*100:.1f}%)")
print(f"  Calib: {calib_fail:,} FAIL ({calib_fail/len(calib)*100:.1f}%), "
      f"{calib_pass:,} PASS ({calib_pass/len(calib)*100:.1f}%)\n")

# ============================================================================
# STEP 2: CREATE TEXT FEATURES (TF-IDF)
# ============================================================================

print("[STEP 2/5] Creating text features (TF-IDF)...")

train['combined_text'] = train['passage'] + " " + train['answer']
calib['combined_text'] = calib['passage'] + " " + calib['answer']

print("  Fitting TF-IDF vectorizer on training set...")
tfidf = TfidfVectorizer(
    max_features=None,          # Keep all features
    min_df=2,                   # Remove words appearing in < 2 documents
    max_df=0.95,                # Remove words appearing in > 95% of documents
    ngram_range=(1, 2),         # Unigrams and bigrams
    lowercase=True,
    stop_words='english'
)

X_train = tfidf.fit_transform(train['combined_text'])
print(f"✓ Train TF-IDF matrix: {X_train.shape[0]:,} samples × {X_train.shape[1]:,} features")

print("  Transforming calibration set...")
X_calib = tfidf.transform(calib['combined_text'])
print(f"✓ Calib TF-IDF matrix: {X_calib.shape[0]:,} samples × {X_calib.shape[1]:,} features")

y_train = (train['label'] == 'FAIL').astype(int)
y_calib = (calib['label'] == 'FAIL').astype(int)

print(f"✓ Features created and ready\n")

# ============================================================================
# STEP 3: TRAIN LOGISTIC REGRESSION
# ============================================================================

print("[STEP 3/5] Training logistic regression classifier...")

print("  Running 5-fold cross-validation...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='saga',  # Works well with sparse matrices
    n_jobs=-1       # Use all CPU cores
)

cv_results = cross_validate(
    lr_model, X_train, y_train, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=-1
)

cv_accuracy = cv_results['test_accuracy'].mean()
cv_precision = cv_results['test_precision'].mean()
cv_recall = cv_results['test_recall'].mean()
cv_f1 = cv_results['test_f1'].mean()

print(f"✓ Model trained successfully")
print(f"  CV Accuracy: {cv_accuracy*100:.1f}% ± {cv_results['test_accuracy'].std()*100:.1f}%")
print(f"  CV Precision: {cv_precision*100:.1f}%")
print(f"  CV Recall: {cv_recall*100:.1f}%")
print(f"  CV F1: {cv_f1*100:.1f}%\n")

print("  Training final model on full training set...")
lr_model.fit(X_train, y_train)
print(f"✓ Final model ready\n")

# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================

print("[STEP 4/5] Getting predictions on calibration set...")

calib_probs = lr_model.predict_proba(X_calib)  # [[P(PASS), P(FAIL)], ...]
calib_preds = lr_model.predict(X_calib)         # [0 or 1, ...]

calib_results = calib.copy()
calib_results['pred_prob_fail'] = calib_probs[:, 1]  # P(FAIL)
calib_results['pred_prob_pass'] = calib_probs[:, 0]  # P(PASS)
calib_results['pred_class'] = calib_preds
calib_results['pred_class'] = calib_results['pred_class'].map({0: 'PASS', 1: 'FAIL'})
calib_results['correct'] = (calib_results['pred_class'] == calib_results['label'])

accuracy = calib_results['correct'].mean()
print(f"✓ Predictions complete")
print(f"  Accuracy on calibration set: {accuracy*100:.1f}%\n")

# ============================================================================
# STEP 5: HALLUCINATION SCORE (label-free, used for CRC)
# ============================================================================

print("[STEP 5/5] Computing hallucination scores...")

# The CRC score is the detector's predicted probability of FAIL, s(x) = P(FAIL | x).
# It does NOT use the true label, so it is computed the SAME way on calibration
# and on test data. Downstream flagging rule: flag as hallucination if s(x) >= tau.
#
# (The previous version used -log P(true label | x). That needs the true label,
#  so it is not computable at test time and inverts the flag direction; replaced.)
calib_results['score'] = calib_results['pred_prob_fail']

mean_score = calib_results['score'].mean()
median_score = calib_results['score'].median()
std_score = calib_results['score'].std()
min_score = calib_results['score'].min()
max_score = calib_results['score'].max()

# Quick separation check: mean score on FAIL should exceed mean score on PASS
mean_fail = calib_results.loc[calib_results['label'] == 'FAIL', 'score'].mean()
mean_pass = calib_results.loc[calib_results['label'] == 'PASS', 'score'].mean()

print(f"✓ Hallucination scores computed  (s = P(FAIL | x))")
print(f"  Mean: {mean_score:.3f}   Median: {median_score:.3f}   Std: {std_score:.3f}")
print(f"  Range: [{min_score:.3f}, {max_score:.3f}]")
print(f"  Mean score | FAIL: {mean_fail:.3f}   Mean score | PASS: {mean_pass:.3f}   "
      f"(separation: {mean_fail - mean_pass:+.3f})\n")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

print("Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(calib_results['label'], calib_results['pred_class'],
                      labels=['PASS', 'FAIL'])

ax = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=False, ax=ax,
            xticklabels=['PASS', 'FAIL'], yticklabels=['PASS', 'FAIL'])
ax.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

ax = axes[0, 1]
for label in ['PASS', 'FAIL']:
    subset = calib_results[calib_results['label'] == label]
    ax.hist(subset['pred_prob_fail'], alpha=0.6, label=label, bins=50)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('P(FAIL)')
ax.set_ylabel('Count')
ax.set_title('Predicted Probability Distribution', fontweight='bold', fontsize=12)
ax.legend()

# Hallucination score by true label: FAIL should sit higher than PASS.
ax = axes[1, 0]
calib_results.boxplot(column='score', by='label', ax=ax)
ax.set_title('Hallucination Score s = P(FAIL|x) by Label', fontweight='bold', fontsize=12)
ax.set_ylabel('Score')
ax.set_xlabel('True Label')
plt.suptitle('')  # remove the automatic "Boxplot grouped by label" supertitle

ax = axes[1, 1]
domain_accuracy = calib_results.groupby('source_ds')['correct'].mean().sort_values()
domain_accuracy.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Accuracy')
ax.set_title('Detector Accuracy by Domain', fontweight='bold', fontsize=12)
ax.set_xlim(0, 1)

plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/02_detector_performance.png", dpi=300, bbox_inches='tight')
print("✓ Saved visualization to python/outputs/02_detector_performance.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\nSaving results...")

calib_results_save = calib_results[[
    'id', 'passage', 'question', 'answer', 'label', 'source_ds',
    'pred_prob_fail', 'pred_prob_pass', 'pred_class', 'correct', 'score'
]]
calib_results_save.to_csv("data/processed/calib_predictions.csv", index=False)
print("✓ Saved predictions to data/processed/calib_predictions.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("BLOCK 2 COMPLETE - SUMMARY")
print("="*80 + "\n")

tn = ((calib_results['pred_class'] == 'PASS') & (calib_results['label'] == 'PASS')).sum()
fp = ((calib_results['pred_class'] == 'FAIL') & (calib_results['label'] == 'PASS')).sum()
fn = ((calib_results['pred_class'] == 'PASS') & (calib_results['label'] == 'FAIL')).sum()
tp = ((calib_results['pred_class'] == 'FAIL') & (calib_results['label'] == 'FAIL')).sum()

overall_accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"""
Detector Model: TF-IDF + Logistic Regression
Features: {X_train.shape[1]:,} TF-IDF features (FULL feature set)
Samples: {len(calib_results):,} calibration

Performance Metrics (0.5 threshold, for reference only):
  - Overall Accuracy: {overall_accuracy*100:.1f}%
  - Precision (Detect FAIL): {precision*100:.1f}%
  - Recall (Detect FAIL): {recall*100:.1f}%
  - F1 Score: {f1*100:.1f}%

Cross-Validation (on training set):
  - CV Accuracy: {cv_accuracy*100:.1f}% ± {cv_results['test_accuracy'].std()*100:.1f}%
  - CV Precision: {cv_precision*100:.1f}%
  - CV Recall: {cv_recall*100:.1f}%
  - CV F1: {cv_f1*100:.1f}%

Confusion Matrix (0.5 threshold):
  - True Positives (FAIL → FAIL): {tp:,}
  - False Positives (PASS → FAIL): {fp:,}
  - True Negatives (PASS → PASS): {tn:,}
  - False Negatives (FAIL → PASS): {fn:,}

Hallucination score for CRC: s(x) = P(FAIL | x)  (label-free)
  - Mean: {mean_score:.3f}   Median: {median_score:.3f}   Std: {std_score:.3f}
  - Mean | FAIL: {mean_fail:.3f}   Mean | PASS: {mean_pass:.3f}   (sep: {mean_fail - mean_pass:+.3f})

Files Saved:
  ✓ data/processed/calib_predictions.csv   (now carries 'score' = P(FAIL|x))
  ✓ python/outputs/02_detector_performance.png

✓ Ready for Block 3 - Fit Mondrian CRC thresholds per domain
""")

print("="*80 + "\n")
