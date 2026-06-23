################################################################################
# BLOCK 7: TRAIN HALLUCINATION DETECTOR WITH BERT EMBEDDINGS
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
# Improvement: BERT embeddings (vs TF-IDF in Block 2)
################################################################################

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
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

# Check label balance
train_fail = (train['label'] == 'FAIL').sum()
train_pass = (train['label'] == 'PASS').sum()
calib_fail = (calib['label'] == 'FAIL').sum()
calib_pass = (calib['label'] == 'PASS').sum()

print(f"  Train: {train_fail:,} FAIL ({train_fail/len(train)*100:.1f}%), "
      f"{train_pass:,} PASS ({train_pass/len(train)*100:.1f}%)")
print(f"  Calib: {calib_fail:,} FAIL ({calib_fail/len(calib)*100:.1f}%), "
      f"{calib_pass:,} PASS ({calib_pass/len(calib)*100:.1f}%)\n")

# ============================================================================
# STEP 2: GENERATE BERT EMBEDDINGS
# ============================================================================

print("[STEP 2/5] Generating BERT embeddings...\n")

# Load pre-trained BERT model (sentence-transformers)
print("  Loading BERT model (sentence-transformers/all-MiniLM-L6-v2)...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("  ✓ Model loaded\n")

# Combine passage + answer for semantic representation
train['combined_text'] = train['passage'] + " " + train['answer']
calib['combined_text'] = calib['passage'] + " " + calib['answer']

# Generate embeddings
print("  Encoding train set embeddings...")
train_embeddings = model.encode(train['combined_text'].tolist(), 
                                show_progress_bar=True, 
                                batch_size=32)
print(f"  ✓ Train embeddings shape: {train_embeddings.shape}")

print("  Encoding calibration set embeddings...")
calib_embeddings = model.encode(calib['combined_text'].tolist(), 
                                show_progress_bar=True, 
                                batch_size=32)
print(f"  ✓ Calib embeddings shape: {calib_embeddings.shape}\n")

# Convert labels to binary (FAIL=1, PASS=0)
y_train = (train['label'] == 'FAIL').astype(int)
y_calib = (calib['label'] == 'FAIL').astype(int)

print(f"✓ Embeddings ready for training\n")

# ============================================================================
# STEP 3: TRAIN LOGISTIC REGRESSION
# ============================================================================

print("[STEP 3/5] Training logistic regression on BERT embeddings...\n")

print("  Running 5-fold cross-validation...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',  # Works well with dense BERT embeddings
    n_jobs=-1
)

# 5-fold cross-validation
cv_results = cross_validate(
    lr_model, 
    train_embeddings, 
    y_train,
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    n_jobs=-1
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

# Train final model on full training set
print("  Training final model on full training set...")
lr_model.fit(train_embeddings, y_train)
print(f"✓ Final model ready\n")

# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================

print("[STEP 4/5] Getting predictions on calibration set...\n")

# Get probability predictions
calib_probs = lr_model.predict_proba(calib_embeddings)
calib_preds = lr_model.predict(calib_embeddings)

# Create results dataframe
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
# STEP 5: COMPUTE CONFORMITY SCORES
# ============================================================================

print("[STEP 5/5] Computing conformity scores...\n")

# Conformity score = -log(probability of predicted label)
calib_results['pred_prob'] = calib_results.apply(
    lambda row: row['pred_prob_fail'] if row['label'] == 'FAIL' 
                else row['pred_prob_pass'],
    axis=1
)

# Clip to avoid log(0)
calib_results['pred_prob'] = calib_results['pred_prob'].clip(lower=1e-10)

# Conformity score
calib_results['conformity_score'] = -np.log(calib_results['pred_prob'])

mean_score = calib_results['conformity_score'].mean()
median_score = calib_results['conformity_score'].median()
std_score = calib_results['conformity_score'].std()
min_score = calib_results['conformity_score'].min()
max_score = calib_results['conformity_score'].max()

print(f"✓ Conformity scores computed")
print(f"  Mean: {mean_score:.3f}")
print(f"  Median: {median_score:.3f}")
print(f"  Std Dev: {std_score:.3f}")
print(f"  Range: [{min_score:.3f}, {max_score:.3f}]\n")

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

print("Creating visualizations...\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(calib_results['label'], calib_results['pred_class'], 
                      labels=['PASS', 'FAIL'])

ax = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False, ax=ax,
            xticklabels=['PASS', 'FAIL'], yticklabels=['PASS', 'FAIL'],
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})
ax.set_title('Confusion Matrix (BERT)', fontweight='bold', fontsize=12)
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# Plot 2: Probability distribution
ax = axes[0, 1]
for label in ['PASS', 'FAIL']:
    subset = calib_results[calib_results['label'] == label]
    ax.hist(subset['pred_prob_fail'], alpha=0.6, label=label, bins=50)
ax.axvline(0.5, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('P(FAIL)')
ax.set_ylabel('Count')
ax.set_title('Predicted Probability Distribution (BERT)', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: Conformity scores by label
ax = axes[1, 0]
calib_results.boxplot(column='conformity_score', by='label', ax=ax)
ax.set_title('Conformity Scores by Label (BERT)', fontweight='bold', fontsize=12)
ax.set_ylabel('Conformity Score')
ax.set_xlabel('True Label')
plt.sca(ax)
plt.xticks([1, 2], ['PASS', 'FAIL'])

# Plot 4: Accuracy by domain
ax = axes[1, 1]
domain_accuracy = calib_results.groupby('source_ds')['correct'].mean().sort_values()
domain_accuracy.plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Accuracy')
ax.set_title('BERT Detector Accuracy by Domain', fontweight='bold', fontsize=12)
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/07_bert_detector_performance.png", dpi=300, bbox_inches='tight')
print("✓ Saved visualization to python/outputs/07_bert_detector_performance.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\nSaving results...\n")

# Save calibration predictions
calib_results_save = calib_results[[
    'id', 'passage', 'question', 'answer', 'label', 'source_ds',
    'pred_prob_fail', 'pred_prob_pass', 'pred_class', 'correct', 'conformity_score'
]]

calib_results_save.to_csv("data/processed/calib_predictions_bert.csv", index=False)
print("✓ Saved predictions to data/processed/calib_predictions_bert.csv")

# Save embeddings for later use
np.save("data/processed/train_embeddings_bert.npy", train_embeddings)
np.save("data/processed/calib_embeddings_bert.npy", calib_embeddings)
print("✓ Saved embeddings to data/processed/\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("BLOCK 7 COMPLETE - BERT DETECTOR SUMMARY")
print("="*80 + "\n")

# Calculate metrics
tn = ((calib_results['pred_class'] == 'PASS') & (calib_results['label'] == 'PASS')).sum()
fp = ((calib_results['pred_class'] == 'FAIL') & (calib_results['label'] == 'PASS')).sum()
fn = ((calib_results['pred_class'] == 'PASS') & (calib_results['label'] == 'FAIL')).sum()
tp = ((calib_results['pred_class'] == 'FAIL') & (calib_results['label'] == 'FAIL')).sum()

overall_accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"""
BERT Detector Performance (Calibration Set):

Model: Logistic Regression on BERT Embeddings
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Embedding Dimension: 384
Samples: {len(calib_results):,}

Performance Metrics:
  - Overall Accuracy: {overall_accuracy*100:.1f}%
  - Precision (Detect FAIL): {precision*100:.1f}%
  - Recall (Detect FAIL): {recall*100:.1f}%
  - F1 Score: {f1*100:.1f}%

Cross-Validation (on training set):
  - CV Accuracy: {cv_accuracy*100:.1f}% ± {cv_results['test_accuracy'].std()*100:.1f}%
  - CV Precision: {cv_precision*100:.1f}%
  - CV Recall: {cv_recall*100:.1f}%
  - CV F1: {cv_f1*100:.1f}%

Confusion Matrix:
  - True Positives (FAIL → FAIL): {tp:,}
  - False Positives (PASS → FAIL): {fp:,}
  - True Negatives (PASS → PASS): {tn:,}
  - False Negatives (FAIL → PASS): {fn:,}

Conformity Scores (for CRC):
  - Mean: {mean_score:.3f}
  - Median: {median_score:.3f}
  - Std Dev: {std_score:.3f}
  - Range: [{min_score:.3f}, {max_score:.3f}]

Files Saved:
  ✓ data/processed/calib_predictions_bert.csv
  ✓ data/processed/train_embeddings_bert.npy
  ✓ data/processed/calib_embeddings_bert.npy
  ✓ python/outputs/07_bert_detector_performance.png

✓ Ready for Block 8 - Apply CRC to BERT embeddings
""")

print("="*80 + "\n")

print("✓ Block 7 (BERT Detector) complete!\n")
