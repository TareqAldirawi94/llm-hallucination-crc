################################################################################
# BLOCK 8: APPLY CRC TO BERT EMBEDDINGS & COMPREHENSIVE COMPARISON
# Author: Tareq Aldirawi
# Date: June 2026
# FIXED VERSION - No .map() errors, correct CSV columns
################################################################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 8: APPLY CRC TO BERT & COMPARE BERT VS TF-IDF")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD DATA & EMBEDDINGS
# ============================================================================

print("[STEP 1/5] Loading data and BERT embeddings...\n")

train = pd.read_csv("data/processed/train.csv")
calib = pd.read_csv("data/processed/calibration.csv")
test = pd.read_csv("data/processed/test.csv")

train_embeddings_bert = np.load("data/processed/train_embeddings_bert.npy")
calib_embeddings_bert = np.load("data/processed/calib_embeddings_bert.npy")

print(f"✓ Train: {len(train):,} samples")
print(f"✓ Calib: {len(calib):,} samples")
print(f"✓ Test: {len(test):,} samples")
print(f"✓ BERT embeddings shape: {calib_embeddings_bert.shape}\n")

# ============================================================================
# STEP 2: TRAIN BERT DETECTOR
# ============================================================================

print("[STEP 2/5] Training BERT detector on full training set...\n")

y_train = (train['label'] == 'FAIL').astype(int)
y_calib = (calib['label'] == 'FAIL').astype(int)

lr_model_bert = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    n_jobs=-1
)

lr_model_bert.fit(train_embeddings_bert, y_train)
print("✓ BERT detector trained\n")

# ============================================================================
# STEP 3: BERT PREDICTIONS ON CALIBRATION
# ============================================================================

print("[STEP 3/5] Computing BERT predictions & conformity scores...\n")

calib_probs_bert = lr_model_bert.predict_proba(calib_embeddings_bert)

calib_bert = calib.copy()
calib_bert['pred_prob_fail'] = calib_probs_bert[:, 1]
calib_bert['pred_class'] = np.where(calib_probs_bert[:, 1] > 0.5, 'FAIL', 'PASS')
calib_bert['correct'] = (calib_bert['pred_class'] == calib_bert['label'])

# Conformity scores
calib_bert['pred_prob'] = np.where(
    calib_bert['label'] == 'FAIL',
    calib_bert['pred_prob_fail'],
    1 - calib_bert['pred_prob_fail']
)
calib_bert['pred_prob'] = calib_bert['pred_prob'].clip(lower=1e-10)
calib_bert['conformity_score'] = -np.log(calib_bert['pred_prob'])

print(f"✓ Calibration results computed\n")

# ============================================================================
# STEP 4: FIT MONDRIAN CRC FOR BERT
# ============================================================================

print("[STEP 4/5] Fitting Mondrian CRC thresholds for BERT...\n")

alpha = 0.10
crc_thresholds_bert = {}

for domain in sorted(calib_bert['source_ds'].unique()):
    domain_calib = calib_bert[calib_bert['source_ds'] == domain]
    n = len(domain_calib)
    
    quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
    quantile_level = min(quantile_level, 1.0)
    threshold = np.quantile(domain_calib['conformity_score'], quantile_level)
    
    crc_thresholds_bert[domain] = threshold
    
    fail_scores = domain_calib[domain_calib['label'] == 'FAIL']['conformity_score'].values
    coverage = (fail_scores >= threshold).sum() / len(fail_scores) if len(fail_scores) > 0 else 0
    
    print(f"  {domain:20s}: τ = {threshold:.3f}, Coverage = {coverage*100:.1f}%")

print()

# ============================================================================
# STEP 5: EVALUATE ON TEST SET
# ============================================================================

print("[STEP 5/5] Evaluating BERT + CRC on test set...\n")

# Load TF-IDF results
tfidf_results = pd.read_csv("data/processed/test_evaluation.csv")

# Generate BERT embeddings for test set
print("  Generating BERT embeddings for test set...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

test['combined_text'] = test['passage'] + " " + test['answer']
test_embeddings_bert = model.encode(test['combined_text'].tolist(), 
                                     show_progress_bar=True, 
                                     batch_size=32)
print("  ✓ Test embeddings generated\n")

# Predictions
test_probs_bert = lr_model_bert.predict_proba(test_embeddings_bert)

test_bert = test.copy()
test_bert['pred_prob_fail'] = test_probs_bert[:, 1]
test_bert['pred_class'] = np.where(test_probs_bert[:, 1] > 0.5, 'FAIL', 'PASS')

# Conformity scores
test_bert['pred_prob'] = np.where(
    test_bert['label'] == 'FAIL',
    test_bert['pred_prob_fail'],
    1 - test_bert['pred_prob_fail']
)
test_bert['pred_prob'] = test_bert['pred_prob'].clip(lower=1e-10)
test_bert['conformity_score'] = -np.log(test_bert['pred_prob'])

# Apply CRC
test_bert['crc_threshold'] = test_bert['source_ds'].map(crc_thresholds_bert)
test_bert['flagged_by_crc'] = test_bert['conformity_score'] >= test_bert['crc_threshold']

print("BERT CRC Performance on Test Set:\n")

bert_eval_stats = []
for domain in sorted(test_bert['source_ds'].unique()):
    domain_test = test_bert[test_bert['source_ds'] == domain]
    
    tp = ((domain_test['label'] == 'FAIL') & (domain_test['flagged_by_crc'])).sum()
    fp = ((domain_test['label'] == 'PASS') & (domain_test['flagged_by_crc'])).sum()
    fn = ((domain_test['label'] == 'FAIL') & (~domain_test['flagged_by_crc'])).sum()
    tn = ((domain_test['label'] == 'PASS') & (~domain_test['flagged_by_crc'])).sum()
    
    n_fail = (domain_test['label'] == 'FAIL').sum()
    n_pass = (domain_test['label'] == 'PASS').sum()
    
    coverage = tp / n_fail if n_fail > 0 else 0
    false_alarm = fp / n_pass if n_pass > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"{domain}: Coverage={coverage*100:.1f}%, FAR={false_alarm*100:.1f}%, Precision={precision*100:.1f}%")
    
    bert_eval_stats.append({
        'domain': domain,
        'coverage': coverage,
        'false_alarm': false_alarm,
        'precision': precision,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    })

bert_eval_df = pd.DataFrame(bert_eval_stats)
print()

# ============================================================================
# COMPARE BERT VS TF-IDF
# ============================================================================

print("="*80)
print("BERT vs TF-IDF COMPARISON")
print("="*80 + "\n")

bert_coverage = bert_eval_df['coverage'].mean()
bert_false_alarm = bert_eval_df['false_alarm'].mean()
bert_precision = bert_eval_df['precision'].mean()

tfidf_coverage = tfidf_results['coverage'].mean()
tfidf_false_alarm = tfidf_results['false_alarm_rate'].mean()
tfidf_precision = tfidf_results['precision'].mean()

print(f"Overall Performance:\n")
print(f"{'Metric':<20} {'TF-IDF':<15} {'BERT':<15} {'Change':<15}")
print("-" * 65)
print(f"{'Coverage':<20} {tfidf_coverage*100:>6.1f}%{'':<8} {bert_coverage*100:>6.1f}%{'':<8} {(bert_coverage-tfidf_coverage)*100:>+6.1f}%")
print(f"{'False Alarm':<20} {tfidf_false_alarm*100:>6.1f}%{'':<8} {bert_false_alarm*100:>6.1f}%{'':<8} {(bert_false_alarm-tfidf_false_alarm)*100:>+6.1f}%")
print(f"{'Precision':<20} {tfidf_precision*100:>6.1f}%{'':<8} {bert_precision*100:>6.1f}%{'':<8} {(bert_precision-tfidf_precision)*100:>+6.1f}%")
print()

# ============================================================================
# CREATE COMPARISON VISUALIZATIONS
# ============================================================================

print("Creating comparison visualization...\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Coverage
ax = axes[0, 0]
methods = ['TF-IDF', 'BERT']
coverage_vals = [tfidf_coverage*100, bert_coverage*100]
ax.bar(methods, coverage_vals, color=['#e74c3c', '#2ecc71'], alpha=0.8, width=0.6)
ax.set_ylabel('Coverage (%)')
ax.set_title('Coverage: BERT vs TF-IDF', fontweight='bold')
ax.set_ylim(0, 25)
ax.grid(axis='y', alpha=0.3)

# Plot 2: False Alarm
ax = axes[0, 1]
false_alarm_vals = [tfidf_false_alarm*100, bert_false_alarm*100]
ax.bar(methods, false_alarm_vals, color=['#3498db', '#9b59b6'], alpha=0.8, width=0.6)
ax.set_ylabel('False Alarm Rate (%)')
ax.set_title('False Alarm: BERT vs TF-IDF', fontweight='bold')
ax.set_ylim(0, 15)
ax.grid(axis='y', alpha=0.3)

# Plot 3: Precision
ax = axes[1, 0]
precision_vals = [tfidf_precision*100, bert_precision*100]
ax.bar(methods, precision_vals, color=['#f39c12', '#27ae60'], alpha=0.8, width=0.6)
ax.set_ylabel('Precision (%)')
ax.set_title('Precision: BERT vs TF-IDF', fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Plot 4: Per-domain coverage
ax = axes[1, 1]
tfidf_domains = tfidf_results.set_index('domain')['coverage'].sort_values()
bert_by_domain = bert_eval_df.set_index('domain')['coverage'].reindex(tfidf_domains.index)

x = np.arange(len(tfidf_domains))
width = 0.35
ax.bar(x - width/2, tfidf_domains.values*100, width, label='TF-IDF', color='#e74c3c', alpha=0.8)
ax.bar(x + width/2, bert_by_domain.values*100, width, label='BERT', color='#2ecc71', alpha=0.8)
ax.set_ylabel('Coverage (%)')
ax.set_title('Per-Domain Coverage', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tfidf_domains.index, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/08_bert_vs_tfidf_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved comparison visualization\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("Saving results...\n")

bert_thresholds_df = pd.DataFrame([
    {'domain': domain, 'bert_threshold': threshold}
    for domain, threshold in crc_thresholds_bert.items()
])
bert_thresholds_df.to_csv("data/processed/crc_thresholds_bert.csv", index=False)

test_bert[['id', 'label', 'source_ds', 'pred_prob_fail', 'conformity_score',
           'crc_threshold', 'flagged_by_crc']].to_csv(
    "data/processed/test_crc_results_bert.csv", index=False)

comparison_df = pd.DataFrame({
    'Method': ['TF-IDF', 'BERT'],
    'Coverage': [tfidf_coverage, bert_coverage],
    'False_Alarm': [tfidf_false_alarm, bert_false_alarm],
    'Precision': [tfidf_precision, bert_precision]
})
comparison_df.to_csv("data/processed/method_comparison.csv", index=False)

print("✓ All results saved\n")

print("="*80)
print("✓ BLOCK 8 COMPLETE - Phase 2 Finished!")
print("="*80 + "\n")

print(f"""
BERT + CRC Summary:
  Coverage:     {bert_coverage*100:.1f}%
  False Alarm:  {bert_false_alarm*100:.1f}%
  Precision:    {bert_precision*100:.1f}%

vs TF-IDF:
  Coverage:     {tfidf_coverage*100:.1f}%
  False Alarm:  {tfidf_false_alarm*100:.1f}%
  Precision:    {tfidf_precision*100:.1f}%

✓ Ready to push to GitHub!
""")
