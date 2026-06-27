################################################################################
# BLOCK 8: APPLY CRC TO BERT & COMPARE BERT vs TF-IDF
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# Honest comparison of the two detectors under identical Mondrian FNR control.
# Both use the label-free score s(x) = P(FAIL|x). The headline is NOT recall (both
# are controlled to the target) but separating power: pooled AUC vs per-domain AUC.
# A pooled AUC above the per-domain AUC means the score tracks DOMAIN IDENTITY, not
# hallucination -- which conditioning on domain (Mondrian) strips away.
################################################################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 8: APPLY CRC TO BERT & COMPARE BERT vs TF-IDF")
print("="*80 + "\n")

alpha = 0.10

def conformal_fnr_threshold(fail_scores, a):
    m = len(fail_scores)
    if m == 0:
        return -np.inf
    k = int(np.floor(a * (m + 1)))
    return -np.inf if k < 1 else np.sort(fail_scores)[k - 1]

def per_domain_metrics(frame, thr_map):
    rows = []
    for dom in sorted(frame['source_ds'].unique()):
        d = frame[frame['source_ds'] == dom]
        flagged = d['score'].values >= thr_map[dom]
        is_fail = (d['label'] == 'FAIL').values
        tp = (flagged & is_fail).sum(); fn = (~flagged & is_fail).sum()
        fp = (flagged & ~is_fail).sum(); tn = (~flagged & ~is_fail).sum()
        yy = is_fail.astype(int)
        auc = roc_auc_score(yy, d['score'].values) if len(np.unique(yy)) > 1 else np.nan
        rows.append({'domain': dom,
                     'recall': tp/(tp+fn) if (tp+fn) else np.nan,
                     'far': fp/(fp+tn) if (fp+tn) else np.nan,
                     'precision': tp/(tp+fp) if (tp+fp) else np.nan,
                     'auc': auc})
    return pd.DataFrame(rows)

# ============================================================================
# STEP 1: LOAD
# ============================================================================

print("[STEP 1/5] Loading data, embeddings, and TF-IDF test scores...\n")

train = pd.read_csv("data/processed/train.csv")
calib = pd.read_csv("data/processed/calibration.csv")
test = pd.read_csv("data/processed/test.csv")

train_emb = np.load("data/processed/train_embeddings_bert.npy")
calib_emb = np.load("data/processed/calib_embeddings_bert.npy")

# TF-IDF test scores (label-free) from Block 4
tfidf_test = pd.read_csv("data/processed/test_crc_results.csv")  # has 'score','label','source_ds'
print(f"✓ Train {len(train):,}  Calib {len(calib):,}  Test {len(test):,}")
print(f"✓ BERT calib embeddings: {calib_emb.shape}\n")

# ============================================================================
# STEP 2: TRAIN BERT DETECTOR, SCORE CALIB (label-free)
# ============================================================================

print("[STEP 2/5] Training BERT detector and scoring calibration...\n")

y_train = (train['label'] == 'FAIL').astype(int)
lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', n_jobs=-1)
lr.fit(train_emb, y_train)

calib_bert = calib.copy()
calib_bert['score'] = lr.predict_proba(calib_emb)[:, 1]   # P(FAIL|x), no label used
print("✓ Calibration scored\n")

# ============================================================================
# STEP 3: FIT PER-DOMAIN BERT THRESHOLDS
# ============================================================================

print("[STEP 3/5] Fitting Mondrian FNR thresholds (BERT)...\n")

bert_thr = {}
for dom in sorted(calib_bert['source_ds'].unique()):
    fs = calib_bert.loc[(calib_bert.source_ds == dom) & (calib_bert.label == 'FAIL'), 'score'].values
    bert_thr[dom] = conformal_fnr_threshold(fs, alpha)
    print(f"  {dom:14s}: tau = {bert_thr[dom]:.3f}")
print()

# ============================================================================
# STEP 4: SCORE TEST (label-free) AND EVALUATE
# ============================================================================

print("[STEP 4/5] Encoding test set and evaluating...\n")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
test['combined_text'] = test['passage'] + " " + test['answer']
test_emb = model.encode(test['combined_text'].tolist(), show_progress_bar=True, batch_size=32)

test_bert = test.copy()
test_bert['score'] = lr.predict_proba(test_emb)[:, 1]   # label-free

bert_metrics = per_domain_metrics(test_bert, bert_thr)

# TF-IDF metrics under its own Block-3 thresholds, recomputed here for matched reporting
tfidf_thr = dict(pd.read_csv("data/processed/crc_thresholds.csv").values)
tfidf_metrics = per_domain_metrics(tfidf_test, tfidf_thr)

# pooled AUCs
bert_pooled_auc = roc_auc_score((test_bert.label == 'FAIL').astype(int), test_bert['score'])
tfidf_pooled_auc = roc_auc_score((tfidf_test.label == 'FAIL').astype(int), tfidf_test['score'])

print("Per-domain (test):\n")
print(f"  {'domain':14s} {'TFIDF auc':>9s} {'BERT auc':>9s} | {'TFIDF rec':>9s} {'BERT rec':>9s} | {'TFIDF far':>9s} {'BERT far':>9s}")
for dom in sorted(test_bert['source_ds'].unique()):
    t = tfidf_metrics.set_index('domain').loc[dom]
    b = bert_metrics.set_index('domain').loc[dom]
    print(f"  {dom:14s} {t['auc']:9.3f} {b['auc']:9.3f} | "
          f"{t['recall']*100:8.1f}% {b['recall']*100:8.1f}% | "
          f"{t['far']*100:8.1f}% {b['far']*100:8.1f}%")
print()

# ============================================================================
# STEP 5: COMPARISON SUMMARY
# ============================================================================

print("[STEP 5/5] Comparison summary...\n")

def summarise(name, metrics, pooled_auc):
    return {
        'method': name,
        'pooled_auc': pooled_auc,
        'mean_domain_auc': metrics['auc'].mean(),
        'mean_recall': metrics['recall'].mean(),
        'mean_far': metrics['far'].mean(),
        'mean_precision': metrics['precision'].mean(),
    }

comp = pd.DataFrame([
    summarise('TF-IDF', tfidf_metrics, tfidf_pooled_auc),
    summarise('BERT',   bert_metrics,  bert_pooled_auc),
])

print(comp.to_string(index=False, formatters={
    'pooled_auc': lambda x: f"{x:.3f}",
    'mean_domain_auc': lambda x: f"{x:.3f}",
    'mean_recall': lambda x: f"{x*100:.1f}%",
    'mean_far': lambda x: f"{x*100:.1f}%",
    'mean_precision': lambda x: f"{x*100:.1f}%",
}))
print()

# ============================================================================
# FIGURE
# ============================================================================

print("Creating comparison figure...\n")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pooled vs per-domain AUC (the punchline)
ax = axes[0, 0]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [tfidf_pooled_auc, bert_pooled_auc], w, label='pooled AUC', color='#95a5a6')
ax.bar(x + w/2, [tfidf_metrics['auc'].mean(), bert_metrics['auc'].mean()], w,
       label='mean per-domain AUC', color='#2ecc71')
ax.axhline(0.5, color='black', ls='--', lw=2, label='random')
ax.set_xticks(x); ax.set_xticklabels(['TF-IDF', 'BERT'])
ax.set_ylabel('ROC AUC'); ax.set_ylim(0, 1)
ax.set_title('Pooled vs Per-Domain AUC\n(gap = tracks domain, not hallucination)',
             fontweight='bold', fontsize=11)
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

# Per-domain AUC, both detectors
ax = axes[0, 1]
doms = sorted(test_bert['source_ds'].unique())
t_auc = [tfidf_metrics.set_index('domain').loc[d, 'auc'] for d in doms]
b_auc = [bert_metrics.set_index('domain').loc[d, 'auc'] for d in doms]
y = np.arange(len(doms)); h = 0.38
ax.barh(y - h/2, t_auc, h, label='TF-IDF', color='#e74c3c', alpha=0.85)
ax.barh(y + h/2, b_auc, h, label='BERT', color='#2ecc71', alpha=0.85)
ax.axvline(0.5, color='black', ls='--', lw=2, label='random')
ax.set_yticks(y); ax.set_yticklabels(doms); ax.set_xlim(0, 1)
ax.set_xlabel('Per-domain AUC')
ax.set_title('Within-Domain Separating Power', fontweight='bold', fontsize=11)
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

# Recall (both ~target)
ax = axes[1, 0]
t_rec = [tfidf_metrics.set_index('domain').loc[d, 'recall']*100 for d in doms]
b_rec = [bert_metrics.set_index('domain').loc[d, 'recall']*100 for d in doms]
ax.barh(y - h/2, t_rec, h, label='TF-IDF', color='#e74c3c', alpha=0.85)
ax.barh(y + h/2, b_rec, h, label='BERT', color='#2ecc71', alpha=0.85)
ax.axvline(90, color='black', ls='--', lw=2, label='90% target')
ax.set_yticks(y); ax.set_yticklabels(doms); ax.set_xlim(0, 105)
ax.set_xlabel('Recall (%)')
ax.set_title('Recall (both controlled to target)', fontweight='bold', fontsize=11)
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

# FAR (both high)
ax = axes[1, 1]
t_far = [tfidf_metrics.set_index('domain').loc[d, 'far']*100 for d in doms]
b_far = [bert_metrics.set_index('domain').loc[d, 'far']*100 for d in doms]
ax.barh(y - h/2, t_far, h, label='TF-IDF', color='#e74c3c', alpha=0.85)
ax.barh(y + h/2, b_far, h, label='BERT', color='#2ecc71', alpha=0.85)
ax.set_yticks(y); ax.set_yticklabels(doms); ax.set_xlim(0, 105)
ax.set_xlabel('False-Alarm Rate (%)')
ax.set_title('FAR (cost; high for both)', fontweight='bold', fontsize=11)
ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

plt.suptitle('Block 8: TF-IDF vs BERT under Mondrian CRC', fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/08_bert_vs_tfidf_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/08_bert_vs_tfidf_comparison.png")

# ============================================================================
# SAVE
# ============================================================================

pd.DataFrame([{'domain': d, 'bert_threshold': t} for d, t in bert_thr.items()]
            ).to_csv("data/processed/crc_thresholds_bert.csv", index=False)
test_bert[['id', 'label', 'source_ds', 'score']].assign(
    crc_threshold=test_bert['source_ds'].map(bert_thr)
).assign(flagged=lambda x: x['score'] >= x['crc_threshold']
).to_csv("data/processed/test_crc_results_bert.csv", index=False)
comp.to_csv("data/processed/method_comparison.csv", index=False)
bert_metrics.to_csv("data/processed/bert_test_evaluation.csv", index=False)
print("✓ Saved BERT thresholds, test results, comparison, per-domain metrics\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("BLOCK 8 COMPLETE")
print("="*80)
print(f"""
TF-IDF vs BERT under identical Mondrian FNR control:

                    TF-IDF     BERT
  pooled AUC        {tfidf_pooled_auc:.3f}     {bert_pooled_auc:.3f}
  per-domain AUC    {tfidf_metrics['auc'].mean():.3f}     {bert_metrics['auc'].mean():.3f}   <- the real signal
  mean recall       {tfidf_metrics['recall'].mean()*100:4.1f}%     {bert_metrics['recall'].mean()*100:4.1f}%
  mean FAR          {tfidf_metrics['far'].mean()*100:4.1f}%     {bert_metrics['far'].mean()*100:4.1f}%

The conclusion: both detectors are near random WITHIN domain (per-domain AUC ~ 0.5),
so the semantic model is no more separable than the lexical one on HaluBench. The
higher POOLED AUC reflects the score tracking domain identity, not hallucination --
conditioning on domain (which Mondrian does) removes that and exposes the ~0.5 signal.
Under matched FNR control both detectors give controlled recall at near-maximal FAR.
The conformal layer is equally valid on both; neither detector is deployable as-is.

Files Saved:
  ✓ python/outputs/08_bert_vs_tfidf_comparison.png
  ✓ data/processed/crc_thresholds_bert.csv, test_crc_results_bert.csv,
    method_comparison.csv, bert_test_evaluation.csv
""")
print("="*80 + "\n")
