################################################################################
# BLOCK 9: FAITHFULNESS-AWARE DETECTOR (relational features)
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# Hypothesis from Blocks 7-8: the detectors are near random WITHIN domain because
# they encode "passage + answer" as one pooled vector and never model whether the
# answer is SUPPORTED BY the passage -- which is what (non-)hallucination is.
#
# This block tests that directly. It embeds passage and answer SEPARATELY and feeds
# the logistic head ONLY their relation:
#     |emb_p - emb_a|, emb_p * emb_a, cosine(emb_p, emb_a), lexical overlap.
# Raw embeddings are deliberately excluded -- they are what encoded domain identity
# and inflated the pooled AUC. If per-domain AUC rises above ~0.5 here, genuine
# within-domain faithfulness signal exists that the concatenated detectors missed.
#
# Runtime note: encodes passage and answer for all three splits on CPU (~a few
# minutes). MiniLM downloads on first use.
################################################################################

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 9: FAITHFULNESS-AWARE DETECTOR (relational features)")
print("="*80 + "\n")

alpha = 0.10
TFIDF_AUC, BERT_AUC = 0.537, 0.506  # per-domain means from Blocks 4/8 for reference

def conformal_fnr_threshold(fail_scores, a):
    m = len(fail_scores)
    if m == 0:
        return -np.inf
    k = int(np.floor(a * (m + 1)))
    return -np.inf if k < 1 else np.sort(fail_scores)[k - 1]

_word = re.compile(r"[a-z0-9]+")
def lexical_overlap(passage, answer):
    """Fraction of answer content tokens that appear in the passage."""
    p = set(_word.findall(str(passage).lower()))
    a = _word.findall(str(answer).lower())
    if not a:
        return 0.0
    return sum(1 for t in a if t in p) / len(a)

def relational_features(model, df):
    """ONLY relation between passage and answer -- no raw embeddings."""
    emb_p = model.encode(df['passage'].astype(str).tolist(), show_progress_bar=True, batch_size=32)
    emb_a = model.encode(df['answer'].astype(str).tolist(),  show_progress_bar=True, batch_size=32)
    diff = np.abs(emb_p - emb_a)
    prod = emb_p * emb_a
    # cosine per row
    num = (emb_p * emb_a).sum(axis=1)
    den = (np.linalg.norm(emb_p, axis=1) * np.linalg.norm(emb_a, axis=1)) + 1e-9
    cos = (num / den).reshape(-1, 1)
    overlap = np.array([lexical_overlap(p, a)
                        for p, a in zip(df['passage'], df['answer'])]).reshape(-1, 1)
    return np.hstack([diff, prod, cos, overlap])

# ============================================================================
# STEP 1: LOAD + BUILD FEATURES
# ============================================================================

print("[STEP 1/4] Loading splits and building relational features...\n")

train = pd.read_csv("data/processed/train.csv")
calib = pd.read_csv("data/processed/calibration.csv")
test = pd.read_csv("data/processed/test.csv")

print("  Loading MiniLM...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("  Encoding train (passage, answer)...");  X_train = relational_features(model, train)
print("  Encoding calib (passage, answer)...");  X_calib = relational_features(model, calib)
print("  Encoding test  (passage, answer)...");  X_test  = relational_features(model, test)
print(f"\n✓ Feature shape: {X_train.shape[1]} relational features "
      f"(no raw embeddings)\n")

y_train = (train['label'] == 'FAIL').astype(int)

# ============================================================================
# STEP 2: TRAIN + SCORE (label-free)
# ============================================================================

print("[STEP 2/4] Training logistic regression on relational features...\n")

lr = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs', n_jobs=-1)
lr.fit(X_train, y_train)

calib = calib.copy(); test = test.copy()
calib['score'] = lr.predict_proba(X_calib)[:, 1]   # P(FAIL|x), label-free
test['score']  = lr.predict_proba(X_test)[:, 1]

# ============================================================================
# STEP 3: SEPARATING POWER (the question)
# ============================================================================

print("[STEP 3/4] Separating power vs the TF-IDF / BERT baselines...\n")

pooled_auc = roc_auc_score((test.label == 'FAIL').astype(int), test['score'])
per_dom = {}
for dom in sorted(test['source_ds'].unique()):
    d = test[test['source_ds'] == dom]
    yy = (d['label'] == 'FAIL').astype(int).values
    per_dom[dom] = roc_auc_score(yy, d['score'].values) if len(np.unique(yy)) > 1 else np.nan
mean_dom_auc = np.nanmean(list(per_dom.values()))

print("  Per-domain test AUC:")
for dom, a in sorted(per_dom.items(), key=lambda kv: kv[1], reverse=True):
    flag = "  <-- above random" if a >= 0.55 else ""
    print(f"    {dom:14s} {a:.3f}{flag}")
print(f"    {'MEAN':14s} {mean_dom_auc:.3f}")
print(f"\n  Pooled AUC: {pooled_auc:.3f}   |   Mean per-domain AUC: {mean_dom_auc:.3f}")
print(f"  Baselines (mean per-domain): TF-IDF {TFIDF_AUC:.3f}, BERT {BERT_AUC:.3f}")
delta = mean_dom_auc - max(TFIDF_AUC, BERT_AUC)
verdict = ("relational/faithfulness signal FOUND -- escalate to an NLI cross-encoder"
           if mean_dom_auc >= 0.58 else
           "modest/none -- faithfulness needs a stronger model (NLI cross-encoder)")
print(f"  Change vs best baseline: {delta:+.3f}  ->  {verdict}\n")

# ============================================================================
# STEP 4: RUN CRC, SEE IF THE FRONTIER MOVED
# ============================================================================

print("[STEP 4/4] Mondrian FNR control on the new score...\n")

thr = {dom: conformal_fnr_threshold(
          calib.loc[(calib.source_ds == dom) & (calib.label == 'FAIL'), 'score'].values, alpha)
       for dom in sorted(calib['source_ds'].unique())}

rows = []
for dom in sorted(test['source_ds'].unique()):
    d = test[test['source_ds'] == dom]
    flagged = d['score'].values >= thr[dom]
    is_fail = (d['label'] == 'FAIL').values
    tp = (flagged & is_fail).sum(); fn = (~flagged & is_fail).sum()
    fp = (flagged & ~is_fail).sum(); tn = (~flagged & ~is_fail).sum()
    rows.append({'domain': dom,
                 'recall': tp/(tp+fn) if (tp+fn) else np.nan,
                 'far': fp/(fp+tn) if (fp+tn) else np.nan,
                 'auc': per_dom[dom]})
res = pd.DataFrame(rows)
print(res.assign(recall=lambda x: (x.recall*100).round(1),
                 far=lambda x: (x.far*100).round(1),
                 auc=lambda x: x.auc.round(3)).to_string(index=False))
print(f"\n  Mean recall {res.recall.mean()*100:.1f}%   Mean FAR {res.far.mean()*100:.1f}%"
      f"   (TF-IDF was ~90% / ~88%)\n")

# ============================================================================
# FIGURE + SAVE
# ============================================================================

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

doms = sorted(per_dom.keys())
a0 = ax[0]
y = np.arange(len(doms))
a0.barh(y, [per_dom[d] for d in doms], color=['#2ecc71' if per_dom[d] >= 0.55 else '#e74c3c' for d in doms], alpha=0.85)
a0.axvline(0.5, color='black', ls='--', lw=2, label='random')
a0.axvline(TFIDF_AUC, color='#3498db', ls=':', lw=2, label=f'TF-IDF mean ({TFIDF_AUC})')
a0.set_yticks(y); a0.set_yticklabels(doms); a0.set_xlim(0, 1)
a0.set_xlabel('Per-domain AUC'); a0.set_title('Faithfulness Detector: Separating Power', fontweight='bold')
a0.legend(fontsize=8); a0.grid(axis='x', alpha=0.3)

a1 = ax[1]
a1.bar(['TF-IDF', 'BERT', 'Faithfulness'], [TFIDF_AUC, BERT_AUC, mean_dom_auc],
       color=['#e74c3c', '#9b59b6', '#2ecc71'], alpha=0.85)
a1.axhline(0.5, color='black', ls='--', lw=2, label='random')
a1.set_ylabel('Mean per-domain AUC'); a1.set_ylim(0, 1)
a1.set_title('Detector Comparison', fontweight='bold'); a1.legend(fontsize=8); a1.grid(axis='y', alpha=0.3)

plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/09_faithfulness_detector.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/09_faithfulness_detector.png")

test[['id', 'label', 'source_ds', 'score']].to_csv(
    "data/processed/test_predictions_faithfulness.csv", index=False)
res.to_csv("data/processed/faithfulness_evaluation.csv", index=False)
print("✓ Saved scores and evaluation\n")

print("="*80)
print("BLOCK 9 COMPLETE")
print("="*80)
print(f"""
Relational (faithfulness) detector, per-domain mean AUC: {mean_dom_auc:.3f}
  vs TF-IDF {TFIDF_AUC:.3f}, BERT {BERT_AUC:.3f}.

{verdict}.

If this cleared ~0.58+, the within-domain signal is real and an NLI cross-encoder
(passage => answer entailment) is the next step to push it higher and finally give the
conformal guarantee a usable low-FAR operating point. If it did not move, that is itself
informative: surface relational features are not enough, and the task needs a model
trained on entailment/faithfulness supervision.

Files Saved:
  ✓ python/outputs/09_faithfulness_detector.png
  ✓ data/processed/test_predictions_faithfulness.csv
  ✓ data/processed/faithfulness_evaluation.csv
""")
print("="*80 + "\n")