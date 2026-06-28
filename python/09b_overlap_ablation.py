################################################################################
# BLOCK 9b: OVERLAP-ABLATION DIAGNOSTIC
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# Block 9's relational detector hit ~0.97 AUC on halueval while other domains sat
# at 0.5-0.65. That pattern usually means a SHORTCUT, not faithfulness. The prime
# suspect is lexical overlap: if a domain's hallucinations were built by editing the
# answer's word overlap with the passage, "fraction of answer tokens in passage"
# becomes a near-giveaway for that domain's construction method -- not generalizable.
#
# This block splits the features three ways and reports per-domain test AUC:
#   A. Full          = |p-a|, p*a, cosine, overlap   (reproduces Block 9)
#   B. Embedding-only = |p-a|, p*a, cosine           (drop overlap)
#   C. Overlap-only   = overlap                        (single feature)
# Plus mean overlap for FAIL vs PASS per domain -- the direct artifact check.
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
print("BLOCK 9b: OVERLAP-ABLATION DIAGNOSTIC")
print("="*80 + "\n")

_word = re.compile(r"[a-z0-9]+")
def lexical_overlap(passage, answer):
    p = set(_word.findall(str(passage).lower()))
    a = _word.findall(str(answer).lower())
    return sum(1 for t in a if t in p) / len(a) if a else 0.0

def components(model, df):
    emb_p = model.encode(df['passage'].astype(str).tolist(), show_progress_bar=True, batch_size=32)
    emb_a = model.encode(df['answer'].astype(str).tolist(),  show_progress_bar=True, batch_size=32)
    diff = np.abs(emb_p - emb_a)
    prod = emb_p * emb_a
    num = (emb_p * emb_a).sum(axis=1)
    den = (np.linalg.norm(emb_p, axis=1) * np.linalg.norm(emb_a, axis=1)) + 1e-9
    cos = (num / den).reshape(-1, 1)
    ov = np.array([lexical_overlap(p, a) for p, a in zip(df['passage'], df['answer'])]).reshape(-1, 1)
    return diff, prod, cos, ov

# ----------------------------------------------------------------------------
print("[1/3] Encoding train and test (passage, answer separately)...\n")
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("  train..."); d_tr, p_tr, c_tr, o_tr = components(model, train)
print("  test...");  d_te, p_te, c_te, o_te = components(model, test)
print()

y_train = (train['label'] == 'FAIL').astype(int)
y_test = (test['label'] == 'FAIL').astype(int).values
domains = sorted(test['source_ds'].unique())

feature_sets = {
    'A. Full':           (np.hstack([d_tr, p_tr, c_tr, o_tr]), np.hstack([d_te, p_te, c_te, o_te])),
    'B. Embedding-only': (np.hstack([d_tr, p_tr, c_tr]),       np.hstack([d_te, p_te, c_te])),
    'C. Overlap-only':   (o_tr,                                 o_te),
}

# ----------------------------------------------------------------------------
print("[2/3] Per-domain test AUC by feature set...\n")

aucs = {}  # name -> {domain: auc}
for name, (Xtr, Xte) in feature_sets.items():
    lr = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs', n_jobs=-1)
    lr.fit(Xtr, y_train)
    s = lr.predict_proba(Xte)[:, 1]
    aucs[name] = {}
    for dom in domains:
        m = (test['source_ds'] == dom).values
        yy = y_test[m]
        aucs[name][dom] = roc_auc_score(yy, s[m]) if len(np.unique(yy)) > 1 else np.nan

hdr = f"  {'domain':14s}" + "".join(f"{n.split('.')[0]:>8s}" for n in feature_sets)
print(hdr); print("  " + "-"*(len(hdr)-2))
for dom in domains:
    row = f"  {dom:14s}"
    for name in feature_sets:
        row += f"{aucs[name][dom]:8.3f}"
    print(row)
means = {name: np.nanmean(list(aucs[name].values())) for name in feature_sets}
print("  " + "-"*(len(hdr)-2))
print(f"  {'MEAN':14s}" + "".join(f"{means[n]:8.3f}" for n in feature_sets))
print()

# ----------------------------------------------------------------------------
print("[3/3] Mean lexical overlap, FAIL vs PASS (the artifact check)...\n")
test_ov = test.copy()
test_ov['overlap'] = o_te.ravel()
print(f"  {'domain':14s} {'FAIL ovl':>9s} {'PASS ovl':>9s} {'gap':>7s}")
print("  " + "-"*42)
for dom in domains:
    d = test_ov[test_ov['source_ds'] == dom]
    of = d.loc[d.label == 'FAIL', 'overlap'].mean()
    op = d.loc[d.label == 'PASS', 'overlap'].mean()
    star = "   <-- giveaway" if abs(of - op) > 0.20 else ""
    print(f"  {dom:14s} {of:9.3f} {op:9.3f} {of-op:+7.3f}{star}")
print()

# ----------------------------------------------------------------------------
# Verdict on halueval specifically
hal_full = aucs['A. Full'].get('halueval', np.nan)
hal_emb  = aucs['B. Embedding-only'].get('halueval', np.nan)
hal_ovl  = aucs['C. Overlap-only'].get('halueval', np.nan)

print("="*80)
print("BLOCK 9b COMPLETE - DIAGNOSIS")
print("="*80)
print(f"""
halueval AUC:  full {hal_full:.3f}   embedding-only {hal_emb:.3f}   overlap-only {hal_ovl:.3f}

Reading:
  - If overlap-only is high AND embedding-only collapses toward ~0.5, halueval's score
    is riding on lexical overlap -- a construction artifact of how its hallucinations
    were built, not transferable faithfulness signal. Report it honestly and base the
    headline on the embedding-only (set B) per-domain mean: {means['B. Embedding-only']:.3f}.
  - If embedding-only stays high, the signal is real and overlap is a bonus.

Mean per-domain AUC:  Full {means['A. Full']:.3f}   Embedding-only {means['B. Embedding-only']:.3f}
  (TF-IDF 0.537, BERT 0.506)

The honest headline number is the embedding-only mean -- it excludes the overlap
shortcut and still measures genuine within-domain faithfulness signal.
""")
print("="*80 + "\n")

# small figure
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(domains)); w = 0.26
for i, name in enumerate(feature_sets):
    ax.bar(x + (i-1)*w, [aucs[name][d] for d in domains], w, label=name)
ax.axhline(0.5, color='black', ls='--', lw=2)
ax.set_xticks(x); ax.set_xticklabels(domains, rotation=30, ha='right')
ax.set_ylabel('Per-domain test AUC'); ax.set_ylim(0, 1)
ax.set_title('Overlap Ablation: what drives each domain', fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/09b_overlap_ablation.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/09b_overlap_ablation.png\n")