################################################################################
# BLOCK 10: NLI CROSS-ENCODER FAITHFULNESS DETECTOR (zero-shot)
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# Blocks 9/9b showed surface relational features mostly exploit how each benchmark
# was CONSTRUCTED (lexical-overlap shortcut), not transferable faithfulness. This
# block tests genuine faithfulness with a ZERO-SHOT NLI cross-encoder.
#
#   premise    = passage  (the grounding context)
#   hypothesis = answer
#   score      = s(x) = 1 - P(entailment) = P(contradiction) + P(neutral)
#
# A faithful answer should be ENTAILED by the passage; a hallucinated one should not.
# The model never saw HaluBench, so it cannot use construction artifacts -- any
# within-domain separation here is real, transferable faithfulness signal. The score
# is label-free and fed into the same Mondrian FNR control as every other detector.
#
# Runtime note: scores ~6k (passage, answer) pairs on CPU (~5-10 min). The model
# downloads on first run. Small tokenizer (no sentencepiece) to avoid dep issues.
################################################################################

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 10: NLI CROSS-ENCODER FAITHFULNESS DETECTOR (zero-shot)")
print("="*80 + "\n")

alpha = 0.10
BASELINES = {'TF-IDF': 0.537, 'BERT': 0.506}  # mean per-domain AUC, prior detectors

def conformal_fnr_threshold(fail_scores, a):
    m = len(fail_scores)
    if m == 0:
        return -np.inf
    k = int(np.floor(a * (m + 1)))
    return -np.inf if k < 1 else np.sort(fail_scores)[k - 1]

# ============================================================================
# STEP 1: LOAD MODEL + DATA
# ============================================================================

print("[STEP 1/4] Loading NLI cross-encoder and data...\n")

print("  Loading cross-encoder/nli-distilroberta-base ...")
model = CrossEncoder('cross-encoder/nli-distilroberta-base', max_length=256)

# Robustly find which logit index is 'entailment'
id2label = model.model.config.id2label
entail_idx = next(i for i, lab in id2label.items() if 'entail' in str(lab).lower())
print(f"  ✓ Label map: {id2label}  (entailment = index {entail_idx})\n")

calib = pd.read_csv("data/processed/calibration.csv")
test = pd.read_csv("data/processed/test.csv")
print(f"  Calib {len(calib):,}  Test {len(test):,}\n")

# ============================================================================
# STEP 2: SCORE (zero-shot, label-free)
# ============================================================================

print("[STEP 2/4] Scoring (passage => answer entailment)...\n")

def nli_fail_score(df):
    pairs = list(zip(df['passage'].astype(str), df['answer'].astype(str)))
    logits = model.predict(pairs, batch_size=32, show_progress_bar=True,
                           convert_to_numpy=True)
    probs = softmax(logits, axis=1)
    p_entail = probs[:, entail_idx]
    return 1.0 - p_entail   # P(not entailed) = faithfulness-violation score

print("  Calibration set...")
calib = calib.copy(); calib['score'] = nli_fail_score(calib)
print("  Test set...")
test = test.copy(); test['score'] = nli_fail_score(test)
print()

# ============================================================================
# STEP 3: SEPARATING POWER (the test)
# ============================================================================

print("[STEP 3/4] Separating power vs prior detectors...\n")

y_test = (test['label'] == 'FAIL').astype(int).values
pooled_auc = roc_auc_score(y_test, test['score'].values)

per_dom = {}
for dom in sorted(test['source_ds'].unique()):
    m = (test['source_ds'] == dom).values
    yy = y_test[m]
    per_dom[dom] = roc_auc_score(yy, test['score'].values[m]) if len(np.unique(yy)) > 1 else np.nan
mean_dom_auc = np.nanmean(list(per_dom.values()))

print("  Per-domain test AUC (zero-shot NLI):")
for dom, a in sorted(per_dom.items(), key=lambda kv: kv[1], reverse=True):
    flag = "  <-- real signal" if a >= 0.60 else ("  <-- some signal" if a >= 0.55 else "")
    print(f"    {dom:14s} {a:.3f}{flag}")
print(f"    {'MEAN':14s} {mean_dom_auc:.3f}")
print(f"\n  Pooled AUC {pooled_auc:.3f}   Mean per-domain AUC {mean_dom_auc:.3f}")
print(f"  Prior detectors (mean per-domain): "
      + ", ".join(f"{k} {v:.3f}" for k, v in BASELINES.items()) + "\n")

print("  Mean score by label (higher score = less entailed = predicted FAIL):")
for dom in sorted(test['source_ds'].unique()):
    d = test[test['source_ds'] == dom]
    sf = d.loc[d.label == 'FAIL', 'score'].mean()
    sp = d.loc[d.label == 'PASS', 'score'].mean()
    print(f"    {dom:14s} FAIL {sf:.3f}  PASS {sp:.3f}  (gap {sf-sp:+.3f})")
print()

# ============================================================================
# STEP 4: MONDRIAN FNR CONTROL ON THE NLI SCORE
# ============================================================================

print("[STEP 4/4] Mondrian FNR control on the NLI score...\n")

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
                 'precision': tp/(tp+fp) if (tp+fp) else np.nan,
                 'auc': per_dom[dom]})
res = pd.DataFrame(rows)
print(res.assign(recall=lambda x:(x.recall*100).round(1),
                 far=lambda x:(x.far*100).round(1),
                 precision=lambda x:(x.precision*100).round(1),
                 auc=lambda x:x.auc.round(3)).to_string(index=False))
print(f"\n  Mean recall {res.recall.mean()*100:.1f}%   Mean FAR {res.far.mean()*100:.1f}%"
      f"   (TF-IDF was ~90% / ~88%)")
print("  -> If AUC rose and FAR at controlled recall FELL, the better detector bought")
print("     a more usable operating point. That is the payoff.\n")

# ============================================================================
# FIGURE + SAVE
# ============================================================================

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

doms = sorted(per_dom.keys())
a0 = ax[0]; y = np.arange(len(doms))
a0.barh(y, [per_dom[d] for d in doms],
        color=['#2ecc71' if per_dom[d] >= 0.60 else ('#f39c12' if per_dom[d] >= 0.55 else '#e74c3c')
               for d in doms], alpha=0.85)
a0.axvline(0.5, color='black', ls='--', lw=2, label='random')
for k, v in BASELINES.items():
    a0.axvline(v, ls=':', lw=1.5, label=f'{k} ({v})')
a0.set_yticks(y); a0.set_yticklabels(doms); a0.set_xlim(0, 1)
a0.set_xlabel('Per-domain AUC'); a0.set_title('Zero-Shot NLI: Separating Power', fontweight='bold')
a0.legend(fontsize=8); a0.grid(axis='x', alpha=0.3)

a1 = ax[1]
names = list(BASELINES.keys()) + ['NLI (zero-shot)']
vals = list(BASELINES.values()) + [mean_dom_auc]
a1.bar(names, vals, color=['#e74c3c', '#9b59b6', '#2ecc71'], alpha=0.85)
a1.axhline(0.5, color='black', ls='--', lw=2, label='random')
a1.set_ylabel('Mean per-domain AUC'); a1.set_ylim(0, 1)
a1.set_title('Detector Comparison', fontweight='bold'); a1.legend(fontsize=8); a1.grid(axis='y', alpha=0.3)

plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/10_nli_detector.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/10_nli_detector.png")

test[['id', 'label', 'source_ds', 'score']].to_csv(
    "data/processed/test_predictions_nli.csv", index=False)
res.to_csv("data/processed/nli_evaluation.csv", index=False)
print("✓ Saved scores and evaluation\n")

print("="*80)
print("BLOCK 10 COMPLETE")
print("="*80)
print(f"""
Zero-shot NLI faithfulness detector, mean per-domain AUC: {mean_dom_auc:.3f}
  vs TF-IDF 0.537, BERT 0.506, and the (artifact-prone) surface features.

Because the NLI model never saw HaluBench, any within-domain separation here is
genuine faithfulness signal, not a construction shortcut. Per-domain AUC and whether
the controlled-recall FAR dropped (above) tell you if a real entailment detector gives
the conformal guarantee a usable operating point. If promising, swap in the stronger
cross-encoder/nli-deberta-v3-base; if entailment also struggles, the honest conclusion
is that HaluBench faithfulness needs task-specific fine-tuning, not an off-the-shelf model.

Files Saved:
  ✓ python/outputs/10_nli_detector.png
  ✓ data/processed/test_predictions_nli.csv
  ✓ data/processed/nli_evaluation.csv
""")
print("="*80 + "\n")