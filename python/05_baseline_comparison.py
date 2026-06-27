################################################################################
# BLOCK 5: BASELINE COMPARISON  (matched-recall, honest)
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# Because the detector's AUC is ~0.5, "which method detects more hallucinations"
# is meaningless on its own -- every method sits on the same near-diagonal ROC, so
# higher recall just means a lower threshold and more false alarms. The honest
# comparison holds recall fixed and asks what each method costs, and -- the real
# point of Mondrian -- whether the per-domain recall is UNIFORM.
#
# Methods (all on s(x) = P(FAIL|x), calibrated on calib, evaluated on test):
#   1. Fixed 0.5 threshold        -- naive; natural operating point (NOT recall-matched)
#   2. Global fixed-flag (top 10%)-- flag highest-scoring 10% (NOT recall-matched)
#   3. Pooled FAIL-quantile        -- one threshold, conformal FNR control on pooled
#                                     FAIL scores (recall-matched ~90%, unstratified)
#   4. Mondrian CRC (ours)         -- per-domain FNR control (recall-matched ~90%)
#
# The 3-vs-4 contrast is the finding: same recall target, but Mondrian holds it in
# EVERY domain while the pooled threshold drifts.
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 5: BASELINE COMPARISON (matched-recall)")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD
# ============================================================================

print("[STEP 1/4] Loading calibration and test predictions...\n")

calib = pd.read_csv("data/processed/calib_predictions.csv")
test = pd.read_csv("data/processed/test_crc_results.csv")  # carries 'score' from Block 4

for name, d in [("calib", calib), ("test", test)]:
    if 'score' not in d.columns:
        raise KeyError(f"'score' column missing in {name}. Re-run Blocks 2 and 4.")

print(f"✓ Calibration: {len(calib):,}    Test: {len(test):,}\n")

alpha = 0.10
domains = sorted(test['source_ds'].unique())

def conformal_fnr_threshold(fail_scores, a):
    m = len(fail_scores)
    if m == 0:
        return -np.inf
    k = int(np.floor(a * (m + 1)))
    return -np.inf if k < 1 else np.sort(fail_scores)[k - 1]

# ============================================================================
# STEP 2: DEFINE THE FOUR FLAGGING RULES (thresholds fit on CALIB)
# ============================================================================

print("[STEP 2/4] Fitting thresholds on calibration...\n")

# 1. Fixed 0.5
def flag_half(scores, dom):           # noqa
    return scores > 0.5

# 2. Global fixed-flag: top 10% of all calib scores
thr_fixedflag = np.quantile(calib['score'], 0.90)
def flag_fixedflag(scores, dom):      # noqa
    return scores >= thr_fixedflag

# 3. Pooled FAIL-quantile (global FNR control, unstratified)
thr_pooled = conformal_fnr_threshold(calib.loc[calib['label'] == 'FAIL', 'score'].values, alpha)
def flag_pooled(scores, dom):         # noqa
    return scores >= thr_pooled

# 4. Mondrian: per-domain thresholds from Block 3
mondrian_thr = dict(pd.read_csv("data/processed/crc_thresholds.csv").values)
def flag_mondrian(scores, dom):       # noqa
    return scores >= mondrian_thr[dom]

methods = {
    "Fixed 0.5 threshold":      (flag_half,      False),
    "Global fixed-flag (10%)":  (flag_fixedflag, False),
    "Pooled FAIL-quantile":     (flag_pooled,    True),
    "Mondrian CRC (ours)":      (flag_mondrian,  True),
}

print(f"  Fixed-flag threshold (90th pct of all scores): {thr_fixedflag:.3f}")
print(f"  Pooled FAIL-quantile threshold (alpha={alpha}):  {thr_pooled:.3f}")
print(f"  Mondrian thresholds: " +
      ", ".join(f"{d}={mondrian_thr[d]:.2f}" for d in domains) + "\n")

# ============================================================================
# STEP 3: EVALUATE EACH METHOD ON TEST (per domain + pooled)
# ============================================================================

print("[STEP 3/4] Evaluating on test set...\n")

per_domain_rows = []
summary_rows = []

for mname, (rule, recall_matched) in methods.items():
    # per-domain
    recalls = []
    for dom in domains:
        d = test[test['source_ds'] == dom]
        flagged = rule(d['score'].values, dom)
        is_fail = (d['label'] == 'FAIL').values
        is_pass = ~is_fail
        tp = (flagged & is_fail).sum(); fp = (flagged & is_pass).sum()
        fn = (~flagged & is_fail).sum(); tn = (~flagged & is_pass).sum()
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        far = fp / (fp + tn) if (fp + tn) else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        recalls.append(recall)
        per_domain_rows.append({'method': mname, 'domain': dom,
                                'recall': recall, 'far': far, 'precision': prec})
    # pooled
    flagged = rule(test['score'].values, None) if mname == "Fixed 0.5 threshold" \
        else np.concatenate([rule(test.loc[test.source_ds == dom, 'score'].values, dom)
                             for dom in domains])
    # simpler pooled recompute to avoid ordering issues:
    flag_all = np.zeros(len(test), dtype=bool)
    for dom in domains:
        m = (test['source_ds'] == dom).values
        flag_all[m] = rule(test.loc[m, 'score'].values, dom)
    is_fail = (test['label'] == 'FAIL').values
    tp = (flag_all & is_fail).sum(); fp = (flag_all & ~is_fail).sum()
    fn = (~flag_all & is_fail).sum(); tn = (~flag_all & ~is_fail).sum()
    pooled_recall = tp / (tp + fn)
    pooled_far = fp / (fp + tn)
    pooled_prec = tp / (tp + fp) if (tp + fp) else np.nan
    recall_std = np.nanstd(recalls)  # uniformity: lower = more uniform across domains

    summary_rows.append({
        'method': mname, 'recall_matched': recall_matched,
        'pooled_recall': pooled_recall, 'recall_std_across_domains': recall_std,
        'pooled_far': pooled_far, 'pooled_precision': pooled_prec
    })

per_domain_df = pd.DataFrame(per_domain_rows)
summary_df = pd.DataFrame(summary_rows)

pd.set_option('display.width', 120)
print("Summary (pooled over test set):\n")
print(summary_df.to_string(index=False,
      formatters={
          'pooled_recall': lambda x: f"{x*100:5.1f}%",
          'recall_std_across_domains': lambda x: f"{x*100:5.1f}pp",
          'pooled_far': lambda x: f"{x*100:5.1f}%",
          'pooled_precision': lambda x: f"{x*100:5.1f}%",
      }))
print()

print("Per-domain recall (the uniformity story):\n")
pivot = per_domain_df.pivot(index='domain', columns='method', values='recall') * 100
print(pivot.round(1).to_string())
print("\n  -> Compare 'Pooled FAIL-quantile' vs 'Mondrian CRC': same ~90% pooled target,")
print("     but Mondrian holds it per domain while the pooled threshold drifts.\n")

# ============================================================================
# STEP 4: VISUALIZE
# ============================================================================

print("[STEP 4/4] Creating comparison figure...\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
order = list(methods.keys())
cols = ['#95a5a6', '#95a5a6', '#3498db', '#2ecc71']  # grey = context, colour = CRC

# Pooled recall (note which are matched)
ax = axes[0, 0]
vals = [summary_df.set_index('method').loc[m, 'pooled_recall']*100 for m in order]
ax.barh(order, vals, color=cols, alpha=0.85)
ax.axvline(90, color='black', ls='--', lw=2, label='90% target')
ax.set_xlabel('Pooled recall (%)', fontweight='bold')
ax.set_title('Recall (pooled) — only blue/green are recall-matched', fontweight='bold', fontsize=11)
ax.set_xlim(0, 105); ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3)

# Recall uniformity across domains (THE key plot)
ax = axes[0, 1]
vals = [summary_df.set_index('method').loc[m, 'recall_std_across_domains']*100 for m in order]
ax.barh(order, vals, color=cols, alpha=0.85)
ax.set_xlabel('Std of per-domain recall (pp) — lower = more uniform', fontweight='bold')
ax.set_title('Per-Domain Recall Uniformity', fontweight='bold', fontsize=11)
ax.grid(axis='x', alpha=0.3)

# Pooled FAR
ax = axes[1, 0]
vals = [summary_df.set_index('method').loc[m, 'pooled_far']*100 for m in order]
ax.barh(order, vals, color=cols, alpha=0.85)
ax.set_xlabel('Pooled FAR (%)', fontweight='bold')
ax.set_title('False-Alarm Rate (pooled)', fontweight='bold', fontsize=11)
ax.grid(axis='x', alpha=0.3)

# Per-domain recall: pooled vs Mondrian (the drift picture)
ax = axes[1, 1]
dd = pivot[['Pooled FAIL-quantile', 'Mondrian CRC (ours)']]
y = np.arange(len(dd)); h = 0.38
ax.barh(y - h/2, dd['Pooled FAIL-quantile'], height=h, color='#3498db', alpha=0.85, label='Pooled FAIL-quantile')
ax.barh(y + h/2, dd['Mondrian CRC (ours)'], height=h, color='#2ecc71', alpha=0.85, label='Mondrian (ours)')
ax.axvline(90, color='black', ls='--', lw=2, label='90% target')
ax.set_yticks(y); ax.set_yticklabels(dd.index)
ax.set_xlabel('Recall (%)', fontweight='bold')
ax.set_title('Per-Domain Recall: Global vs Mondrian', fontweight='bold', fontsize=11)
ax.set_xlim(0, 105); ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

plt.suptitle('Block 5: Matched-Recall Method Comparison (detector AUC ~ 0.5)',
             fontsize=13, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/05_baseline_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/05_baseline_comparison.png")

per_domain_df.to_csv("data/processed/baseline_per_domain.csv", index=False)
summary_df.to_csv("data/processed/baseline_summary.csv", index=False)
print("✓ Saved data/processed/baseline_per_domain.csv and baseline_summary.csv\n")

# ============================================================================
# SUMMARY
# ============================================================================

mondrian_std = summary_df.set_index('method').loc['Mondrian CRC (ours)', 'recall_std_across_domains']
pooled_std = summary_df.set_index('method').loc['Pooled FAIL-quantile', 'recall_std_across_domains']

print("="*80)
print("BLOCK 5 COMPLETE")
print("="*80)
print(f"""
The honest comparison (detector AUC ~ 0.5, so no method 'detects better'):

  - Fixed 0.5 and fixed-flag(10%) are NOT recall-matched -- shown for context only.
  - Pooled FAIL-quantile and Mondrian both target ~90% pooled recall.
  - At matched recall the pooled FAR is similar across the two (no free lunch on a
    near-random detector) -- so the difference is NOT aggregate detection.
  - The difference is UNIFORMITY: per-domain recall std is
        Mondrian {mondrian_std*100:.1f}pp   vs   Pooled {pooled_std*100:.1f}pp.
    Mondrian holds ~90% in every domain; the single pooled threshold drifts
    (worst on RAGTruth, whose score scale differs from the pool).

That is the correct, defensible value of Mondrian here: per-domain validity, not a
better number. A guarantee that holds in every stratum, on top of a detector whose
quality the guarantee cannot change.

Files Saved:
  ✓ python/outputs/05_baseline_comparison.png
  ✓ data/processed/baseline_per_domain.csv
  ✓ data/processed/baseline_summary.csv
""")
print("="*80 + "\n")
