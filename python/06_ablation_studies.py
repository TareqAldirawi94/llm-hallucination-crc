################################################################################
# BLOCK 6: ABLATION STUDIES
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# Three honest ablations on the label-free score s(x) = P(FAIL|x), FNR control:
#   1. Effect of alpha (test set): the operating-point knob. Recall tracks 1-alpha;
#      FAR is the cost. (3b showed the in-sample ROC frontier; this is the out-of-
#      sample knob.)
#   2. Effect of calibration size: how reliably the per-domain guarantee transfers
#      to test as the number of calibration FAILs shrinks. Explains why small
#      domains (e.g. RAGTruth, ~26 calib FAILs) are fragile. (Replaces the old,
#      ad-hoc 'asymmetric loss' study, which adjusted thresholds by a made-up factor.)
#   3. Mondrian vs global: per-domain recall uniformity at matched recall
#      (same finding as Block 5, quantified as an ablation).
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 6: ABLATION STUDIES")
print("="*80 + "\n")

calib = pd.read_csv("data/processed/calib_predictions.csv")
test = pd.read_csv("data/processed/test_crc_results.csv")
for nm, d in [("calib", calib), ("test", test)]:
    if 'score' not in d.columns:
        raise KeyError(f"'score' column missing in {nm}. Re-run Blocks 2 and 4.")
print(f"✓ Calibration: {len(calib):,}    Test: {len(test):,}\n")

domains = sorted(test['source_ds'].unique())

def conformal_fnr_threshold(fail_scores, alpha):
    m = len(fail_scores)
    if m == 0:
        return -np.inf
    k = int(np.floor(alpha * (m + 1)))
    return -np.inf if k < 1 else np.sort(fail_scores)[k - 1]

def eval_domain(thr_map, frame):
    """Return per-domain recall/FAR given {domain: tau}."""
    out = []
    for dom in domains:
        d = frame[frame['source_ds'] == dom]
        flagged = d['score'].values >= thr_map[dom]
        is_fail = (d['label'] == 'FAIL').values
        tp = (flagged & is_fail).sum(); fn = (~flagged & is_fail).sum()
        fp = (flagged & ~is_fail).sum(); tn = (~flagged & ~is_fail).sum()
        out.append({'domain': dom,
                    'recall': tp/(tp+fn) if (tp+fn) else np.nan,
                    'far': fp/(fp+tn) if (fp+tn) else np.nan})
    return pd.DataFrame(out)

# ============================================================================
# ABLATION 1: EFFECT OF ALPHA (TEST SET)
# ============================================================================

print("[ABLATION 1/3] Effect of alpha (test set)...\n")

alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
abl1 = []
for a in alphas:
    thr = {dom: conformal_fnr_threshold(
              calib.loc[(calib.source_ds == dom) & (calib.label == 'FAIL'), 'score'].values, a)
           for dom in domains}
    ev = eval_domain(thr, test)
    abl1.append({'alpha': a, 'target_recall': 1 - a,
                 'mean_recall': ev['recall'].mean(),
                 'recall_std': ev['recall'].std(),
                 'mean_far': ev['far'].mean()})
    print(f"  alpha={a:.2f} (target recall {int((1-a)*100)}%): "
          f"mean recall {ev['recall'].mean()*100:4.1f}%, "
          f"recall std {ev['recall'].std()*100:4.1f}pp, "
          f"mean FAR {ev['far'].mean()*100:4.1f}%")
abl1_df = pd.DataFrame(abl1)
print()

# ============================================================================
# ABLATION 2: EFFECT OF CALIBRATION SIZE (finite-sample reliability)
# ============================================================================

print("[ABLATION 2/3] Effect of calibration size on the per-domain guarantee...\n")

alpha = 0.10
fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
n_reps = 20
rng = np.random.default_rng(42)
abl2 = []

for frac in fractions:
    devs, recalls = [], []
    for _ in range(n_reps):
        thr = {}
        for dom in domains:
            fs = calib.loc[(calib.source_ds == dom) & (calib.label == 'FAIL'), 'score'].values
            k = max(1, int(round(len(fs) * frac)))
            sub = rng.choice(fs, size=k, replace=False) if k <= len(fs) else fs
            thr[dom] = conformal_fnr_threshold(sub, alpha)
        ev = eval_domain(thr, test)
        # how far each domain's test recall sits from the 1-alpha target
        devs.append((ev['recall'] - (1 - alpha)).abs().mean())
        recalls.append(ev['recall'].mean())
    abl2.append({'fraction': frac,
                 'mean_abs_dev_from_target': np.mean(devs),
                 'dev_std': np.std(devs),
                 'mean_recall': np.mean(recalls)})
    print(f"  calib fraction {frac:4.2f}: mean |recall-target| "
          f"{np.mean(devs)*100:4.1f}pp  (±{np.std(devs)*100:3.1f})   "
          f"mean recall {np.mean(recalls)*100:4.1f}%")
abl2_df = pd.DataFrame(abl2)
print("\n  -> Smaller calibration sets make the per-domain threshold noisier, so test")
print("     recall departs further from target. This is why tiny domains are fragile.\n")

# ============================================================================
# ABLATION 3: MONDRIAN VS GLOBAL (uniformity)
# ============================================================================

print("[ABLATION 3/3] Mondrian vs global stratification...\n")

# Mondrian thresholds (Block 3)
mondrian_thr = dict(pd.read_csv("data/processed/crc_thresholds.csv").values)
# Global: single pooled FAIL-quantile threshold
global_thr_val = conformal_fnr_threshold(calib.loc[calib.label == 'FAIL', 'score'].values, alpha)
global_thr = {dom: global_thr_val for dom in domains}

ev_m = eval_domain(mondrian_thr, test)
ev_g = eval_domain(global_thr, test)

print("  Per-domain recall (%):")
print(f"  {'domain':14s} {'Mondrian':>9s} {'Global':>9s}")
for dom in domains:
    rm = ev_m.set_index('domain').loc[dom, 'recall'] * 100
    rg = ev_g.set_index('domain').loc[dom, 'recall'] * 100
    print(f"  {dom:14s} {rm:8.1f}% {rg:8.1f}%")
print(f"\n  Recall std across domains:  Mondrian {ev_m['recall'].std()*100:.1f}pp"
      f"   vs   Global {ev_g['recall'].std()*100:.1f}pp")
print(f"  Mean recall (matched):      Mondrian {ev_m['recall'].mean()*100:.1f}%"
      f"   vs   Global {ev_g['recall'].mean()*100:.1f}%\n")

# ============================================================================
# FIGURE
# ============================================================================

print("Creating figure...\n")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# A1: recall & FAR vs alpha
ax = axes[0, 0]
ax.plot(abl1_df['alpha'], abl1_df['mean_recall']*100, 'o-', color='#2ecc71', lw=2, label='recall')
ax.plot(abl1_df['alpha'], abl1_df['mean_far']*100, 's-', color='#e67e22', lw=2, label='FAR')
ax.plot(abl1_df['alpha'], abl1_df['target_recall']*100, '--', color='grey', lw=1, label='target recall')
ax.set_xlabel('alpha', fontweight='bold'); ax.set_ylabel('%', fontweight='bold')
ax.set_title('Ablation 1: Effect of alpha (test)', fontweight='bold', fontsize=11)
ax.set_ylim(0, 105); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# A1b: recall uniformity vs alpha
ax = axes[0, 1]
ax.plot(abl1_df['alpha'], abl1_df['recall_std']*100, 'o-', color='#34495e', lw=2)
ax.set_xlabel('alpha', fontweight='bold')
ax.set_ylabel('Recall std across domains (pp)', fontweight='bold')
ax.set_title('Recall Uniformity vs alpha', fontweight='bold', fontsize=11)
ax.grid(alpha=0.3)

# A2: calibration size
ax = axes[1, 0]
ax.errorbar(abl2_df['fraction'], abl2_df['mean_abs_dev_from_target']*100,
            yerr=abl2_df['dev_std']*100, fmt='o-', color='#9b59b6', lw=2, capsize=4)
ax.set_xlabel('Calibration fraction used', fontweight='bold')
ax.set_ylabel('Mean |recall - target| (pp)', fontweight='bold')
ax.set_title('Ablation 2: Calibration Size vs Guarantee Reliability',
             fontweight='bold', fontsize=11)
ax.grid(alpha=0.3)

# A3: Mondrian vs global per-domain recall
ax = axes[1, 1]
y = np.arange(len(domains)); h = 0.38
rm = [ev_m.set_index('domain').loc[d, 'recall']*100 for d in domains]
rg = [ev_g.set_index('domain').loc[d, 'recall']*100 for d in domains]
ax.barh(y - h/2, rg, height=h, color='#3498db', alpha=0.85, label='Global')
ax.barh(y + h/2, rm, height=h, color='#2ecc71', alpha=0.85, label='Mondrian')
ax.axvline(90, color='black', ls='--', lw=2, label='90% target')
ax.set_yticks(y); ax.set_yticklabels(domains)
ax.set_xlabel('Recall (%)', fontweight='bold')
ax.set_title('Ablation 3: Mondrian vs Global (per domain)', fontweight='bold', fontsize=11)
ax.set_xlim(0, 105); ax.legend(fontsize=8); ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/06_ablation_studies.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/06_ablation_studies.png")

abl1_df.to_csv("data/processed/ablation_alpha.csv", index=False)
abl2_df.to_csv("data/processed/ablation_calib_size.csv", index=False)
ev_m.assign(method='Mondrian').to_csv("data/processed/ablation_mondrian.csv", index=False)
ev_g.assign(method='Global').to_csv("data/processed/ablation_global.csv", index=False)
print("✓ Saved ablation CSVs\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("BLOCK 6 COMPLETE")
print("="*80)
print(f"""
Ablation 1 (alpha): recall tracks the 1-alpha target on test; FAR is the cost and
  rises as alpha shrinks. No alpha escapes the high-FAR regime, because the detector
  is near random (AUC ~ 0.5) -- consistent with 3b.

Ablation 2 (calibration size): as the calibration FAIL count shrinks, the per-domain
  threshold gets noisier and test recall departs further from the 90% target. This is
  the finite-sample behaviour of the guarantee, and it explains why small domains
  (RAGTruth ~26 calib FAILs) are the least stable.

Ablation 3 (stratification): at matched recall, Mondrian holds ~90% in every domain
  (recall std {ev_m['recall'].std()*100:.1f}pp) while a single global threshold drifts
  (std {ev_g['recall'].std()*100:.1f}pp). Stratification buys per-domain validity, not
  aggregate detection -- the same conclusion as Block 5.

Files Saved:
  ✓ python/outputs/06_ablation_studies.png
  ✓ data/processed/ablation_alpha.csv, ablation_calib_size.csv,
    ablation_mondrian.csv, ablation_global.csv
""")
print("="*80 + "\n")
