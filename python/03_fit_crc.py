################################################################################
# BLOCK 3: FIT MONDRIAN CRC THRESHOLDS PER DOMAIN  (FNR control)
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# What this does
# --------------
# Score:  s(x) = P(FAIL | x)  from Block 2 (label-free, computable at test time).
# Loss:   false-negative rate (FNR) = P(not flagged | FAIL).
# Goal:   control FNR <= alpha WITHIN each domain (Mondrian stratification),
#         i.e. flag at least a (1 - alpha) fraction of true hallucinations per domain.
#
# Threshold rule (split-conformal, FNR loss)
# ------------------------------------------
# Flag if s(x) >= tau_d. A FAIL is MISSED when s(x) < tau_d. To keep the miss rate
# among FAILs at or below alpha, tau_d is the conformal lower-quantile of the
# domain's calibration FAIL scores at rank k = floor(alpha * (m + 1)), m = #FAIL.
# By exchangeability of the m calibration FAILs and a new test FAIL, the probability
# a new FAIL falls below tau_d is <= alpha, so recall >= 1 - alpha per domain.
#
# Note the cost: lowering tau_d to catch FAILs also flags more PASS examples, so the
# false-alarm rate (FAR) rises -- by how much depends on how well the detector
# separates the classes in that domain. The FAR column below is exactly the price of
# the guarantee, and is the number to look at when choosing alpha.
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 3: FIT MONDRIAN CRC THRESHOLDS PER DOMAIN  (FNR control)")
print("="*80 + "\n")

# ============================================================================
# STEP 1: LOAD CALIBRATION DATA WITH HALLUCINATION SCORES
# ============================================================================

print("[STEP 1/4] Loading calibration data with scores...")

calib_results = pd.read_csv("data/processed/calib_predictions.csv")

if 'score' not in calib_results.columns:
    raise KeyError(
        "Expected a 'score' column (s = P(FAIL|x)) from the corrected Block 2. "
        "Re-run Block 2 first."
    )

print(f"✓ Loaded {len(calib_results):,} calibration samples")
print(f"✓ Columns: {', '.join(calib_results.columns.tolist())}\n")

print("Data summary:")
print(f"  Domains: {calib_results['source_ds'].nunique()}")
print(f"  Domain list: {', '.join(sorted(calib_results['source_ds'].unique()))}")
print(f"  Label balance: {(calib_results['label'] == 'FAIL').sum():,} FAIL, "
      f"{(calib_results['label'] == 'PASS').sum():,} PASS\n")

# ============================================================================
# STEP 2: CONTROL LEVEL
# ============================================================================

# alpha = target false-negative rate per domain. recall target = 1 - alpha.
# This is the knob: smaller alpha = catch more hallucinations = higher false alarms.
alpha = 0.10

print("[STEP 2/4] Control level...")
print(f"  Target FNR per domain (alpha): {alpha*100:.0f}%")
print(f"  => Target recall per domain:    {(1-alpha)*100:.0f}%\n")

domains = sorted(calib_results['source_ds'].unique())

# ============================================================================
# STEP 3: FIT PER-DOMAIN THRESHOLDS ON THE FAIL-SCORE DISTRIBUTION
# ============================================================================

print("[STEP 3/4] Fitting per-domain thresholds (conformal, on FAIL scores)...\n")

def conformal_fnr_threshold(fail_scores, alpha):
    """Lower-quantile threshold so that <= alpha of FAILs fall below it.
    Flag rule downstream is s >= tau. Returns -inf when the sample is too
    small to admit any miss budget (then everything is flagged: recall = 1)."""
    m = len(fail_scores)
    if m == 0:
        return -np.inf
    k = int(np.floor(alpha * (m + 1)))   # conformal rank for FNR <= alpha
    if k < 1:
        return -np.inf
    return np.sort(fail_scores)[k - 1]

crc_thresholds = {}
crc_stats = []

for domain in domains:
    d = calib_results[calib_results['source_ds'] == domain]
    fail_scores = d.loc[d['label'] == 'FAIL', 'score'].values
    pass_scores = d.loc[d['label'] == 'PASS', 'score'].values

    tau = conformal_fnr_threshold(fail_scores, alpha)
    crc_thresholds[domain] = tau

    recall = (fail_scores >= tau).mean() if len(fail_scores) else float('nan')
    far    = (pass_scores >= tau).mean() if len(pass_scores) else float('nan')

    print(f"{domain}:")
    print(f"  Threshold tau: {tau:.3f}")
    print(f"  Recall  (FAIL flagged): {recall*100:5.1f}%   (target >= {(1-alpha)*100:.0f}%)")
    print(f"  FAR     (PASS flagged): {far*100:5.1f}%   <-- cost of the guarantee")
    print(f"  n_FAIL = {len(fail_scores)},  n_PASS = {len(pass_scores)}\n")

    crc_stats.append({
        'domain': domain, 'threshold': tau,
        'recall': recall, 'false_alarm_rate': far,
        'n_fail': len(fail_scores), 'n_pass': len(pass_scores)
    })

crc_stats_df = pd.DataFrame(crc_stats)

# ============================================================================
# STEP 4: APPLY THRESHOLDS AND SUMMARISE (calibration set)
# ============================================================================

print("[STEP 4/4] Applying thresholds on the calibration set...\n")
print("NOTE: recall here is in-sample (same FAILs used to set tau), so it sits at")
print("      ~1-alpha by construction. The honest out-of-sample check is Block 4")
print("      on the held-out TEST set, where recall should land near 1-alpha with")
print("      finite-sample noise.\n")

calib_results['crc_threshold'] = calib_results['source_ds'].map(crc_thresholds)
calib_results['flagged_by_crc'] = calib_results['score'] >= calib_results['crc_threshold']

eval_stats = []
for domain in domains:
    dd = calib_results[calib_results['source_ds'] == domain]
    tp = ((dd['label'] == 'FAIL') & dd['flagged_by_crc']).sum()
    fp = ((dd['label'] == 'PASS') & dd['flagged_by_crc']).sum()
    fn = ((dd['label'] == 'FAIL') & ~dd['flagged_by_crc']).sum()
    tn = ((dd['label'] == 'PASS') & ~dd['flagged_by_crc']).sum()
    n_fail, n_pass = tp + fn, fp + tn

    recall = tp / n_fail if n_fail else float('nan')
    far    = fp / n_pass if n_pass else float('nan')
    prec   = tp / (tp + fp) if (tp + fp) else float('nan')

    print(f"{domain}:")
    print(f"  Recall: {recall*100:5.1f}% ({tp}/{n_fail})   "
          f"FAR: {far*100:5.1f}% ({fp}/{n_pass})   Precision: {prec*100:5.1f}%")

    eval_stats.append({
        'domain': domain, 'recall': recall, 'false_alarm_rate': far,
        'precision': prec, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    })

eval_stats_df = pd.DataFrame(eval_stats)
print()

# ============================================================================
# VISUALISATIONS
# ============================================================================

print("Creating visualizations...\n")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Plot 1: thresholds by domain
ax = axes[0, 0]
s = crc_stats_df.sort_values('threshold')
ax.barh(s['domain'], s['threshold'], color='#34495e', alpha=0.85)
ax.set_xlabel('Threshold tau = flag if P(FAIL|x) >= tau', fontweight='bold')
ax.set_title('Per-Domain FNR-Control Thresholds', fontweight='bold', fontsize=12)
ax.grid(axis='x', alpha=0.3)

# Plot 2: achieved recall vs target
ax = axes[0, 1]
s = eval_stats_df.sort_values('recall')
ax.barh(s['domain'], s['recall']*100, color='#2ecc71', alpha=0.85)
ax.axvline((1-alpha)*100, color='black', linestyle='--', linewidth=2,
           label=f'Target ({(1-alpha)*100:.0f}%)')
ax.set_xlabel('Recall (FAIL flagged, %)', fontweight='bold')
ax.set_title('Achieved Recall by Domain', fontweight='bold', fontsize=12)
ax.set_xlim(0, 105)
ax.legend(); ax.grid(axis='x', alpha=0.3)

# Plot 3: false-alarm cost
ax = axes[1, 0]
s = eval_stats_df.sort_values('false_alarm_rate')
ax.barh(s['domain'], s['false_alarm_rate']*100, color='#e67e22', alpha=0.85)
ax.set_xlabel('False-Alarm Rate (PASS flagged, %)', fontweight='bold')
ax.set_title('Cost of the Guarantee: FAR by Domain', fontweight='bold', fontsize=12)
ax.grid(axis='x', alpha=0.3)

# Plot 4: recall vs FAR
ax = axes[1, 1]
ax.scatter(eval_stats_df['false_alarm_rate']*100, eval_stats_df['recall']*100,
           s=200, alpha=0.75, c=range(len(eval_stats_df)), cmap='viridis')
for i, dom in enumerate(eval_stats_df['domain']):
    ax.annotate(dom, (eval_stats_df.iloc[i]['false_alarm_rate']*100,
                      eval_stats_df.iloc[i]['recall']*100),
                fontsize=9, ha='center')
ax.axhline((1-alpha)*100, color='red', linestyle='--', linewidth=1, alpha=0.6,
           label=f'Recall target ({(1-alpha)*100:.0f}%)')
ax.set_xlabel('False-Alarm Rate (%)', fontweight='bold')
ax.set_ylabel('Recall (%)', fontweight='bold')
ax.set_title('Recall vs False Alarm (per domain)', fontweight='bold', fontsize=12)
ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend()

plt.tight_layout()
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/03_crc_thresholds.png", dpi=300, bbox_inches='tight')
print("✓ Saved visualization to python/outputs/03_crc_thresholds.png\n")

# ============================================================================
# SAVE
# ============================================================================

print("Saving results...\n")

pd.DataFrame(
    [{'domain': dom, 'crc_threshold': tau} for dom, tau in crc_thresholds.items()]
).to_csv("data/processed/crc_thresholds.csv", index=False)
print("✓ Saved thresholds to data/processed/crc_thresholds.csv")

eval_stats_df.to_csv("data/processed/crc_evaluation.csv", index=False)
print("✓ Saved evaluation to data/processed/crc_evaluation.csv")

calib_results[['id', 'label', 'source_ds', 'pred_prob_fail', 'score',
               'crc_threshold', 'flagged_by_crc']].to_csv(
    "data/processed/calib_crc_results.csv", index=False)
print("✓ Saved calibration CRC results to data/processed/calib_crc_results.csv\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("BLOCK 3 COMPLETE - SUMMARY")
print("="*80 + "\n")

mean_recall = eval_stats_df['recall'].mean()
mean_far = eval_stats_df['false_alarm_rate'].mean()

print(f"""
Mondrian Conformal Risk Control - per-domain FNR control

Control level: alpha = {alpha*100:.0f}%  (target recall {(1-alpha)*100:.0f}% per domain)
Score:         s(x) = P(FAIL | x)   (label-free; identical on calib and test)
Threshold:     flag if s(x) >= tau_d ;  tau_d calibrated on each domain's FAIL scores

Calibration-set summary (in-sample; see Block 4 for the held-out check):
  - Mean recall across domains: {mean_recall*100:.1f}%   (sits near target by construction)
  - Mean false-alarm rate:      {mean_far*100:.1f}%   <-- this is the cost; read per domain

How to read this:
  - Recall is pinned near {(1-alpha)*100:.0f}% in every domain -- that is the guarantee working.
  - The FAR column is the price. Where the detector separates classes well
    (e.g. RAGTruth) FAR stays low; where it is near chance, FAR is high.
  - If the FAR is unacceptable on the weak domains, raise alpha (accept lower recall)
    or improve the base detector. That trade is the substantive finding.

Files Saved:
  ✓ data/processed/crc_thresholds.csv
  ✓ data/processed/crc_evaluation.csv
  ✓ data/processed/calib_crc_results.csv
  ✓ python/outputs/03_crc_thresholds.png

Next: Block 4 - apply tau_d to the TEST set (must compute s = P(FAIL|x) the same way).
""")

print("="*80 + "\n")
print("✓ Block 3 complete.\n")

################################################################################
# BLOCK 3b: ALPHA SWEEP - per-domain recall / FAR frontier
# Author: Tareq Aldirawi
# Date: June 2026
# Language: Python
################################################################################
#
# Why this exists
# ---------------
# FNR control at a single alpha gives one operating point per domain. Sweeping
# alpha traces the whole frontier: for each target recall (1 - alpha), what false
# alarm rate does each domain pay? Because the conformal threshold is monotone in
# alpha, every one of these operating points lies on the domain's ROC curve. So
# the frontier per domain == the ROC curve, and the per-domain AUC is the single
# number that says whether ANY usable operating point exists:
#     AUC ~ 0.50  ->  curve hugs the diagonal, no threshold beats random.
#     AUC ->  1.0  ->  high recall reachable at low FAR.
#
# This is an in-sample (calibration) diagnostic. It characterises the detector,
# which does not change between calib and test, so it is the right place to read
# off the frontier.
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 3b: ALPHA SWEEP - per-domain recall / FAR frontier")
print("="*80 + "\n")

# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
calib = pd.read_csv("data/processed/calib_predictions.csv")
if 'score' not in calib.columns:
    raise KeyError("Expected 'score' column from corrected Block 2. Re-run Block 2.")

domains = sorted(calib['source_ds'].unique())
alphas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

print(f"Domains: {', '.join(domains)}")
print(f"Alpha grid: {alphas}\n")

# ----------------------------------------------------------------------------
# Conformal FNR threshold (same rule as Block 3)
# ----------------------------------------------------------------------------
def conformal_fnr_threshold(fail_scores, alpha):
    m = len(fail_scores)
    if m == 0:
        return -np.inf
    k = int(np.floor(alpha * (m + 1)))
    if k < 1:
        return -np.inf
    return np.sort(fail_scores)[k - 1]

# ----------------------------------------------------------------------------
# Sweep + per-domain AUC
# ----------------------------------------------------------------------------
rows = []
auc_by_domain = {}

for dom in domains:
    d = calib[calib['source_ds'] == dom]
    fail_scores = d.loc[d['label'] == 'FAIL', 'score'].values
    pass_scores = d.loc[d['label'] == 'PASS', 'score'].values
    y = (d['label'] == 'FAIL').astype(int).values
    s = d['score'].values

    auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float('nan')
    auc_by_domain[dom] = auc

    for a in alphas:
        tau = conformal_fnr_threshold(fail_scores, a)
        recall = (fail_scores >= tau).mean() if len(fail_scores) else float('nan')
        far = (pass_scores >= tau).mean() if len(pass_scores) else float('nan')
        rows.append({
            'domain': dom, 'alpha': a, 'target_recall': 1 - a,
            'recall': recall, 'far': far, 'auc': auc,
            'n_fail': len(fail_scores), 'n_pass': len(pass_scores)
        })

sweep = pd.DataFrame(rows)

# ----------------------------------------------------------------------------
# Console: AUC summary (the headline number)
# ----------------------------------------------------------------------------
print("Per-domain separating power (ROC AUC; 0.50 = random):\n")
for dom in sorted(auc_by_domain, key=auc_by_domain.get, reverse=True):
    n_fail = int(sweep.loc[sweep.domain == dom, 'n_fail'].iloc[0])
    flag = "  <-- near random" if auc_by_domain[dom] < 0.55 else ""
    print(f"  {dom:14s}  AUC = {auc_by_domain[dom]:.3f}   (n_FAIL={n_fail}){flag}")
print()

# ----------------------------------------------------------------------------
# Console: FAR matrix (domains x alpha). Recall ~ 1-alpha by construction,
# so FAR is the informative axis.
# ----------------------------------------------------------------------------
print("False-alarm rate (%) to reach each recall target (1 - alpha):\n")
header = "  domain        " + "".join(f"  r={int((1-a)*100):>3d}%" for a in alphas)
print(header)
print("  " + "-"*(len(header)-2))
for dom in domains:
    cells = ""
    for a in alphas:
        far = sweep[(sweep.domain == dom) & (sweep.alpha == a)]['far'].iloc[0]
        cells += f"  {far*100:5.0f} "
    print(f"  {dom:14s}{cells}")
print("\n  (Read across: as the recall target rises, FAR rises. A near-random")
print("   detector pushes FAR toward 100% for any useful recall.)\n")

# ----------------------------------------------------------------------------
# Figure: per-domain ROC with conformal operating points overlaid
# ----------------------------------------------------------------------------
print("Creating frontier figure...\n")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()

for i, dom in enumerate(domains):
    ax = axes[i]
    d = calib[calib['source_ds'] == dom]
    y = (d['label'] == 'FAIL').astype(int).values
    s = d['score'].values

    # Full ROC curve (FAR = FPR on x, recall = TPR on y)
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr*100, tpr*100, color='#2c3e50', lw=2, label='ROC')

    # Conformal FNR-control operating points across alpha
    sub = sweep[sweep.domain == dom].sort_values('alpha')
    ax.scatter(sub['far']*100, sub['recall']*100, color='#e74c3c', s=45, zorder=5,
               label='FNR-control points')
    # annotate the alpha=0.10 point
    p = sub[sub.alpha == 0.10]
    if len(p):
        ax.annotate('α=0.10', (p['far'].iloc[0]*100, p['recall'].iloc[0]*100),
                    textcoords="offset points", xytext=(-38, -2), fontsize=8,
                    color='#e74c3c')

    # Random-detector diagonal
    ax.plot([0, 100], [0, 100], '--', color='grey', lw=1, label='random')

    ax.set_title(f"{dom}  (AUC={auc_by_domain[dom]:.2f})", fontweight='bold', fontsize=11)
    ax.set_xlabel('False-Alarm Rate (%)')
    ax.set_ylabel('Recall (%)')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8, loc='lower right')

plt.suptitle('Per-Domain Recall / FAR Frontier  (ROC + conformal FNR-control points)',
             fontweight='bold', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
Path("python/outputs").mkdir(parents=True, exist_ok=True)
plt.savefig("python/outputs/03b_alpha_sweep_frontier.png", dpi=300, bbox_inches='tight')
print("✓ Saved figure to python/outputs/03b_alpha_sweep_frontier.png")

# ----------------------------------------------------------------------------
# Save sweep
# ----------------------------------------------------------------------------
sweep.to_csv("data/processed/alpha_sweep.csv", index=False)
print("✓ Saved sweep to data/processed/alpha_sweep.csv\n")

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
mean_auc = np.nanmean(list(auc_by_domain.values()))
print("="*80)
print("BLOCK 3b COMPLETE - SUMMARY")
print("="*80)
print(f"""
Mean per-domain AUC: {mean_auc:.3f}   (0.50 = random, 1.00 = perfect)

Interpretation:
  - The ROC curve IS the achievable frontier; the conformal FNR-control points
    sit on it. Where the curve hugs the diagonal, every operating point trades
    recall for FAR almost 1:1 -- there is no 'good' alpha, only a choice of how
    much to flag.
  - This is the substantive result: Mondrian CRC delivers a valid per-domain FNR
    guarantee, but the price (FAR) is set by detector quality, which is near
    random on most HaluBench domains. The fix is a better detector, not a
    different alpha.

Files Saved:
  ✓ python/outputs/03b_alpha_sweep_frontier.png
  ✓ data/processed/alpha_sweep.csv
""")
print("="*80 + "\n")
