################################################################################
# BLOCK 1: LOAD HALUBENCH, EDA, CREATE TRAIN/CALIB/TEST SPLITS
# Author: Tareq Aldirawi
# Date: June 2026
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("BLOCK 1: LOAD & EDA - HaluBench Hallucination Detection Dataset")
print("="*80 + "\n")

# ============================================================================
# STEP 1: SET WORKING DIRECTORY & CREATE FOLDERS
# ============================================================================

print("[STEP 1/6] Setting up directories...\n")

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("python/outputs").mkdir(parents=True, exist_ok=True)

print("✓ Directories created\n")

# ============================================================================
# STEP 2: LOAD PARQUET FILE
# ============================================================================

print("[STEP 2/6] Loading HaluBench from parquet file...\n")

try:
    import pyarrow.parquet as pq
    df = pd.read_parquet("halubench_raw.parquet")
except ImportError:
    print("Installing pyarrow...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyarrow"])
    df = pd.read_parquet("halubench_raw.parquet")

df = df.reset_index(drop=True)

print(f"✓ Successfully loaded")
print(f"✓ Data shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("[STEP 3/6] Exploratory Data Analysis...\n")

print("Column names:")
print(df.columns.tolist())
print()

print("First example:")
print(f"  ID: {df['id'].iloc[0]}")
print(f"  Passage: {df['passage'].iloc[0][:80]}...")
print(f"  Question: {df['question'].iloc[0]}")
print(f"  Answer: {df['answer'].iloc[0][:80]}...")
print(f"  Label: {df['label'].iloc[0]}")
print(f"  Source: {df['source_ds'].iloc[0]}\n")

# Label distribution
print("Label distribution:")
label_dist = df['label'].value_counts().sort_values(ascending=False)
print(label_dist)
pass_count = label_dist.get('PASS', 0)
fail_count = label_dist.get('FAIL', 0)
pass_pct = pass_count / len(df) * 100 if pass_count > 0 else 0
fail_pct = fail_count / len(df) * 100 if fail_count > 0 else 0
print(f"  Balance: {pass_pct:.1f}% PASS ({pass_count:,}), {fail_pct:.1f}% FAIL ({fail_count:,})\n")

# Domain distribution
print("Domain/Source distribution:")
domain_dist = df['source_ds'].value_counts().sort_values(ascending=False)
print(domain_dist)
print()

# Text statistics
print("Text statistics:")
print(f"  Passage length - Mean: {df['passage'].str.len().mean():.0f}, Max: {df['passage'].str.len().max():.0f}")
print(f"  Question length - Mean: {df['question'].str.len().mean():.0f}, Max: {df['question'].str.len().max():.0f}")
print(f"  Answer length - Mean: {df['answer'].str.len().mean():.0f}, Max: {df['answer'].str.len().max():.0f}\n")

# ============================================================================
# STEP 4: CREATE TRAIN/CALIBRATION/TEST SPLITS (60/20/20)
# ============================================================================

print("[STEP 4/6] Creating stratified train/calibration/test splits (60/20/20)...\n")

from sklearn.model_selection import train_test_split

np.random.seed(42)

# First split: train (60%) vs temp (40%)
train, temp = train_test_split(
    df,
    test_size=0.4,
    random_state=42,
    stratify=df['source_ds']
)

# Second split: calib (20%) vs test (20%)
calib, test = train_test_split(
    temp,
    test_size=0.5,
    random_state=42,
    stratify=temp['source_ds']
)

print(f"  Train: {len(train):,} samples ({len(train)/len(df)*100:.1f}%)")
print(f"  Calibration: {len(calib):,} samples ({len(calib)/len(df)*100:.1f}%)")
print(f"  Test: {len(test):,} samples ({len(test)/len(df)*100:.1f}%)\n")

# Verify stratification by domain
print("Domain distribution in splits:")
print("\nTrain:")
print(train['source_ds'].value_counts().sort_values(ascending=False))
print("\nCalibration:")
print(calib['source_ds'].value_counts().sort_values(ascending=False))
print("\nTest:")
print(test['source_ds'].value_counts().sort_values(ascending=False))
print()

# Save splits to CSV
train.to_csv("data/processed/train.csv", index=False)
calib.to_csv("data/processed/calibration.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("✓ Splits saved to data/processed/\n")

# ============================================================================
# STEP 5: CREATE VISUALIZATIONS
# ============================================================================

print("[STEP 5/6] Creating visualizations...\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Label distribution
p1 = axes[0, 0]
label_counts = df['label'].value_counts()
colors_label = {'FAIL': '#e74c3c', 'PASS': '#2ecc71'}
label_colors = [colors_label.get(label, '#3498db') for label in label_counts.index]
p1.bar(label_counts.index, label_counts.values, color=label_colors, alpha=0.8)
p1.set_title('Label Distribution (PASS vs FAIL)', fontweight='bold', fontsize=12)
p1.set_xlabel('Label')
p1.set_ylabel('Count')
p1.grid(axis='y', alpha=0.3)

# Plot 2: Domain distribution
p2 = axes[0, 1]
domain_counts = df['source_ds'].value_counts().sort_values()
p2.barh(domain_counts.index, domain_counts.values, color='#3498db', alpha=0.8)
p2.set_title('Domain Distribution', fontweight='bold', fontsize=12)
p2.set_xlabel('Count')
p2.grid(axis='x', alpha=0.3)

# Plot 3: Label by domain
p3 = axes[1, 0]
domain_label = pd.crosstab(df['source_ds'], df['label'])
domain_label.plot(kind='barh', ax=p3, color=['#e74c3c', '#2ecc71'], alpha=0.8)
p3.set_title('Label Distribution by Domain', fontweight='bold', fontsize=12)
p3.set_xlabel('Count')
p3.set_ylabel('Domain')
p3.legend(title='Label')
p3.grid(axis='x', alpha=0.3)

# Plot 4: Text lengths
p4 = axes[1, 1]
p4.hist(df['passage'].str.len(), bins=50, alpha=0.6, label='Passage', color='#3498db')
p4.hist(df['question'].str.len(), bins=50, alpha=0.6, label='Question', color='#9b59b6')
p4.hist(df['answer'].str.len(), bins=50, alpha=0.6, label='Answer', color='#e67e22')
p4.set_title('Text Length Distributions', fontweight='bold', fontsize=12)
p4.set_xlabel('Length (characters)')
p4.set_ylabel('Frequency')
p4.legend()
p4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("python/outputs/01_eda_overview.png", dpi=300, bbox_inches='tight')
print("✓ Saved visualization to python/outputs/01_eda_overview.png\n")
plt.close()

# ============================================================================
# STEP 6: SUMMARY
# ============================================================================

print("[STEP 6/6] Summary\n")

print("="*80)
print("BLOCK 1 COMPLETE - SUMMARY")
print("="*80 + "\n")

print(f"""
Dataset: HaluBench (PatronusAI)
Total samples: {len(df):,}
Domains: {', '.join(sorted(df['source_ds'].unique()))}

Label breakdown:
  - PASS (Faithful): {pass_count:,} ({pass_pct:.1f}%)
  - FAIL (Hallucination): {fail_count:,} ({fail_pct:.1f}%)

Splits created (stratified by domain):
  - Train: {len(train):,} samples → Use for detector training
  - Calibration: {len(calib):,} samples → Use for CRC threshold calibration
  - Test: {len(test):,} samples → Use for final evaluation

Files saved:
  ✓ data/processed/train.csv
  ✓ data/processed/calibration.csv
  ✓ data/processed/test.csv
  ✓ python/outputs/01_eda_overview.png

Next Step: Block 2 - Train hallucination detector
""")

print("="*80 + "\n")

print("✓ Block 1 complete!\n")
