import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import numpy as np
import os
import matplotlib.ticker as mticker
import argparse


parser = argparse.ArgumentParser(description="Plot top motifs logo from CSV")
parser.add_argument("--file_path", type=str, required=True, help="Path to input CSV (must contain k_mer,methy_lable)")
parser.add_argument("--save_path", type=str, default=None, help="If set, save figure to this path (e.g., out.png)")
parser.add_argument("--dpi", type=int, default=300, help="Save dpi (default: 300)")
args = parser.parse_args()
file_path = args.file_path
save_path = args.save_path
dpi = args.dpi

USE_POSITIVE_ONLY = False 
TOPK = 5
MOTIF_MODE = "kmer13" 

TARGET_POS_MAX = 0.010  
TARGET_NEG_MIN = -0.015  
YMIN, YMAX = TARGET_NEG_MIN - 0.01, TARGET_POS_MAX + 0.002
YTICKS = [0.000, -0.025, -0.050]
YFMT = "%.3f"

# logomaker 
pos_color_scheme = {'A': '#FF0A0A', 'C': '#FFCE5C', 'G': '#52D99C', 'T': '#629BEF'}
neg_color_scheme = {b: (0.75, 0.75, 0.75, 0.55) for b in "ACGT"}  

# =======================
# 1. Load data
# =======================
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df_all = pd.read_csv(file_path, usecols=["k_mer", "methy_lable"])
df_all["k_mer"] = df_all["k_mer"].astype(str)

# Keep only standard 13-mers (A/C/G/T)
df_all = df_all[df_all["k_mer"].str.fullmatch(r"[ACGT]{13}", na=False)].copy()
# Ensure central base is C (index 6, position 0)
df_all = df_all[df_all["k_mer"].str[6] == "C"].copy()

if len(df_all) == 0:
    raise ValueError("No valid 13-mer (ACGT) with central C found in data.")

# Data for plotting (optionally only positive)
df_use = df_all.copy()
if USE_POSITIVE_ONLY:
    df_use = df_use[df_use["methy_lable"] == 1].copy()

if len(df_use) == 0:
    raise ValueError("No sequences available after filtering (possibly empty positive samples).")

# =======================
# 2.Motif grouping + Top5
# =======================
if MOTIF_MODE == "kmer13":
    df_use["motif"] = df_use["k_mer"]
elif MOTIF_MODE == "center5":
    df_use["motif"] = df_use["k_mer"].str.slice(4, 9)  # Central 5-mer (-2..+2)
else:
    raise ValueError("MOTIF_MODE must be 'kmer13' or 'center5'")

top = df_use["motif"].value_counts().head(TOPK)
top_motifs = top.index.tolist()
top_counts = top.values.tolist()
print("Top motifs:", list(zip(top_motifs, top_counts)))


# =======================
# 3. Calculate PFM (13 positions: -6..6)
# =======================
def calculate_pfm(sequences):
    seq_matrix = np.array([list(s) for s in sequences])
    pfm = pd.DataFrame(0.0, index=list(range(-6, 7)), columns=list("ACGT"))
    for i, pos in enumerate(range(-6, 7)):
        counts = pd.Series(seq_matrix[:, i]).value_counts()
        for b in "ACGT":
            pfm.loc[pos, b] = counts.get(b, 0)
    pfm = pfm.div(pfm.sum(axis=1), axis=0).fillna(0)
    return pfm



bg_pfm = calculate_pfm(df_all["k_mer"].tolist())

bg_pfm.loc[0, :] = 0.25
bg_pfm = bg_pfm.div(bg_pfm.sum(axis=1), axis=0).fillna(0)

# =======================
# 4. Precompute scores for all motifs for unified scaling to target heights
# =======================
scores = {}
global_pos_max = 0.0
global_neg_min = 0.0 

for motif in top_motifs:
    group_seqs = df_use.loc[df_use["motif"] == motif, "k_mer"].tolist()
    fg_pfm = calculate_pfm(group_seqs)
    score = fg_pfm - bg_pfm
    scores[motif] = score

    pos_max = score.clip(lower=0).to_numpy().max()
    neg_min = score.clip(upper=0).to_numpy().min() 

    global_pos_max = max(global_pos_max, pos_max)
    global_neg_min = min(global_neg_min, neg_min)


pos_scale = (TARGET_POS_MAX / global_pos_max) if global_pos_max > 0 else 1.0
neg_scale = (abs(TARGET_NEG_MIN) / abs(global_neg_min)) if global_neg_min < 0 else 1.0

print(f"pos_scale={pos_scale:.4f}, neg_scale={neg_scale:.4f}, "
      f"raw_pos_max={global_pos_max:.4f}, raw_neg_min={global_neg_min:.4f}")

# =======================
# 5. Plot motif logos
# =======================
fig, axes = plt.subplots(1, TOPK, figsize=(15, 3), sharey=True)

for j, (motif, cnt) in enumerate(zip(top_motifs, top_counts)):
    ax = axes[j]
    score = scores[motif]

    score_pos = score.clip(lower=0) * pos_scale
    score_neg = score.clip(upper=0) * neg_scale

    logomaker.Logo(score_neg, ax=ax, color_scheme=neg_color_scheme, vpad=.05, stack_order="big_on_top")
    logomaker.Logo(score_pos, ax=ax, color_scheme=pos_color_scheme, vpad=.05, stack_order="big_on_top")


    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.35)

    ax.set_xticks(range(-6, 7))
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks(YTICKS)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(YFMT))
    ax.tick_params(axis="both", labelsize=10, direction="in", width=1.0)

    ax.grid(True, linestyle="--", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    if j == 0:
        ax.set_ylabel("motif score", fontsize=12)

plt.tight_layout()

if save_path is not None and len(save_path) > 0:
    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")

plt.show()
