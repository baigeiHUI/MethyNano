import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import numpy as np
import os
import matplotlib.ticker as mticker
import argparse

# =======================
# 0. 配置
# =======================
parser = argparse.ArgumentParser(description="Plot top motifs logo from CSV")
parser.add_argument("--file_path", type=str, required=True, help="Path to input CSV (must contain k_mer,methy_lable)")
parser.add_argument("--save_path", type=str, default=None, help="If set, save figure to this path (e.g., out.png)")
parser.add_argument("--dpi", type=int, default=300, help="Save dpi (default: 300)")
args = parser.parse_args()
file_path = args.file_path
save_path = args.save_path
dpi = args.dpi

USE_POSITIVE_ONLY = False  # True: 只用 methy_lable==1 来找Top5并作图；False: 全部
TOPK = 5
MOTIF_MODE = "kmer13"  # "kmer13" 或 "center5"

# 目标显示范围（模仿你截图：0线靠上）
TARGET_POS_MAX = 0.010  # 上面彩色最高到 0.01 左右
TARGET_NEG_MIN = -0.015  # 下面灰色最低到 -0.05 左右
YMIN, YMAX = TARGET_NEG_MIN - 0.01, TARGET_POS_MAX + 0.002
YTICKS = [0.000, -0.025, -0.050]
YFMT = "%.3f"

# logomaker 配色
pos_color_scheme = {'A': '#FF0A0A', 'C': '#FFCE5C', 'G': '#52D99C', 'T': '#629BEF'}
neg_color_scheme = {b: (0.75, 0.75, 0.75, 0.55) for b in "ACGT"}  # 灰+半透明

# =======================
# 1. 读数据
# =======================
if not os.path.exists(file_path):
    raise FileNotFoundError(f"未找到文件: {file_path}")

df_all = pd.read_csv(file_path, usecols=["k_mer", "methy_lable"])
df_all["k_mer"] = df_all["k_mer"].astype(str)

# 只保留标准13-mer（A/C/G/T）
df_all = df_all[df_all["k_mer"].str.fullmatch(r"[ACGT]{13}", na=False)].copy()
# 推荐：确保中心位点是 C（index=6 对应位置0）
df_all = df_all[df_all["k_mer"].str[6] == "C"].copy()

if len(df_all) == 0:
    raise ValueError("数据里没有合法的 13-mer (ACGT) 或中心不是C。")

# 作图用的数据（可选只取阳性）
df_use = df_all.copy()
if USE_POSITIVE_ONLY:
    df_use = df_use[df_use["methy_lable"] == 1].copy()

if len(df_use) == 0:
    raise ValueError("过滤后没有可用序列（可能正样本为空）。")

# =======================
# 2. motif 分组 + Top5
# =======================
if MOTIF_MODE == "kmer13":
    df_use["motif"] = df_use["k_mer"]
elif MOTIF_MODE == "center5":
    df_use["motif"] = df_use["k_mer"].str.slice(4, 9)  # 中心5-mer（-2..+2）
else:
    raise ValueError("MOTIF_MODE 只能是 'kmer13' 或 'center5'")

top = df_use["motif"].value_counts().head(TOPK)
top_motifs = top.index.tolist()
top_counts = top.values.tolist()
print("Top motifs:", list(zip(top_motifs, top_counts)))


# =======================
# 3. 计算 PFM（13位置：-6..6）
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


# 背景建议用全体数据（更稳定）
bg_pfm = calculate_pfm(df_all["k_mer"].tolist())

# ✅ 关键：中心位点背景设为均匀，中心C就不会被抵消成0
bg_pfm.loc[0, :] = 0.25
bg_pfm = bg_pfm.div(bg_pfm.sum(axis=1), axis=0).fillna(0)

# =======================
# 4. 先预计算所有 motif 的 score，用于统一缩放到目标高度
# =======================
scores = {}
global_pos_max = 0.0
global_neg_min = 0.0  # 负数

for motif in top_motifs:
    group_seqs = df_use.loc[df_use["motif"] == motif, "k_mer"].tolist()
    fg_pfm = calculate_pfm(group_seqs)
    score = fg_pfm - bg_pfm
    scores[motif] = score

    pos_max = score.clip(lower=0).to_numpy().max()
    neg_min = score.clip(upper=0).to_numpy().min()  # 最负

    global_pos_max = max(global_pos_max, pos_max)
    global_neg_min = min(global_neg_min, neg_min)

# 分别缩放正/负到目标范围（这一步就是你截图效果的关键）
pos_scale = (TARGET_POS_MAX / global_pos_max) if global_pos_max > 0 else 1.0
neg_scale = (abs(TARGET_NEG_MIN) / abs(global_neg_min)) if global_neg_min < 0 else 1.0

print(f"pos_scale={pos_scale:.4f}, neg_scale={neg_scale:.4f}, "
      f"raw_pos_max={global_pos_max:.4f}, raw_neg_min={global_neg_min:.4f}")

# =======================
# 5. 画 1×5
# =======================
fig, axes = plt.subplots(1, TOPK, figsize=(15, 3), sharey=True)

for j, (motif, cnt) in enumerate(zip(top_motifs, top_counts)):
    ax = axes[j]
    score = scores[motif]

    score_pos = score.clip(lower=0) * pos_scale
    score_neg = score.clip(upper=0) * neg_scale

    # 先画阴影，再画彩色
    logomaker.Logo(score_neg, ax=ax, color_scheme=neg_color_scheme, vpad=.05, stack_order="big_on_top")
    logomaker.Logo(score_pos, ax=ax, color_scheme=pos_color_scheme, vpad=.05, stack_order="big_on_top")

    # 0线（分割线）+ 中心虚线
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.35)

    # 轴设置（模仿截图）

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

