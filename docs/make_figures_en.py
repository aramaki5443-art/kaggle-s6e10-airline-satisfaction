"""README.en.md用に、EDAとSHAPの図を英語のラベルで描き直す(中身はnotebooks/EDA.ipynb・experiments/shap_report.pyの図と同じ)。

使い方: リポジトリのルートで python docs/make_figures_en.py
入力: data/train.csv、data/shap_lgb_shap_tuned.npz(`python src/lgb_fe_check.py --fe none --shap N --tag shap_tuned`で作る)
出力: docs/img/*_en.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

OUT = "docs/img"
train = pd.read_csv("data/train.csv")
TARGET = "satisfaction"
NUM = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
CAT = ["Gender", "Customer Type", "Type of Travel", "Class"]
SVC = [c for c in train.columns if c not in NUM + CAT + ["id", TARGET]]
y = train[TARGET].astype(int)

# 図1: Class × Type of Travel(EDA.ipynb 5節と同じ集計)
fig = plt.figure(figsize=(13, 9.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
ax_n, ax_r, ax_d = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
share = pd.crosstab(train["Class"], train["Type of Travel"], normalize=True) * 100
rate = pd.crosstab(train["Class"], train["Type of Travel"], values=train[TARGET], aggfunc="mean")
sns.heatmap(share, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "Share of all passengers (%)"}, ax=ax_n)
ax_n.set_title("Share of passengers (%, train total = 100)")
sns.heatmap(rate, annot=True, fmt=".1%", cmap="RdBu_r", center=train[TARGET].mean(), vmin=0, vmax=1,
            cbar_kws={"label": "Satisfaction rate"}, ax=ax_r)
ax_r.set_title("Satisfaction rate (share of satisfaction = True)")
bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 5000]
d = train.assign(dist=pd.cut(train["Flight Distance"], bins), seg=train["Class"] + " × " + train["Type of Travel"])
line = d.groupby(["seg", "dist"], observed=True)[TARGET].agg(["mean", "size"]).reset_index()
line = line[line["size"] >= 200]
for seg, g in line.groupby("seg"):
    ax_d.plot(g["dist"].astype(str), g["mean"], marker="o", label=seg)
ax_d.set_xlabel("Flight Distance bin"); ax_d.set_ylabel("Satisfaction rate"); ax_d.set_ylim(0, 1)
ax_d.set_title("Satisfaction rate by flight-distance bin for each of the 6 segments (points with < 200 passengers omitted)")
ax_d.legend(fontsize=9, ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/class_x_travel_en.png", dpi=100); plt.close(fig)

# 図2: 旅行目的ごとの評価別の満足率(EDA.ipynb 8節のヒートマップから4項目)
items = ["Seat comfort", "Inflight entertainment", "On-board service", "Leg room service"]
fig, axes = plt.subplots(2, 2, figsize=(16, 4.4))
for ax, c in zip(axes.flat, items):
    r = y.groupby([train["Type of Travel"], train[c]]).mean().unstack() * 100
    n = y.groupby([train["Type of Travel"], train[c]]).size().unstack()
    sns.heatmap(r.where(n >= 100), ax=ax, cmap="RdYlGn", vmin=0, vmax=100, annot=True, fmt=".0f", cbar=False, annot_kws={"size": 9})
    ax.set_title(c, fontsize=10); ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=0); ax.tick_params(axis="y", labelsize=8, rotation=0)
fig.suptitle("Satisfaction rate (%) by rating. Rows = type of travel, columns = rating (0–5), blank = fewer than 100 passengers", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/gate_type_of_travel_en.png", dpi=100); plt.close(fig)

# 図3: 評価0点・1点・5点の人の満足率(EDA.ipynb 7節と同じ集計)
rows = []
for c in [c for c in SVC if (train[c] == 0).any()]:
    other_avg = train[[o for o in SVC if o != c]].replace(0, np.nan).mean(axis=1)
    z = train[c] == 0
    rows.append({"item": c, "sat0": y[z].mean(), "sat1": y[train[c] == 1].mean(), "sat5": y[train[c] == 5].mean(),
                 "other0": other_avg[z].mean(), "other1": other_avg[train[c] == 1].mean()})
zero = pd.DataFrame(rows).set_index("item").sort_values("sat0")
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
yy = np.arange(len(zero))
for off, col, lab, color in [(-0.27, "sat1", "Rated 1", "#4C72B0"), (0, "sat0", "Rated 0", "#C44E52"), (0.27, "sat5", "Rated 5", "#55A868")]:
    ax0.barh(yy + off, zero[col], height=0.27, color=color, label=lab)
ax0.set_yticks(yy, zero.index); ax0.set_xlim(0, 1); ax0.set_xlabel("Satisfaction rate (share of satisfaction = True)")
ax0.set_title("Satisfaction of passengers who rated the item 0, 1 or 5"); ax0.legend(loc="lower right")
ax1.barh(yy - 0.2, zero["other1"], height=0.4, color="#4C72B0", label="Rated 1")
ax1.barh(yy + 0.2, zero["other0"], height=0.4, color="#C44E52", label="Rated 0")
ax1.set_yticks(yy, []); ax1.set_xlim(2, 3.6); ax1.set_xlabel("Average rating given to the other items (0 excluded)")
ax1.set_title("How they rated the other items"); ax1.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/rating_zero_en.png", dpi=100); plt.close(fig)

# 図4: 単変量AUC(EDA.ipynb 6節と同じ計算)
rows = []
for c in NUM + SVC + CAT:
    x = train[c]
    raw = roc_auc_score(y, x.fillna(-1)) if pd.api.types.is_numeric_dtype(x) else np.nan
    key = (pd.qcut(x, 50, duplicates="drop").astype(str) if x.nunique() > 50 else x.astype(str)).fillna("NA")
    rows.append((c, raw, roc_auc_score(y, y.groupby(key).transform("mean"))))
auc = pd.DataFrame(rows, columns=["feature", "raw", "rate"]).set_index("feature").sort_values("rate")
fig, ax = plt.subplots(figsize=(11, 8))
yy = np.arange(len(auc))
ax.barh(yy + 0.2, auc["rate"], height=0.4, color="#DD8452", label="Satisfaction rate per value as the score")
ax.barh(yy - 0.2, auc["raw"], height=0.4, color="#4C72B0", label="Raw value as the score (numeric only)")
ax.axvline(0.5, color="k", ls="--", lw=1)
ax.set_yticks(yy, auc.index); ax.set_xlim(0.4, 0.9)
ax.set_xlabel("Univariate AUC (0.5 = no separation, 1.0 = perfect separation)")
ax.set_title("Univariate AUC of each feature alone (train)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/univariate_auc_en.png", dpi=100); plt.close(fig)

# SHAP: 列ごとの寄与の大きさ(experiments/shap_report.pyの①と同じ)
s = np.load("data/shap_lgb_shap_tuned.npz", allow_pickle=True)
S, cols = s["shap"], list(s["cols"])
ma = np.abs(S).mean(0); o = np.argsort(ma)
fig, ax = plt.subplots(figsize=(8, 0.28 * len(cols) + 1))
ax.barh(np.array(cols)[o], ma[o])
ax.set_xlabel("Mean |SHAP| (logit; larger = moves predictions more)")
ax.set_title("Contribution of each feature (my LightGBM, fold-0 validation rows)")
fig.tight_layout(); fig.savefig(f"{OUT}/shap_lightgbm_en.png", dpi=110); plt.close(fig)
print("saved")
