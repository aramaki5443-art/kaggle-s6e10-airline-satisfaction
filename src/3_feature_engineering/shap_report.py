"""自分のLightGBMのSHAP値から、どの列がどう効いているかを表と図にする。

目的: 特徴量重要度(gain・分岐回数)とSHAPで、各列の寄与の大きさ・向き・列どうしの組み合わせ効果を確認する。
ひとことで言うと: 1人ずつの予測を「基準値+列ごとの押し上げ/押し下げ」に分解し、それを全員分集めて眺める。
やり方:
  - 入力は lgb_check.py --shap N が保存した data/shap_lgb_<tag>.npz(SHAP値・元の値)と、モデル・抜き出した行。
  - 表: 列ごとの平均|SHAP|(大きさ)・gain・分岐回数の順位と割合、SHAPと値の順位相関(+なら値が大きいほど満足側)。
  - 図(data/shap/<tag>/): ①平均|SHAP|の棒グラフ ②beeswarm(点1つが1人。横軸SHAP、色が値の大小)
    ③上位列と遅延の依存図(横軸=列の値、縦軸=SHAP、色=最も組み合わせ効果が強い相手の列)
    ④交互作用(2列の組み合わせ効果)の大きさの上位ペア(--inter行で計算。計算量が列数の2乗なので少なめの行で)。
  - SHAPの単位はlogit(対数オッズ)。+1は「満足のオッズを約2.7倍にする向きに押した」の意味。
なぜ: gainは「学習中にどれだけ損失を減らしたか」で、分岐回数は「何回使われたか」。どちらも向きが分からず、
  数値の種類が多い列(距離など)で大きく出やすい。SHAPは予測そのものを分解するので、向きと組み合わせ効果まで見られる。
使い方: python src/3_feature_engineering/shap_report.py --tag shap_tuned [--inter 3000]
"""
import argparse
import os

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="shap_tuned")
ap.add_argument("--inter", type=int, default=3000, help="交互作用を計算する行数(0なら計算しない)")
args = ap.parse_args()
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]  # 日本語の軸ラベル用(Windows)

d = np.load(f"data/shap_lgb_{args.tag}.npz", allow_pickle=True)
S, X, cols = d["shap"], d["X"], list(d["cols"])
out = f"data/shap/{args.tag}"
os.makedirs(out, exist_ok=True)
print(f"{len(S)}行・{len(cols)}列、fold0 va AUC {float(d['va_auc']):.5f}、木{int(d['iters'])}本")
# SHAPを全列足すと(基準値と合わせて)予測のlogitに一致するかの確認
m = lgb.Booster(model_file=f"data/shap_lgb_{args.tag}.txt")
Xs = pd.read_pickle(f"data/shap_lgb_{args.tag}_X.pkl")
raw = m.predict(Xs, raw_score=True)
print(f"検算: |基準値+SHAP合計−予測logit|の最大 {np.abs(d['base'] + S.sum(1) - raw).max():.2e}")

ma = np.abs(S).mean(0)
gain, split = d["gain"].astype(float), d["split"].astype(float)
def dir_corr(j):
    ok = ~np.isnan(X[:, j])  # 条件付きの列(ビジネスクラスだけ等)は対象外の行が欠損なので除く
    return spearmanr(X[ok, j], S[ok, j])[0] if ok.sum() > 10 and np.std(X[ok, j]) > 0 else np.nan
dirc = [dir_corr(j) for j in range(len(cols))]
t = pd.DataFrame({"列": cols, "平均|SHAP|": ma, "SHAP割合%": 100 * ma / ma.sum(), "gain割合%": 100 * gain / gain.sum(),
                  "分岐割合%": 100 * split / split.sum(), "値とSHAPの順位相関": dirc})
for c in ["平均|SHAP|", "gain割合%", "分岐割合%"]:
    t[c.replace("割合%", "") + "順位"] = t[c].rank(ascending=False).astype(int)
t = t.sort_values("平均|SHAP|", ascending=False).reset_index(drop=True)
t.to_csv(f"{out}/importance.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(t.round(4).to_string())

# ① 棒グラフ
o = np.argsort(ma)
fig, ax = plt.subplots(figsize=(8, 0.28 * len(cols) + 1))
ax.barh(np.array(cols)[o], ma[o])
ax.set_xlabel("平均|SHAP|(logit、大きいほど予測を大きく動かす列)")
ax.set_title("列ごとの寄与の大きさ(自分のLightGBM、fold0の検証行)")
fig.tight_layout(); fig.savefig(f"{out}/1_bar.png", dpi=110); plt.close(fig)
# ② beeswarm
shap.summary_plot(S, X, feature_names=cols, show=False, max_display=len(cols))
plt.xlabel("SHAP(右=満足の方向に押す、左=不満の方向に押す)")
plt.gcf().savefig(f"{out}/2_beeswarm.png", dpi=110, bbox_inches="tight"); plt.close("all")

# ④ 交互作用(2列の組み合わせ効果)
if args.inter:
    n = min(args.inter, len(Xs))
    Xn = Xs.iloc[:n].copy()
    for c in Xn.columns[Xn.dtypes == "category"]:
        Xn[c] = Xn[c].cat.codes.astype(float)  # LightGBMは内部でカテゴリ番号を使うので番号で渡せば同じ予測になる
    Xn = Xn.astype(float)
    print(f"番号で渡したときの予測の一致: 最大差 {np.abs(m.predict(Xn.values, raw_score=True) - raw[:n]).max():.2e}")
    I = shap.TreeExplainer(m).shap_interaction_values(Xn.values)
    I = I[1] if isinstance(I, list) else I
    A = np.abs(I).mean(0)
    np.fill_diagonal(A, 0)
    pairs = sorted(((A[i, j] * 2, cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))), reverse=True)
    pd.DataFrame(pairs[:30], columns=["組み合わせ効果の大きさ(平均|交互作用|×2)", "列A", "列B"]).to_csv(
        f"{out}/interactions.csv", index=False, encoding="utf-8-sig")
    print("\n組み合わせ効果の上位15ペア(平均|SHAP交互作用|×2、logit):")
    for v, a, b in pairs[:15]:
        print(f"  {v:.4f}  {a} × {b}")
    print(f"  (参考)主効果の合計に対する交互作用の合計の比率: {A.sum() / np.abs(np.einsum('nii->ni', I)).mean(0).sum():.3f}")

# ③ 依存図: 上位6列+到着遅延
top = list(t["列"][:6]) + [c for c in cols if "Delay" in c and c not in list(t["列"][:6])]
fig, axes = plt.subplots((len(top) + 2) // 3, 3, figsize=(15, 4.2 * ((len(top) + 2) // 3)))
for ax, c in zip(axes.ravel(), top):
    j = cols.index(c)
    k = shap.utils.approximate_interactions(j, S, X)[0]  # 色に使う相手(組み合わせ効果が強そうな列)
    sc = ax.scatter(X[:, j], S[:, j], c=X[:, k], s=3, cmap="coolwarm", alpha=0.6)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel(c); ax.set_ylabel("SHAP")
    ax.set_title(f"{c}(色={cols[k]})", fontsize=9)
    if "Delay" in c or c == "Flight Distance":
        ax.set_xscale("symlog")
    fig.colorbar(sc, ax=ax)
for ax in axes.ravel()[len(top):]:
    ax.axis("off")
fig.suptitle("依存図: 横軸=列の値、縦軸=SHAP(上=満足の方向)、色=組み合わせ効果が強い相手の列の値")
fig.tight_layout(); fig.savefig(f"{out}/3_dependence.png", dpi=100); plt.close(fig)
print(f"図と表を保存: {out}/")
