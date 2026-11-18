"""公開OOFの点検: 1本ずつ単体AUC・欠損・値の範囲・自分のモデルとの相関を調べ、スタッキングに入れてよいかを判定する。

目的: 他の参加者が公開したOOF(1段目の予測)を2段目に入れる前に、壊れたもの・答えを見たものを外す。
ひとことで言うと: 「よその料理人の味見の記録」を使う前に、記録が全行そろっているか・不自然に良すぎないかを1本ずつ確かめる。
やり方:
  - 入力はフォルダ内のoof_<名前>.npyとtest_<名前>.npy(行の順番はtrain.csv・test.csvと同じ前提)。
  - 行数・欠損・0〜1の範囲・単体AUC・自分のRealMLPとの順位相関・testの平均を表にしてCSVに保存する。
  - 単体AUCが自分の5モデルのスタック(0.9617)より高いものは、それ自体が答えを使ったスタックの可能性が高いので「要確認」にする。
なぜ: 1本でも答えを見たOOFが混ざると、2段目はそれに頼り切ってOOFだけ高くなり、LBで崩れる。
使い方: python src/8_public_oof_check/pub_oof_inspect.py data/pub_oof/cdeotte124
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
d = sys.argv[1]
y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).values
n_test = len(pd.read_csv("data/test.csv", usecols=["id"]))
ref = np.load("data/oof_realmlp_e4_avg2.npy")
idx = np.random.default_rng(0).choice(len(y), 100000, replace=False)  # 相関は10万行で計算(速さのため)
rows = []
for f in sorted(glob.glob(f"{d}/oof_*.npy")):
    name = os.path.basename(f)[4:-4]
    o = np.load(f).astype(float).ravel()
    tf = f"{d}/test_{name}.npy"
    t = np.load(tf).astype(float).ravel() if os.path.exists(tf) else None
    r = {"name": name, "n": len(o), "nan": int(np.isnan(o).sum()), "min": o.min(), "max": o.max(),
         "test_n": None if t is None else len(t), "test_mean": None if t is None else float(np.nanmean(t))}
    if len(o) == len(y) and r["nan"] == 0:
        r["auc"] = roc_auc_score(y, o); r["corr_rmlp"] = spearmanr(o[idx], ref[idx])[0]
    rows.append(r)
df = pd.DataFrame(rows)
ok = (df["n"] == len(y)) & (df["nan"] == 0) & (df["test_n"] == n_test)
df["判定"] = np.where(~ok, "形が合わない", np.where(df["auc"] > 0.9617, "要確認(高すぎ)", "OK"))
df.to_csv(f"{d}/_inspect.csv", index=False)
print(df["判定"].value_counts().to_string())
print(df.sort_values("auc", ascending=False).head(12)[["name", "auc", "corr_rmlp", "min", "max", "test_mean", "判定"]].to_string(index=False))
print(df[df["判定"] != "OK"][["name", "n", "nan", "test_n", "auc", "判定"]].to_string(index=False))
print("AUCの分布", df["auc"].describe().round(6).to_dict())
