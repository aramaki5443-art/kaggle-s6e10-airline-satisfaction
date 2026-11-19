"""公開OOFの点検: 1本ずつ単体AUC・欠損・値の範囲・自分のモデルとの相関を調べ、スタッキングに入れてよいかを判定する。

目的: 他の参加者が公開したOOF(1段目の予測)を2段目に入れる前に、壊れたもの・答えを見たものを外す。
ひとことで言うと: 他の参加者の予測を使う前に、全行そろっているか・単体のAUCが不自然に高すぎないかを1本ずつ確かめる。
やり方:
  - 入力はフォルダ内のoof_<名前>.npyとtest_<名前>.npy(行の順番はtrain.csv・test.csvと同じ前提)。
  - 行数・欠損・0〜1の範囲・単体AUC・自分のRealMLPとの順位相関・testの平均を表にしてCSVに保存する。
  - 単体AUCが自分の5モデルのスタック(0.9617)より高いものは、それ自体が答えを使ったスタックの可能性が高いので「要確認」にする。
なぜ: 1本でも答えを見たOOFが混ざると、2段目はそれに頼り切ってOOFだけ高くなり、LBで崩れる。
使い方: python src/8_public_oof_inspection/inspect_public_oof.py data/pub_oof/cdeotte124
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
folder = sys.argv[1]
y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).values
n_test = len(pd.read_csv("data/test.csv", usecols=["id"]))
own_realmlp = np.load("data/oof_realmlp_e4_avg2.npy")
sample_rows = np.random.default_rng(0).choice(len(y), 100000, replace=False)  # 相関は10万行で計算(速さのため)
rows = []
for oof_file in sorted(glob.glob(f"{folder}/oof_*.npy")):
    name = os.path.basename(oof_file)[4:-4]
    oof = np.load(oof_file).astype(float).ravel()
    test_file = f"{folder}/test_{name}.npy"
    test = np.load(test_file).astype(float).ravel() if os.path.exists(test_file) else None
    row = {"name": name, "n": len(oof), "nan": int(np.isnan(oof).sum()), "min": oof.min(), "max": oof.max(),
           "test_n": None if test is None else len(test), "test_mean": None if test is None else float(np.nanmean(test))}
    if len(oof) == len(y) and row["nan"] == 0:
        row["auc"] = roc_auc_score(y, oof); row["corr_rmlp"] = spearmanr(oof[sample_rows], own_realmlp[sample_rows])[0]
    rows.append(row)
result = pd.DataFrame(rows)
shape_ok = (result["n"] == len(y)) & (result["nan"] == 0) & (result["test_n"] == n_test)
result["判定"] = np.where(~shape_ok, "形が合わない", np.where(result["auc"] > 0.9617, "要確認(高すぎ)", "OK"))
result.to_csv(f"{folder}/_inspect.csv", index=False)
print(result["判定"].value_counts().to_string())
print(result.sort_values("auc", ascending=False).head(12)[["name", "auc", "corr_rmlp", "min", "max", "test_mean", "判定"]].to_string(index=False))
print(result[result["判定"] != "OK"][["name", "n", "nan", "test_n", "auc", "判定"]].to_string(index=False))
print("AUCの分布", result["auc"].describe().round(6).to_dict())
