"""XGBoostをLightGBMと同じ特徴量で学習し、ブレンドに足す価値があるかを診断する。

目的: 調整後のLightGBM+RealMLPのブレンドに、3つ目のモデルとしてXGBoostを足すと伸びるかを、特徴量選びに時間をかける前に確かめる。
ひとことで言うと: 「同じ材料(36列)で別の料理人(XGBoost)に作らせたら、今の2人と違う味になるか」を1回だけ試す。
やり方:
  - 特徴量はfeatures.pyのADOPTED(36列)とTEをそのまま使う(定義・TEの計算はfeatures.pyと同一)。
  - ハイパラはXGBoostの既定(max_depth 6等)。ただし学習率だけ既定の0.3を0.05にし、木の本数は検証foldでの早期停止(100本)で
    決める(LightGBMの判定と同じ条件。0.3だと1本ごとの変化が大きく、判定がぶれやすいため)。
  - fold分割はseed42の5fold(--cv-seedで変更可)。OOFをdata/oof_xgb_<tag>.npyに保存し、ブレンドの診断は別に行う。
  - --predictで各foldのモデルでtestを予測して平均し、data/test_xgb_<tag>.npyに保存する(アンサンブル用、lgb_check.pyと同じやり方)。
なぜ: GBDT同士は予測が似やすく(前コンペでGBDT3種の相関0.998以上)、単体が良くてもブレンドが伸びないことがある。
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# features.pyの特徴量の定義(関数・ADOPTED・TE)を読み込む
src = open("src/2_lgb_baseline_fe/features.py", encoding="utf-8").read()
g = {"__name__": "fe_defs"}
exec(compile(src, "features.py", "exec"), g)

ap = argparse.ArgumentParser()
ap.add_argument("--cv-seed", type=int, default=42)
ap.add_argument("--tag", default="base")
ap.add_argument("--params", default=None, help="既定の一部を置き換える(JSON)")
ap.add_argument("--adopted", default=None, help="使うFEをカンマ区切りで置き換える(空文字なら生の20列だけ。特徴量の違いで多様性が増えるかの確認用)")
ap.add_argument("--predict", action="store_true", help="testも予測して保存する")
ap.add_argument("--fe", default=None, help="ADOPTEDに1つ足すFE(features.pyのFEの名前。XGBoost向けの特徴量選び)")
args = ap.parse_args()
if args.adopted is not None:
    g["ADOPTED"] = [a for a in args.adopted.split(",") if a]
if args.fe:
    g["ADOPTED"] = g["ADOPTED"] + [args.fe]

# データの置き場所・列名・foldの分け方(リポジトリのルートで実行する前提)
cfg = {"train_csv": "data/train.csv", "test_csv": "data/test.csv", "id_col": "id", "target_col": "satisfaction",
       "cv": {"n_splits": 5, "seed": 42}}
train = pd.read_csv(cfg["train_csv"])
y = train[cfg["target_col"]].astype(int).values
X = pd.concat([train[g["BASE"]]] + [g["FE"][a](train) for a in g["ADOPTED"]], axis=1)
for c in g["CAT"]:
    X[c] = X[c].astype("category")
print(f"{X.shape[1]}列", flush=True)
if args.predict:
    test = pd.read_csv(cfg["test_csv"])
    Xt = pd.concat([test[g["BASE"]]] + [g["FE"][a](test) for a in g["ADOPTED"]], axis=1)
    for c in g["CAT"]:
        Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)  # trainと同じカテゴリ番号にそろえる
    pred = np.zeros(len(Xt))

P = dict(learning_rate=0.05, n_estimators=10000, early_stopping_rounds=100, tree_method="hist", enable_categorical=True,
         eval_metric="auc", random_state=42, n_jobs=-1)
if args.params:
    P.update(json.loads(args.params))
oof = np.zeros(len(y)); tr_aucs, va_aucs, iters = [], [], []; t0 = time.time()
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=args.cv_seed).split(X, y)):
    Xtr, others = g["te_fold"](X.iloc[tr], y[tr], [X.iloc[va]] + ([Xt] if args.predict else []), seed=k)  # features.pyと同じTE
    Xva = others[0]
    m = xgb.XGBClassifier(**P)
    m.fit(Xtr, y[tr], eval_set=[(Xva, y[va])], verbose=False)
    oof[va] = m.predict_proba(Xva)[:, 1]
    if args.predict:
        pred += m.predict_proba(others[1])[:, 1] / 5
    tr_aucs.append(roc_auc_score(y[tr], m.predict_proba(Xtr)[:, 1])); va_aucs.append(roc_auc_score(y[va], oof[va]))
    iters.append(int(m.best_iteration))
    print(f"fold{k}: tr {tr_aucs[-1]:.5f} va {va_aucs[-1]:.5f} iter {iters[-1]} {time.time() - t0:.0f}s", flush=True)
tag = f"{args.tag}_cv{args.cv_seed}"
res = {"tag": tag, "params": P, "tr": float(np.mean(tr_aucs)), "oof": float(roc_auc_score(y, oof)),
       "va_folds": [float(v) for v in va_aucs], "iters": iters, "sec": round(time.time() - t0)}
print(json.dumps(res, ensure_ascii=False))
np.save(f"data/oof_xgb_{tag}.npy", oof)
if args.predict:
    np.save(f"data/test_xgb_{tag}.npy", pred)
with open("data/xgb_results.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
