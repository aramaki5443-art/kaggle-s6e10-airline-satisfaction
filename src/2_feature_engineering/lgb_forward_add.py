"""LightGBMのベースライン: 単独で強い特徴量から1つずつ足していく(前向き追加)。

目的: どの特徴量を足すとどれだけOOF AUCが伸びるかを1つずつ測り、ベースラインの土台を作る。
ひとことで言うと: 「単独で一番当たる特徴量」から順にモデルへ1つずつ渡し、伸びが止まる所を探す。
やり方:
  - 特徴量の順番は、EDA.ipynb 6節の単変量AUC(値ごとの満足率をスコアにしたもの)の高い順で固定(ORDER)。
  - `--n N` で上位N個を使い、cfgのfold設定(5fold、StratifiedKFold seed42)でLightGBMを学習してOOF AUCを出す。
    OOF(Out-Of-Fold)= 各行を「その行を学習に使っていないモデル」で予測した値。全行分そろえて評価する。
  - ハイパラは既定値に近いまま(learning_rate 0.05 + early stopping 100のみ設定)。
    feature_fraction(列の間引き)は既定の1.0なので、列数が変わっても木の抽選パターンが変わる副作用は無い。
なぜ: 一度に全部入れると、どの特徴量が効いたのか分からない。1つずつ足せば各特徴量の上乗せ分が分かる。
実行: python src/2_feature_engineering/lgb_forward_add.py --n 1
      --predict を付けると、各foldのモデルでtestを予測して平均し data/submission.csv を作る(提出用)。
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ORDER = [  # EDA 6節の「値ごとの満足率」単変量AUCの高い順
    "Online boarding", "Class", "Inflight entertainment", "Inflight wifi service", "Seat comfort",
    "Flight Distance", "Type of Travel", "On-board service", "Cleanliness", "Leg room service", "Age",
    "Ease of Online booking", "Baggage handling", "Checkin service", "Food and drink", "Gate location",
    "Customer Type", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
    "Departure Delay in Minutes", "Gender",
]
CAT = ["Gender", "Customer Type", "Type of Travel", "Class"]
PARAMS = dict(objective="binary", learning_rate=0.05, n_estimators=10000, random_state=42, verbose=-1)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True, help="ORDERの上位何個を使うか")
ap.add_argument("--predict", action="store_true", help="testを予測してdata/submission.csvを作る")
args = ap.parse_args()

# データの置き場所・列名・foldの分け方(リポジトリのルートで実行する前提)
cfg = {"train_csv": "data/train.csv", "test_csv": "data/test.csv", "id_col": "id", "target_col": "satisfaction",
       "cv": {"n_splits": 5, "seed": 42}}
train = pd.read_csv(cfg["train_csv"])
y = train[cfg["target_col"]].astype(int).values
feats = ORDER[: args.n]
X = train[feats].copy()
test = pd.read_csv(cfg["test_csv"])
Xt = test[feats].copy()
for c in CAT:
    if c in X:
        X[c] = X[c].astype("category")  # LightGBMのカテゴリ分割をそのまま使う
        Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)  # trainと同じカテゴリ番号にそろえる
pred = np.zeros(len(Xt))

oof = np.zeros(len(X))
tr_aucs, va_aucs, iters = [], [], []
skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["cv"]["seed"])
t0 = time.time()
for k, (tr, va) in enumerate(skf.split(X, y)):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X.iloc[tr], y[tr], eval_X=(X.iloc[va],), eval_y=(y[va],), eval_metric="auc",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    tr_aucs.append(roc_auc_score(y[tr], m.predict_proba(X.iloc[tr])[:, 1]))
    va_aucs.append(roc_auc_score(y[va], oof[va]))
    iters.append(m.best_iteration_)
    if args.predict:
        pred += m.predict_proba(Xt)[:, 1] / skf.get_n_splits()
    print(f"fold{k}: tr {tr_aucs[-1]:.5f} va {va_aucs[-1]:.5f} iter {iters[-1]}")

print(f"features({len(feats)}): {feats}")
print(f"tr AUC(fold平均) {np.mean(tr_aucs):.5f} | OOF AUC {roc_auc_score(y, oof):.5f} "
      f"| fold va 平均 {np.mean(va_aucs):.5f} ± {np.std(va_aucs):.5f} | {time.time() - t0:.0f}s")
np.save(f"data/oof_lgb_forward_n{args.n:02d}.npy", oof)  # data/は.gitignore対象
if args.predict:
    pd.DataFrame({cfg["id_col"]: test[cfg["id_col"]], cfg["target_col"]: pred}).to_csv("data/submission.csv", index=False)
    np.save(f"data/test_lgb_forward_n{args.n:02d}.npy", pred)
with open("data/lgb_forward_results.jsonl", "a", encoding="utf-8") as f:  # まとめ実行の結果を1行ずつ残す
    f.write(json.dumps({"n": args.n, "added": feats[-1], "n_splits": cfg["cv"]["n_splits"],
                        "tr": float(np.mean(tr_aucs)), "oof": float(roc_auc_score(y, oof)),
                        "va_folds": [float(v) for v in va_aucs], "iters": [int(i) for i in iters],
                        "sec": round(time.time() - t0)}, ensure_ascii=False) + "\n")
