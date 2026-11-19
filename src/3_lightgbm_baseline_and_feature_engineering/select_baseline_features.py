"""LightGBMのベースライン: 単独で強い特徴量から1つずつ足していく(前向き追加)。

目的: どの特徴量を足すとどれだけOOF AUCが伸びるかを1つずつ測り、ベースラインの土台を作る。
ひとことで言うと: 「単独で一番当たる特徴量」から順にモデルへ1つずつ渡し、伸びが止まる所を探す。
やり方:
  - 特徴量の順番は、EDA.ipynb 6節の単変量AUC(値ごとの満足率をスコアにしたもの)の高い順で固定(ORDER)。
  - `--n-features N` で上位N個を使い、cfgのfold設定(5fold、StratifiedKFold seed42)でLightGBMを学習してOOF AUCを出す。
    OOF(Out-Of-Fold)= 各行を「その行を学習に使っていないモデル」で予測した値。全行分そろえて評価する。
  - ハイパラは既定値に近いまま(learning_rate 0.05 + early stopping 100のみ設定)。
    feature_fraction(列の間引き)は既定の1.0なので、列数が変わっても木の抽選パターンが変わる副作用は無い。
なぜ: 一度に全部入れると、どの特徴量が効いたのか分からない。1つずつ足せば各特徴量の上乗せ分が分かる。
実行: python src/3_lightgbm_baseline_and_feature_engineering/select_baseline_features.py --n-features 1
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

FEATURE_ORDER = [  # EDA 6節の「値ごとの満足率」単変量AUCの高い順
    "Online boarding", "Class", "Inflight entertainment", "Inflight wifi service", "Seat comfort",
    "Flight Distance", "Type of Travel", "On-board service", "Cleanliness", "Leg room service", "Age",
    "Ease of Online booking", "Baggage handling", "Checkin service", "Food and drink", "Gate location",
    "Customer Type", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
    "Departure Delay in Minutes", "Gender",
]
CATEGORICAL_COLUMNS = ["Gender", "Customer Type", "Type of Travel", "Class"]
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, n_estimators=10000, random_state=42, verbose=-1)

ap = argparse.ArgumentParser()
ap.add_argument("--n-features", type=int, required=True, help="FEATURE_ORDERの上位何個を使うか")
ap.add_argument("--predict", action="store_true", help="testを予測してdata/submission.csvを作る")
args = ap.parse_args()

# データの置き場所・列名・foldの分け方(リポジトリのルートで実行する前提)
config = {"train_csv": "data/train.csv", "test_csv": "data/test.csv", "id_col": "id", "target_col": "satisfaction",
          "cv": {"n_splits": 5, "seed": 42}}
train = pd.read_csv(config["train_csv"])
y = train[config["target_col"]].astype(int).values
used_features = FEATURE_ORDER[: args.n_features]
X = train[used_features].copy()
test = pd.read_csv(config["test_csv"])
X_test = test[used_features].copy()
for c in CATEGORICAL_COLUMNS:
    if c in X:
        X[c] = X[c].astype("category")  # LightGBMのカテゴリ分割をそのまま使う
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)  # trainと同じカテゴリ番号にそろえる
test_pred = np.zeros(len(X_test))

oof = np.zeros(len(X))
train_aucs, valid_aucs, best_iterations = [], [], []
folds = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=config["cv"]["seed"])
start_time = time.time()
for k, (train_rows, valid_rows) in enumerate(folds.split(X, y)):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X.iloc[train_rows], y[train_rows], eval_X=(X.iloc[valid_rows],), eval_y=(y[valid_rows],), eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[valid_rows] = model.predict_proba(X.iloc[valid_rows])[:, 1]
    train_aucs.append(roc_auc_score(y[train_rows], model.predict_proba(X.iloc[train_rows])[:, 1]))
    valid_aucs.append(roc_auc_score(y[valid_rows], oof[valid_rows]))
    best_iterations.append(model.best_iteration_)
    if args.predict:
        test_pred += model.predict_proba(X_test)[:, 1] / folds.get_n_splits()
    print(f"fold{k}: tr {train_aucs[-1]:.5f} va {valid_aucs[-1]:.5f} iter {best_iterations[-1]}")

print(f"features({len(used_features)}): {used_features}")
print(f"tr AUC(fold平均) {np.mean(train_aucs):.5f} | OOF AUC {roc_auc_score(y, oof):.5f} "
      f"| fold va 平均 {np.mean(valid_aucs):.5f} ± {np.std(valid_aucs):.5f} | {time.time() - start_time:.0f}s")
np.save(f"data/oof_lgb_forward_n{args.n_features:02d}.npy", oof)  # data/は.gitignore対象
if args.predict:
    pd.DataFrame({config["id_col"]: test[config["id_col"]], config["target_col"]: test_pred}).to_csv("data/submission.csv", index=False)
    np.save(f"data/test_lgb_forward_n{args.n_features:02d}.npy", test_pred)
with open("data/lgb_forward_results.jsonl", "a", encoding="utf-8") as f:  # まとめ実行の結果を1行ずつ残す
    f.write(json.dumps({"n": args.n_features, "added": used_features[-1], "n_splits": config["cv"]["n_splits"],
                        "tr": float(np.mean(train_aucs)), "oof": float(roc_auc_score(y, oof)),
                        "va_folds": [float(v) for v in valid_aucs], "iters": [int(i) for i in best_iterations],
                        "sec": round(time.time() - start_time)}, ensure_ascii=False) + "\n")
