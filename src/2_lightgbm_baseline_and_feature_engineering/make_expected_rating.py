"""サービス評価1項目の期待値(特徴量)を前もって計算して保存する(公開ノートブックの補助タスク特徴量を、1項目ずつ試すための版)。

目的: 「この人の他の答え方からすると、この項目には何点を付けそうか」を列にして、train_lightgbm.pyで効果を測る。
ひとことで言うと: Wi-Fi以外の列からWi-Fiの点数を当てるモデルを作り、その予想点を特徴量にする。
    期待値は「ほかの項目への答え方から見た、この人の評価の高さ」を表す。同じ4点でも、期待値が高い人ほど満足率が高い
    (オンライン搭乗が4点の人で、期待値が低い人は27%、高い人は94%。feature_engineering.ipynbの10)。
やり方:
  - train+test(約100万行)を5分割し、各行の予想は、その行を含まない4/5で学習したLightGBM(多クラス分類、0〜5点。2択の列は2値分類)で出す。
    満足かどうか(目的変数)は使わないので、testの行も学習に使ってよい。
  - 各点数の確率から期待値 Σ k×P(k) を計算し、data/aux_ev_<名前>.npy(trainの行→testの行の順)に保存する。
    feature_definitions.pyのev_<名前>がこのファイルを読み込む。1項目あたりLightGBMを5回学習して重いので、判定のたびには計算しない。
なぜ: 自分の点数を見て学習したモデルの予想は、実際の点数を覚えてしまい予想にならない(=分割して自分を含まないモデルで出す)。
    期待値は20列の組み合わせから作るので、木が少ない分岐では作れない情報になりうる。
    投稿者の報告では、期待値の列は効き、「その点数の起こりにくさ」の列は効かなかった。
  - カテゴリ列(Class・Type of Travel・Customer Type)は順序のある整数に置き換えて同じ計算をする
    (Class: Eco=0・Eco Plus=1・Business=2 / 2値の列はP(1)=その値である確率になる)。
実行: python src/2_lightgbm_baseline_and_feature_engineering/make_expected_rating.py --rating-col "Online boarding" --save-name boarding
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--rating-col", required=True, help="期待値を作るサービス評価の列名(例: Online boarding)")
ap.add_argument("--save-name", required=True, help="保存名。data/aux_ev_<名前>.npyに保存する(例: boarding)")
args = ap.parse_args()

INPUT_FEATURES = [  # feature_definitions.pyのBASE_FEATURES(元の特徴量20個)とGender。予想する列自身は除く
    "Online boarding", "Class", "Inflight entertainment", "Inflight wifi service", "Seat comfort",
    "Flight Distance", "Type of Travel", "On-board service", "Cleanliness", "Leg room service", "Age",
    "Ease of Online booking", "Baggage handling", "Checkin service", "Food and drink", "Gate location",
    "Customer Type", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
    "Departure Delay in Minutes", "Gender",
]
train_and_test = pd.concat([pd.read_csv(f) for f in ("data/train.csv", "data/test.csv")], ignore_index=True)
X = train_and_test[[c for c in INPUT_FEATURES if c != args.rating_col]].copy()
for c in X.select_dtypes(exclude="number").columns:  # 文字列のカテゴリ列(Class等)
    X[c] = X[c].astype("category")
CATEGORY_ORDER = {  # カテゴリ列を予想するときの数値化(大きいほど「上の席・仕事・常連」)
    "Class": {"Eco": 0, "Eco Plus": 1, "Business": 2},
    "Type of Travel": {"Personal Travel": 0, "Business travel": 1},
    "Customer Type": {"disloyal Customer": 0, "Loyal Customer": 1},
}
y = (train_and_test[args.rating_col].map(CATEGORY_ORDER[args.rating_col]) if args.rating_col in CATEGORY_ORDER else train_and_test[args.rating_col]).astype(int).values
classes = np.unique(y)
expected = np.zeros(len(X))
start_time = time.time()
for k, (train_rows, valid_rows) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
    model = lgb.LGBMClassifier(objective="multiclass" if len(classes) > 2 else "binary", learning_rate=0.1, n_estimators=2000, random_state=0, verbose=-1)
    # 早期停止用に学習側の一部を取り分ける(予想を出す行valid_rows自身は使わない)
    rng = np.random.default_rng(k)
    stop_rows = rng.random(len(train_rows)) < 0.1
    model.fit(X.iloc[train_rows[~stop_rows]], y[train_rows[~stop_rows]], eval_X=(X.iloc[train_rows[stop_rows]],), eval_y=(y[train_rows[stop_rows]],),
              callbacks=[lgb.early_stopping(50, verbose=False)])
    expected[valid_rows] = model.predict_proba(X.iloc[valid_rows]) @ model.classes_
    print(f"fold{k}: iter {model.best_iteration_} 実際との相関 {np.corrcoef(expected[valid_rows], y[valid_rows])[0, 1]:.3f} ({time.time() - start_time:.0f}秒)", flush=True)
np.save(f"data/aux_ev_{args.save_name}.npy", expected)
print(f"期待値 平均{expected.mean():.3f} 実際の平均{y.mean():.3f} 相関{np.corrcoef(expected, y)[0, 1]:.3f} R2 {1 - ((y - expected) ** 2).mean() / y.var():.3f}")
