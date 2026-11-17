"""評価1項目の「期待値の列」を前もって計算して保存する(公開ノートブックの補助タスク特徴量を、1項目ずつ試すための版)。

目的: 「この人の他の答え方からすると、この項目には何点を付けそうか」を列にして、lgb_check.pyで効果を測る。
ひとことで言うと: Wi-Fi以外の列からWi-Fiの点数を当てるモデルを作り、その予想点を特徴量にする。
    実際の点数と予想点のずれが、その人の「この項目への特別な不満・満足」を表す。
やり方:
  - train+test(約100万行)を5分割し、各行の予想は、その行を含まない4/5で学習したLightGBM(多クラス分類、0〜5点。2択の列は2値分類)で出す。
    満足かどうか(目的変数)は使わないので、testの行も学習に使ってよい。
  - 各点数の確率から期待値 Σ k×P(k) を計算し、data/aux_ev_<名前>.npy(trainの行→testの行の順)に保存する。
    features.pyのev_<名前>がこのファイルを読み込む。1項目あたりLightGBMを5回学習して重いので、判定のたびには計算しない。
なぜ: 自分の点数を見て学習したモデルの予想は、実際の点数を覚えてしまい予想にならない(=分割して自分を含まないモデルで出す)。
    期待値は20列の組み合わせから作るので、木が少ない分岐では作れない情報になりうる。
    投稿者の報告では、期待値の列は効き、「その点数の起こりにくさ」の列は効かなかった。
  - カテゴリ列(Class・Type of Travel・Customer Type)は順序のある整数に置き換えて同じ計算をする
    (Class: Eco=0・Eco Plus=1・Business=2 / 2値の列はP(1)=その値である確率になる)。
実行: python src/3_lgb_baseline_fe/make_expected_value.py --col "Online boarding" --name boarding
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ap = argparse.ArgumentParser()
ap.add_argument("--col", required=True)
ap.add_argument("--name", required=True)
args = ap.parse_args()

FEATS = [  # features.pyのBASE(元の特徴量20個)とGender。予想する列自身は除く
    "Online boarding", "Class", "Inflight entertainment", "Inflight wifi service", "Seat comfort",
    "Flight Distance", "Type of Travel", "On-board service", "Cleanliness", "Leg room service", "Age",
    "Ease of Online booking", "Baggage handling", "Checkin service", "Food and drink", "Gate location",
    "Customer Type", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
    "Departure Delay in Minutes", "Gender",
]
both = pd.concat([pd.read_csv(f) for f in ("data/train.csv", "data/test.csv")], ignore_index=True)
X = both[[c for c in FEATS if c != args.col]].copy()
for c in X.select_dtypes(exclude="number").columns:  # 文字列のカテゴリ列(Class等)
    X[c] = X[c].astype("category")
CAT_ORDER = {  # カテゴリ列を予想するときの数値化(大きいほど「上の席・仕事・常連」)
    "Class": {"Eco": 0, "Eco Plus": 1, "Business": 2},
    "Type of Travel": {"Personal Travel": 0, "Business travel": 1},
    "Customer Type": {"disloyal Customer": 0, "Loyal Customer": 1},
}
y = (both[args.col].map(CAT_ORDER[args.col]) if args.col in CAT_ORDER else both[args.col]).astype(int).values
classes = np.unique(y)
ev = np.zeros(len(X))
t0 = time.time()
for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
    m = lgb.LGBMClassifier(objective="multiclass" if len(classes) > 2 else "binary", learning_rate=0.1, n_estimators=2000, random_state=0, verbose=-1)
    # 早期停止用に学習側の一部を取り分ける(予想を出す行va自身は使わない)
    rng = np.random.default_rng(k)
    es = rng.random(len(tr)) < 0.1
    m.fit(X.iloc[tr[~es]], y[tr[~es]], eval_X=(X.iloc[tr[es]],), eval_y=(y[tr[es]],),
          callbacks=[lgb.early_stopping(50, verbose=False)])
    ev[va] = m.predict_proba(X.iloc[va]) @ m.classes_
    print(f"fold{k}: iter {m.best_iteration_} 実際との相関 {np.corrcoef(ev[va], y[va])[0, 1]:.3f} ({time.time() - t0:.0f}秒)", flush=True)
np.save(f"data/aux_ev_{args.name}.npy", ev)
print(f"期待値 平均{ev.mean():.3f} 実際の平均{y.mean():.3f} 相関{np.corrcoef(ev, y)[0, 1]:.3f} R2 {1 - ((y - ev) ** 2).mean() / y.var():.3f}")
