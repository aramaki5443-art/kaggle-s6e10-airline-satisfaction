"""アンサンブルの物差し: 2段目のロジスティック回帰(スタッキング)と順位ブレンドを、同じ条件で公平に比べる。

目的: モデルを足したときに、アンサンブルが本当に伸びたかを測る。OOFの上で重みを探すブレンドは、役に立たないモデルにも
  重み0を選べるため、モデルを足してもOOFは下がらない仕組みになっている(実際に、OOF+0.000046でLBは-0.00002)。
ひとことで言うと: 「混ぜ方を決める行」と「採点する行」を分け、混ぜ方の当てはめすぎを取り除いた点数で比べる。
やり方:
  - 入力は各モデルのOOF(1段目、答えを見ていない予測)。OOFを1段目とは別のfold分割(seed7の5fold)で分け、
    4つで混ぜ方を学び、残り1つで採点する(上位の公開ノートブックと同じ。調査は2026-10-17)。
  - stack: ロジスティック回帰(L2、C=1.0)。入力は各モデルの確率をlogit=log(p/(1-p))に直したもの。
  - rank: 順位の加重平均。重みは4つのfoldで格子探索して決め、残り1foldに当てる(重みもfoldの外で決める)。
  - 各方式の2段目のOOFでAUCを出す(foldごとの値の平均ではなく、全行をつないだAUC)。
なぜ: 上位ノートブックは全てこの形(logitのLR)だった。客の種類などの列を2段目に足す例は悪化の報告のみのため入れない。
使い方: python src/7_stacking_ensemble/evaluate_stacking.py 名前=OOFのパス 名前=OOFのパス ...
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("members", nargs="+", help="混ぜるモデルを「名前=OOFのファイル」の形で並べる(例: XGBoost=data/oof_xgb_t35_cv42.npy)")
ap.add_argument("--meta-seed", type=int, default=7, help="2段目の評価の5分割のseed")
ap.add_argument("--weight-step", type=float, default=0.05, help="順位ブレンドの重みの格子の幅")
args = ap.parse_args()

y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).values
names = [m.split("=")[0] for m in args.members]
predictions = np.column_stack([np.load(m.split("=")[1]) for m in args.members])
eps = 1e-6
logits = np.log(np.clip(predictions, eps, 1 - eps) / (1 - np.clip(predictions, eps, 1 - eps)))
ranks = np.column_stack([rankdata(predictions[:, j]) / len(y) for j in range(predictions.shape[1])])
for j, name in enumerate(names):
    print(f"単体 {name}: {roc_auc_score(y, predictions[:, j]):.6f}")

# 順位ブレンドの重みの候補(合計1になる組み合わせを --weight-step 刻みで全部)
weight_grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, args.weight_step), repeat=len(names) - 1) if sum(w) <= 1 + 1e-9]
weight_grid = [np.array(list(w) + [1 - sum(w)]) for w in weight_grid]
stack_oof, rank_oof = np.zeros(len(y)), np.zeros(len(y))
coefs, chosen_weights = [], []
for train_rows, valid_rows in StratifiedKFold(5, shuffle=True, random_state=args.meta_seed).split(logits, y):
    stacker = LogisticRegression(C=1.0, max_iter=1000).fit(logits[train_rows], y[train_rows])
    stack_oof[valid_rows] = stacker.decision_function(logits[valid_rows]); coefs.append(stacker.coef_[0])
    # 順位ブレンドの重みも学習側の行だけで決める(採点する行の答えを見ない)
    scores = [roc_auc_score(y[train_rows], ranks[train_rows] @ w) for w in weight_grid]
    best_weight = weight_grid[int(np.argmax(scores))]; chosen_weights.append(best_weight)
    rank_oof[valid_rows] = ranks[valid_rows] @ best_weight
print(f"stack(LR logit) OOF {roc_auc_score(y, stack_oof):.6f} 係数平均 {np.round(np.mean(coefs, 0), 3).tolist()}")
print(f"rank(重みfold外) OOF {roc_auc_score(y, rank_oof):.6f} 重み {[np.round(w, 2).tolist() for w in chosen_weights]}")
