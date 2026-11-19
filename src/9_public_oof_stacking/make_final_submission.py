"""最終提出のスタック: 自分のモデル4本+公開OOF 222本(計226本)を2段目で混ぜ、提出ファイルを作る。

目的: 9の「公開OOFの導入」で決めた最終構成を、メンバーの一覧(final_stack_members.csv)から作り直す。
ひとことで言うと: 226本の予測を「重み付きの足し算」でまとめる係(ロジスティック回帰)に、
  「この予測とこの予測が割れたら…」と条件で使い分ける係(浅いLightGBM)の意見を2割だけ足す。
やり方:
  - メンバーはfinal_stack_members.csvの順に読む(folder=ownは data/oof_<name>.npy、それ以外は data/<folder>/oof_<name>.npy。
    testは同じ場所の test_<name>.npy)。各予測をlogit=log(p/(1-p))にして並べる。
  - 評価: 1段目とは別の5分割(--meta-seed、既定7)で、4つで学び残り1つで採点する。各foldで、ロジスティック回帰(L2、C=0.003)と
    浅いLightGBM(学習率0.03・葉15・葉の最小行数2000・列の抽出0.5・行の抽出0.8・L2=10。木の本数は学習行の1割で早期停止)を
    同じ行で学ぶ。2つの予測をそれぞれ平均0・標準偏差1にそろえ、0.8×ロジスティック回帰+0.2×LightGBMで混ぜてAUCを出す。
  - --saveを付けると、全行で学び直してtestに当て、data/submission_<tag>.csvを作る。LightGBMは--lgb-seeds個のseedで平均する
    (testのぶれを減らすため。最終提出は5)。木の本数は評価時の平均の1.1倍(全行で学ぶぶん少し増やす)。
なぜ: ロジスティック回帰は安定だが、メンバーどうしの条件付きの関係は表せない。浅く強く正則化した木を少し混ぜると、
  その関係を当てはめすぎずに拾える。混ぜる割合0.2と木の設定は公開ノートブックの値をそのまま使った(手元で選ぶと甘くなるため)。
使い方: python src/9_public_oof_stacking/make_final_submission.py --tag final226 --lgb-seeds 5 --save
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--members", default="src/9_public_oof_stacking/final_stack_members.csv")
ap.add_argument("--C", type=float, default=0.003, help="ロジスティック回帰の正則化の強さ(小さいほど強い)")
ap.add_argument("--meta-seed", type=int, default=7, help="2段目の評価の5分割のseed")
ap.add_argument("--w", type=float, default=0.2, help="LightGBMを混ぜる割合")
ap.add_argument("--lgb-seeds", type=int, default=1, help="--saveのとき、全行で学ぶLightGBMをこの数のseedで平均する")
ap.add_argument("--tag", default="final226")
ap.add_argument("--save", action="store_true", help="全行で学び直してtestに当て、提出ファイルを作る")
args = ap.parse_args()

members = pd.read_csv(args.members)
y = pd.read_csv("data/train.csv", usecols=["satisfaction"])["satisfaction"].astype(int).values


def member_path(member, kind):
    """kind: "oof" または "test"。自分のモデル(folder=own)は data/ 直下、公開OOFは data/<folder>/ にある。"""
    folder = "data" if member["folder"] == "own" else f"data/{member['folder']}"
    return f"{folder}/{kind}_{member['name']}.npy"


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def load_logits(kind):
    return np.column_stack([to_logit(np.load(member_path(m, kind)).astype(float).ravel()) for _, m in members.iterrows()])


def standardize(v):
    return (v - v.mean()) / v.std()


oof_logits = load_logits("oof")
print(f"メンバー{oof_logits.shape[1]}本", flush=True)
STACKER_LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
                          bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)


def train_lightgbm_stacker(X_train, y_train):
    # 木の本数は学習行の1割で決める(採点するfoldの答えは使わない)
    X_fit, X_stop, y_fit, y_stop = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=0)
    return lgb.train(STACKER_LGB_PARAMS, lgb.Dataset(X_fit, y_fit), 5000, valid_sets=[lgb.Dataset(X_stop, y_stop)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


lr_oof, lgb_oof, best_iterations = np.zeros(len(y)), np.zeros(len(y)), []
folds = StratifiedKFold(5, shuffle=True, random_state=args.meta_seed).split(oof_logits, y)
for k, (train_rows, valid_rows) in enumerate(folds):
    X_train, y_train, X_valid = oof_logits[train_rows], y[train_rows], oof_logits[valid_rows]
    lr_oof[valid_rows] = LogisticRegression(C=args.C, max_iter=2000).fit(X_train, y_train).decision_function(X_valid)
    stacker = train_lightgbm_stacker(X_train, y_train)
    lgb_oof[valid_rows] = stacker.predict(X_valid, raw_score=True)
    best_iterations.append(stacker.best_iteration)
    print(f"fold{k}: ロジスティック回帰 {roc_auc_score(y[valid_rows], lr_oof[valid_rows]):.6f} "
          f"LightGBM {roc_auc_score(y[valid_rows], lgb_oof[valid_rows]):.6f} 木{stacker.best_iteration}本", flush=True)
print(f"ロジスティック回帰だけ {roc_auc_score(y, lr_oof):.6f} / LightGBMだけ {roc_auc_score(y, lgb_oof):.6f}")
print(f"スタックのOOF AUC(w={args.w}): {roc_auc_score(y, (1 - args.w) * standardize(lr_oof) + args.w * standardize(lgb_oof)):.7f}")
if args.save:
    test_logits = load_logits("test")
    lr_test = LogisticRegression(C=args.C, max_iter=2000).fit(oof_logits, y).decision_function(test_logits)
    n_trees = int(np.mean(best_iterations) * 1.1)
    lgb_test = np.mean([lgb.train({**STACKER_LGB_PARAMS, "seed": seed}, lgb.Dataset(oof_logits, y), n_trees).predict(test_logits, raw_score=True)
                        for seed in range(args.lgb_seeds)], axis=0)
    score = (1 - args.w) * standardize(lr_test) + args.w * standardize(lgb_test)
    ids = pd.read_csv("data/test.csv", usecols=["id"])["id"]
    pd.DataFrame({"id": ids, "satisfaction": 1 / (1 + np.exp(-score))}).to_csv(f"data/submission_{args.tag}.csv", index=False)
    print(f"保存 data/submission_{args.tag}.csv")
