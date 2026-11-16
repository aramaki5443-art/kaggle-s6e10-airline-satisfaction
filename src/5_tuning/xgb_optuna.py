"""XGBoostのハイパラをOptunaで複数同時に探す。

目的: 既定ハイパラのXGBoost(36列、OOF 0.960527)を調整し、スタッキングへの上乗せが増えるかを確かめる。
ひとことで言うと: LightGBMの調整(lgb_optuna.py)と同じやり方を、XGBoostの8つのつまみに当てはめる。
やり方:
  - 特徴量・TE・fold分割(seed42の5fold)・学習率0.05+早期停止100本はxgb_check.pyと同じ。TEはfoldごとに1回だけ計算して使い回す。
  - 1回目の試行は既定の設定(max_depth 6・min_child_weight 1・L2 1・L1は範囲の下限1e-3・gamma 0・間引きなし・max_bin 255)。
  - 途中のfoldまでの平均AUCが過去の試行の中央値より低ければ打ち切る(MedianPruner)。結果はdata/optuna_xgb.dbに保存し、
    再実行すると続きから回る。OOFは各試行ごとにdata/optuna_xgb_oof/に保存し、上位の設定のスタックでの伸びを後で測れるようにする。
なぜ: 単体の最良がスタックでの最良とは限らない(調整後のLightGBMはRealMLPと似て、混ぜると伸びが縮んだ)ため、
  上位数個のOOFでstack_check.pyを回してから選ぶ。seed42に合わせ込んだだけの設定を拾わないよう、選んだ設定は3seedで確認する。
結果: 最良#35(max_depth 10・min_child_weight 23・L2 17・L1 0.11・gamma 0.10・max_bin 1023・subsample 0.97・colsample 0.45)が
  3seed平均+0.000244(3seedとも改善)。ただしスタックへの上乗せは+0.000008にとどまった(調整後LightGBMとの順位相関0.992)。
使い方: python src/5_tuning/xgb_optuna.py --n-trials 50
"""
import argparse
import os
import time

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# features.pyの特徴量の定義(関数・ADOPTED・TE)を読み込む
src = open("src/3_feature_engineering/features.py", encoding="utf-8").read()
g = {"__name__": "fe_defs"}
exec(compile(src, "features.py", "exec"), g)

ap = argparse.ArgumentParser()
ap.add_argument("--n-trials", type=int, default=50)
ap.add_argument("--study", default="xgb_c36_v1")
args = ap.parse_args()

# データの置き場所・列名(リポジトリのルートで実行する前提)
train = pd.read_csv("data/train.csv")
y = train["satisfaction"].astype(int).values
X = pd.concat([train[g["BASE"]]] + [g["FE"][a](train) for a in g["ADOPTED"]], axis=1)
for c in g["CAT"]:
    X[c] = X[c].astype("category")
print(f"{X.shape[1]}列", flush=True)
folds = []
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    Xtr, (Xva,) = g["te_fold"](X.iloc[tr], y[tr], [X.iloc[va]], seed=k)  # features.pyと同じTE
    folds.append((Xtr, y[tr], Xva, va))
os.makedirs("data/optuna_xgb_oof", exist_ok=True)

BASE = dict(learning_rate=0.05, n_estimators=10000, early_stopping_rounds=100, tree_method="hist", enable_categorical=True,
            eval_metric="auc", random_state=42, n_jobs=-1)
DEFAULT = dict(max_depth=6, min_child_weight=1.0, reg_lambda=1.0, reg_alpha=1e-3, gamma=0.0, max_bin=255,
               subsample=1.0, colsample_bytree=1.0)


def objective(trial):
    p = dict(max_depth=trial.suggest_int("max_depth", 3, 12),
             min_child_weight=trial.suggest_float("min_child_weight", 0.1, 100, log=True),  # 葉の最少の重み(葉の最少人数に相当)
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100, log=True),  # L2
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),  # L1
             gamma=trial.suggest_float("gamma", 0.0, 1.0),  # 分岐に必要な最小の改善(LightGBMのmin_split_gain)
             max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511, 1023]),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),  # 行の間引き
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0))  # 列の間引き
    oof = np.zeros(len(y)); aucs, iters = [], []; t0 = time.time()
    for k, (Xtr, ytr, Xva, va) in enumerate(folds):
        m = xgb.XGBClassifier(**BASE, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, y[va])], verbose=False)
        oof[va] = m.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(y[va], oof[va])); iters.append(int(m.best_iteration))
        trial.report(float(np.mean(aucs)), k)
        if trial.should_prune():
            print(f"#{trial.number} 打ち切り fold{k} 平均{np.mean(aucs):.6f} {time.time() - t0:.0f}s {p}", flush=True)
            raise optuna.TrialPruned()
    auc = roc_auc_score(y, oof)
    np.save(f"data/optuna_xgb_oof/oof_{args.study}_t{trial.number}.npy", oof)
    trial.set_user_attr("iters", iters); trial.set_user_attr("va_folds", aucs); trial.set_user_attr("sec", round(time.time() - t0))
    print(f"#{trial.number} OOF {auc:.6f} iters {iters} {time.time() - t0:.0f}s {p}", flush=True)
    return auc


study = optuna.create_study(study_name=args.study, storage="sqlite:///data/optuna_xgb.db", direction="maximize", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
if len(study.trials) == 0:
    study.enqueue_trial(DEFAULT)
done = len([t for t in study.trials if t.state.is_finished()])
study.optimize(objective, n_trials=max(0, args.n_trials - done))
top = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
for t in top:
    print(f"上位 #{t.number} OOF {t.value:.6f} {t.params} iters {t.user_attrs.get('iters')}")
