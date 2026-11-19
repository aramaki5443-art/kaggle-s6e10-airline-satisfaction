"""LightGBMのハイパラをOptunaで複数同時に探す。

目的: 1つずつのグリッドサーチ(num_leavesは31が最良)では、パラメータどうしの組み合わせ効果(例: 葉を増やすなら
  葉の最少人数も増やす)を見逃すため、8つを同時に探す。
ひとことで言うと: 過去の試行の結果から「次はこのあたりが良さそう」と推測しながら、値の組み合わせを50通りほど試す。
やり方:
  - 特徴量は採用済みの36列(feature_definitions.pyのADOPTED_FEATURES)で固定。fold分割はseed42の5fold。学習率は0.05に固定し、
    木の本数は検証foldでの早期停止(100本)で決める(train_lightgbm.pyと同じ。学習率は最後に下げる)。
  - 物差しは5foldのOOF AUC。1回目の試行は既定に近い設定(正則化0の代わりに1e-3、OOF 0.960493。完全な既定は0.960458)。
  - TEの列はfoldごとに1回だけ計算して使い回す(試行ごとに同じ値になるため)。
  - 見込みのない試行は、途中のfoldまでの平均AUCが過去の試行の中央値より低ければ打ち切る(MedianPruner)。
  - 結果はdata/optuna_lgb.db(SQLite)に保存し、再実行すると続きから回る。
なぜ: seed42のfold分割に合わせ込んだだけの設定を拾わないよう、上位の設定は後で3seed(42・0・1)で既定と比べる。
結果: 50試行の最良#14(num_leaves 61・min_child_samples 5・lambda_l2 29.3・lambda_l1 0.34・min_split_gain 0.35・
  max_bin 511・subsample 0.90・colsample 0.51)が3seed平均+0.000319。学習率0.02で+0.000388(0.01は0.02と見分けられず)。
"""
import argparse
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 特徴量の定義を読み込む(フォルダ名が数字で始まりパッケージにできないため、そのフォルダを検索パスに足す)
sys.path.insert(0, "src/2_lightgbm_baseline_and_feature_engineering")
import feature_definitions as features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-trials", type=int, default=50)
ap.add_argument("--study", default="lgb_c33_v1")
args = ap.parse_args()

# データの置き場所・列名・foldの分け方(リポジトリのルートで実行する前提)
config = {"train_csv": "data/train.csv", "test_csv": "data/test.csv", "id_col": "id", "target_col": "satisfaction",
          "cv": {"n_splits": 5, "seed": 42}}
train = pd.read_csv(config["train_csv"])
y = train[config["target_col"]].astype(int).values
X = pd.concat([train[features.BASE_FEATURES]] + [features.FEATURE_BUILDERS[a](train) for a in features.ADOPTED_FEATURES], axis=1)
for c in features.CATEGORICAL_COLUMNS:
    X[c] = X[c].astype("category")
print(f"{X.shape[1]}列", flush=True)
folds = []
for k, (train_rows, valid_rows) in enumerate(StratifiedKFold(config["cv"]["n_splits"], shuffle=True, random_state=config["cv"]["seed"]).split(X, y)):
    X_fit, (X_valid,) = features.target_encode_in_folds(X.iloc[train_rows], y[train_rows], [X.iloc[valid_rows]], seed=k)  # feature_definitions.pyと同じTE
    folds.append((X_fit, y[train_rows], X_valid, valid_rows))

FIXED_PARAMS = dict(objective="binary", learning_rate=0.05, n_estimators=10000, random_state=42, verbose=-1)
DEFAULT_PARAMS = dict(num_leaves=31, min_child_samples=20, reg_lambda=1e-3, reg_alpha=1e-3, min_split_gain=0.0, max_bin=255,
                      subsample=1.0, colsample_bytree=1.0)  # 既定設定(正則化0の代わりに範囲の下限1e-3)


def objective(trial):
    params = dict(num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                  min_child_samples=trial.suggest_int("min_child_samples", 5, 500, log=True),
                  reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100, log=True),  # lambda_l2
                  reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),  # lambda_l1
                  min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
                  max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511, 1023]),
                  subsample=trial.suggest_float("subsample", 0.5, 1.0),  # bagging_fraction(行の間引き)
                  colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0))  # feature_fraction(列の間引き)
    oof = np.zeros(len(y)); aucs, best_iterations = [], []; start_time = time.time()
    for k, (X_fit, y_fit, X_valid, valid_rows) in enumerate(folds):
        model = lgb.LGBMClassifier(**FIXED_PARAMS, **params, subsample_freq=1)
        model.fit(X_fit, y_fit, eval_set=[(X_valid, y[valid_rows])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[valid_rows] = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y[valid_rows], oof[valid_rows])); best_iterations.append(model.best_iteration_)
        trial.report(float(np.mean(aucs)), k)
        if trial.should_prune():
            print(f"#{trial.number} 打ち切り fold{k} 平均{np.mean(aucs):.6f} {params}", flush=True)
            raise optuna.TrialPruned()
    auc = roc_auc_score(y, oof)
    trial.set_user_attr("iters", best_iterations); trial.set_user_attr("va_folds", aucs); trial.set_user_attr("sec", round(time.time() - start_time))
    print(f"#{trial.number} OOF {auc:.6f} iters {best_iterations} {time.time() - start_time:.0f}s {params}", flush=True)
    return auc


study = optuna.create_study(study_name=args.study, storage="sqlite:///data/optuna_lgb.db", direction="maximize", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
if len(study.trials) == 0:
    study.enqueue_trial(DEFAULT_PARAMS)
done = len([t for t in study.trials if t.state.is_finished()])
study.optimize(objective, n_trials=max(0, args.n_trials - done))
best_trials = sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]
for t in best_trials:
    print(f"上位 #{t.number} OOF {t.value:.6f} {t.params} iters {t.user_attrs.get('iters')}")
