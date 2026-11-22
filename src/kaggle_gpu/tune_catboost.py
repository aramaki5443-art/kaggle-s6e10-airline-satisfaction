"""CatBoost(全特徴量をカテゴリ扱い)のハイパラを、KaggleのGPUでOptunaにより探す(notebooks/04_hyperparameter_tuning の3節)。

- データ・値ごとの満足率の置き換え・全特徴量のカテゴリ化は train_catboost.py と同じ。分け方はseed 42の5分割、学習率0.05、早期停止100本。
- 探すハイパラ: depth 4〜10・l2_leaf_reg 1〜30・border_count 32/64/128/254・random_strength 0.1〜10・bagging_temperature 0〜1・
  max_ctr_complexity 1〜4(カテゴリを何個まで組み合わせて集計するか)。1試行目は既定の設定。
- 途中のfoldまでの平均AUCが過去の中央値より低ければ打ち切る。KaggleのGPUの12時間の上限に収まるよう、10.8時間で新しい試行を止める。
- 結果: 最良の試行#21(depth 7・l2 24.9・border 254・random_strength 0.17・bagging_temperature 0.42・max_ctr_complexity 4)が
  3seed平均+0.000076(3seedとも改善)。5モデルのスタックへの上乗せは+0.000020。
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

N_TRIALS, LIMIT_HOURS = 30, 10.8
TARGET_ENCODING_SPECS = {"distance_satisfaction_rate": ("Flight Distance", 3), "distance_class_satisfaction_rate": (["Flight Distance", "Class"], 3)}
N_BASE_FEATURES = 20  # 先頭20列が元の特徴量(src/features.pyのBASE_FEATURES)
argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()  # 引数は無い。--helpで説明だけ出して止まるようにする
START_TIME = time.time()

kaggle_files = glob.glob("/kaggle/input/**/features36_train.csv", recursive=True)
data_dir = os.path.dirname(kaggle_files[0]) if kaggle_files else "data"  # Kaggleではデータセット、手元ではdata/(GPUが要る)
X = pd.read_csv(f"{data_dir}/features36_train.csv")
y = X.pop("satisfaction").values
X = X.drop(columns="id")
base_cols = list(X.columns[:N_BASE_FEATURES])
X = X.assign(**{c: X[c].fillna(-1).astype(str) for c in base_cols})  # 全列カテゴリ化(train_catboost.pyのallcatと同じ)


def target_encode(key_fit, y_fit, key_apply, model):  # src/features.pyと同一
    prior = y_fit.mean()
    stats = pd.DataFrame({"key": key_fit, "y": y_fit}).groupby("key")["y"].agg(["sum", "count"])
    return pd.Series(key_apply).map((stats["sum"] + model * prior) / (stats["count"] + model)).fillna(prior).values


def target_encode_in_folds(X_fit, y_fit, others, seed):  # src/features.pyと同一
    X_fit = X_fit.copy(); others = [other.copy() for other in others]
    for name, (_, model) in TARGET_ENCODING_SPECS.items():
        key = X_fit[name].values; encoded = np.zeros(len(X_fit))
        for inner_fit, inner_apply in StratifiedKFold(5, shuffle=True, random_state=seed).split(key, y_fit):
            encoded[inner_apply] = target_encode(key[inner_fit], y_fit[inner_fit], key[inner_apply], model)
        for other in others:
            other[name] = target_encode(key, y_fit, other[name].values, model)
        X_fit[name] = encoded
    return X_fit, others


folds = []
for k, (train_rows, valid_rows) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    X_fit, (X_valid,) = target_encode_in_folds(X.iloc[train_rows], y[train_rows], [X.iloc[valid_rows]], seed=k)
    folds.append((X_fit, y[train_rows], X_valid, valid_rows))
FIXED_PARAMS = dict(iterations=20000, learning_rate=0.05, eval_metric="AUC", random_seed=42, task_type="GPU",
                    od_type="Iter", od_wait=100, use_best_model=True, verbose=0)


def objective(trial):
    params = dict(depth=trial.suggest_int("depth", 4, 10),
                  l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
                  border_count=trial.suggest_categorical("border_count", [32, 64, 128, 254]),
                  random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
                  bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                  max_ctr_complexity=trial.suggest_int("max_ctr_complexity", 1, 4))
    oof = np.zeros(len(y)); aucs, best_iterations = [], []; start_time = time.time()
    for k, (X_fit, y_fit, X_valid, valid_rows) in enumerate(folds):
        model = CatBoostClassifier(**FIXED_PARAMS, **params, cat_features=base_cols)
        model.fit(X_fit, y_fit, eval_set=(X_valid, y[valid_rows]))
        oof[valid_rows] = model.predict_proba(X_valid)[:, 1]
        aucs.append(roc_auc_score(y[valid_rows], oof[valid_rows])); best_iterations.append(int(model.get_best_iteration()))
        trial.report(float(np.mean(aucs)), k)
        if trial.should_prune():
            print(f"#{trial.number} 打ち切り fold{k} 平均{np.mean(aucs):.6f} {time.time() - start_time:.0f}s {params}", flush=True)
            raise optuna.TrialPruned()
    auc = roc_auc_score(y, oof)
    np.save(f"optuna_cat/oof_t{trial.number}.npy", oof)
    result = {"trial": trial.number, "oof": auc, "params": params, "va_folds": aucs, "iters": best_iterations, "sec": round(time.time() - start_time)}
    open("optuna_cat/results.jsonl", "a").write(json.dumps(result) + "\n")
    print(f"#{trial.number} OOF {auc:.6f} iters {best_iterations} {time.time() - start_time:.0f}s {params}", flush=True)
    return auc


# 1回目は既定の設定: depth 6・l2 3・border 128・random_strength 1・bagging_temperature 1。max_ctr_complexityの既定は自動のため4を入れる
os.makedirs("optuna_cat", exist_ok=True)  # 探索の記録(データベース・各試行のOOF)の保存先。探索を始める直前に作る
study = optuna.create_study(study_name="cat_allcat_v1", storage="sqlite:///optuna_cat/optuna_cat.db", direction="maximize",
                            load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1))
study.enqueue_trial(dict(depth=6, l2_leaf_reg=3.0, border_count=128, random_strength=1.0, bagging_temperature=1.0, max_ctr_complexity=4))
study.optimize(objective, n_trials=N_TRIALS, timeout=LIMIT_HOURS * 3600 - (time.time() - START_TIME))
for t in sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]:
    print(f"上位 #{t.number} OOF {t.value:.6f} {t.params} iters {t.user_attrs.get('iters')}")
