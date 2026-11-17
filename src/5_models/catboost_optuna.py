"""CatBoost(全列カテゴリ化)のハイパラを、KaggleのGPUでOptunaにより探す。

目的: 全列カテゴリ化で単体0.960832になったCatBoostを調整し、スタッキングへの上乗せが増えるかを確かめる。
ひとことで言うと: LightGBM・XGBoostの調整と同じやり方を、CatBoostの6つのつまみに当てはめる。
やり方:
  - データ・TE・全列カテゴリ化はcatboost_kaggle.pyと同一(入力はlgb_check.py --export-feで書き出したfe36_*.csv。
    Kaggleではデータセットとして読み込ませる。生の20列を文字列のカテゴリ列に、TEの2列はfoldごとに計算)。
    fold分割はseed42の5fold、学習率0.05、早期停止100本。TEはfoldごとに1回だけ計算して使い回す。
  - つまみ: depth 4〜10・l2_leaf_reg 1〜30・border_count 32/64/128/254・random_strength 0.1〜10・bagging_temperature 0〜1・
    max_ctr_complexity 1〜4(カテゴリ列を何列まで組み合わせて集計するか)。1回目は既定の設定。
  - 途中のfoldまでの平均AUCが過去の中央値より低ければ打ち切る(MedianPruner)。KaggleのGPUの12時間の上限に収まるよう、
    10.8時間で新しい試行を止める。
  - 各試行のOOFと結果をoptuna_cat/に保存する(上位の設定のスタックでの伸びを後で測るため)。testの予測は選んだ設定だけ後で作る。
なぜ: 公開ノートブックで深さ6〜7・l2 10が+0.00007と報告されていた。全列をカテゴリ列にしたため、組み合わせの深さ
  (max_ctr_complexity)が効きやすいと考えた。単体の最良がスタックの最良とは限らないため、上位数個で比べる。
結果: 最良#21(depth 7・l2 24.9・border 254・random_strength 0.17・bagging_temperature 0.42・max_ctr_complexity 4)が
  3seed平均+0.000076(3seedとも改善)。スタックへの上乗せは+0.000020(RealMLPとの順位相関が0.9700→0.9715に上昇)。
"""
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

N_TRIALS, LIMIT_H = 30, 10.8
TE = {"te_distance_m3": ("Flight Distance", 3), "te_dist_class": (["Flight Distance", "Class"], 3)}
RAW_N = 20  # 先頭20列が生の列(features.pyのBASE)
T0 = time.time()
os.makedirs("optuna_cat", exist_ok=True)

hit = glob.glob("/kaggle/input/**/fe36_train.csv", recursive=True)
d = os.path.dirname(hit[0]) if hit else "data"  # Kaggleではデータセット、手元ではdata/(GPUが要る)
X = pd.read_csv(f"{d}/fe36_train.csv")
y = X.pop("satisfaction").values
X = X.drop(columns="id")
raw = list(X.columns[:RAW_N])
X = X.assign(**{c: X[c].fillna(-1).astype(str) for c in raw})  # 全列カテゴリ化(catboost_kaggle.pyのallcatと同一)


def te_map(key_fit, y_fit, key_apply, m):  # features.pyと同一
    p = y_fit.mean()
    g = pd.DataFrame({"k": key_fit, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
    return pd.Series(key_apply).map((g["sum"] + m * p) / (g["count"] + m)).fillna(p).values


def te_fold(Xtr, ytr, others, seed):  # features.pyと同一
    Xtr = Xtr.copy(); others = [o.copy() for o in others]
    for name, (_, m) in TE.items():
        key = Xtr[name].values; v = np.zeros(len(Xtr))
        for itr, iva in StratifiedKFold(5, shuffle=True, random_state=seed).split(key, ytr):
            v[iva] = te_map(key[itr], ytr[itr], key[iva], m)
        for o in others:
            o[name] = te_map(key, ytr, o[name].values, m)
        Xtr[name] = v
    return Xtr, others


folds = []
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    Xtr, (Xva,) = te_fold(X.iloc[tr], y[tr], [X.iloc[va]], seed=k)
    folds.append((Xtr, y[tr], Xva, va))
BASE = dict(iterations=20000, learning_rate=0.05, eval_metric="AUC", random_seed=42, task_type="GPU",
            od_type="Iter", od_wait=100, use_best_model=True, verbose=0)


def objective(trial):
    p = dict(depth=trial.suggest_int("depth", 4, 10),
             l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
             border_count=trial.suggest_categorical("border_count", [32, 64, 128, 254]),
             random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
             bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
             max_ctr_complexity=trial.suggest_int("max_ctr_complexity", 1, 4))
    oof = np.zeros(len(y)); aucs, iters = [], []; t0 = time.time()
    for k, (Xtr, ytr, Xva, va) in enumerate(folds):
        m = CatBoostClassifier(**BASE, **p, cat_features=raw)
        m.fit(Xtr, ytr, eval_set=(Xva, y[va]))
        oof[va] = m.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(y[va], oof[va])); iters.append(int(m.get_best_iteration()))
        trial.report(float(np.mean(aucs)), k)
        if trial.should_prune():
            print(f"#{trial.number} 打ち切り fold{k} 平均{np.mean(aucs):.6f} {time.time() - t0:.0f}s {p}", flush=True)
            raise optuna.TrialPruned()
    auc = roc_auc_score(y, oof)
    np.save(f"optuna_cat/oof_t{trial.number}.npy", oof)
    r = {"trial": trial.number, "oof": auc, "params": p, "va_folds": aucs, "iters": iters, "sec": round(time.time() - t0)}
    open("optuna_cat/results.jsonl", "a").write(json.dumps(r) + "\n")
    print(f"#{trial.number} OOF {auc:.6f} iters {iters} {time.time() - t0:.0f}s {p}", flush=True)
    return auc


# 1回目は既定の設定: depth 6・l2 3・border 128・random_strength 1・bagging_temperature 1。max_ctr_complexityの既定は自動のため4を入れる
study = optuna.create_study(study_name="cat_allcat_v1", storage="sqlite:///optuna_cat/optuna_cat.db", direction="maximize",
                            load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1))
study.enqueue_trial(dict(depth=6, l2_leaf_reg=3.0, border_count=128, random_strength=1.0, bagging_temperature=1.0, max_ctr_complexity=4))
study.optimize(objective, n_trials=N_TRIALS, timeout=LIMIT_H * 3600 - (time.time() - T0))
for t in sorted([t for t in study.trials if t.value is not None], key=lambda t: -t.value)[:5]:
    print(f"上位 #{t.number} OOF {t.value:.6f} {t.params} iters {t.user_attrs.get('iters')}")
