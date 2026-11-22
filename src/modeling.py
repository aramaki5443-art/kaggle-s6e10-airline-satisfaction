"""学習と評価の共通処理: 5分割の交差検証(LightGBM・XGBoost)、採用したハイパラ、Optunaの探索、2段目のスタッキング。

ノートブック(notebooks/03〜06)から読み込んで使う。分け方はどのモデルも StratifiedKFold(5, shuffle=True, random_state=42) にそろえ、
同じ行どうしでOOF(各行を、その行を学習していないモデルで予測した値)を比べたり混ぜたりできるようにしている。
"""
import itertools
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import features

N_SPLITS, CV_SEED = 5, 42

DEFAULT_LIGHTGBM_PARAMS = dict(objective="binary", learning_rate=0.05, n_estimators=10000, random_state=42, verbose=-1)
# 04_hyperparameter_tuning で採用した設定(Optunaの試行14の値に、学習率を0.05→0.02に下げたもの)
TUNED_LIGHTGBM_PARAMS = {**DEFAULT_LIGHTGBM_PARAMS, "num_leaves": 61, "min_child_samples": 5, "reg_lambda": 29.265955275846228,
                         "reg_alpha": 0.3399216318523539, "min_split_gain": 0.34853053056853356, "max_bin": 511,
                         "subsample": 0.9039977941161244, "subsample_freq": 1, "colsample_bytree": 0.5146287416154858, "learning_rate": 0.02}
DEFAULT_XGBOOST_PARAMS = dict(learning_rate=0.05, n_estimators=10000, early_stopping_rounds=100, tree_method="hist",
                              enable_categorical=True, eval_metric="auc", random_state=42, n_jobs=-1)
# 04_hyperparameter_tuning で採用した設定(Optunaの試行35の値)
TUNED_XGBOOST_PARAMS = {**DEFAULT_XGBOOST_PARAMS, "max_depth": 10, "min_child_weight": 23.27750908295225, "reg_lambda": 17.154422085714742,
                        "reg_alpha": 0.10595012492749806, "gamma": 0.09813717336201243, "max_bin": 1023,
                        "subsample": 0.9653056625598389, "colsample_bytree": 0.4517043587579834}


def cross_validate_lightgbm(X, y, params=DEFAULT_LIGHTGBM_PARAMS, X_test=None, cv_seed=CV_SEED, verbose=True):
    """LightGBMを5分割で学習し、OOFとtestの予測(5回の平均)を返す。木の本数は検証foldのAUCで早期停止して決める。
    Target Encodingの列があれば、foldごとに答えを見ない満足率へ置き換える(features.target_encode_in_folds)。"""
    oof, test_prediction = np.zeros(len(y)), None if X_test is None else np.zeros(len(X_test))
    start = time.time()
    for k, (fit_rows, valid_rows) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=cv_seed).split(X, y)):
        X_fit, (X_valid, X_test_fold) = features.target_encode_in_folds(
            X.iloc[fit_rows], y[fit_rows], [X.iloc[valid_rows], X.iloc[:1] if X_test is None else X_test], seed=k)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_fit, y[fit_rows], eval_X=(X_valid,), eval_y=(y[valid_rows],), eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[valid_rows] = model.predict_proba(X_valid)[:, 1]
        if X_test is not None:
            test_prediction += model.predict_proba(X_test_fold)[:, 1] / N_SPLITS
        if verbose:
            print(f"fold{k}: AUC {roc_auc_score(y[valid_rows], oof[valid_rows]):.6f} 木{model.best_iteration_}本 ({time.time() - start:.0f}秒)")
    return oof, test_prediction


def cross_validate_xgboost(X, y, params=DEFAULT_XGBOOST_PARAMS, X_test=None, cv_seed=CV_SEED, verbose=True):
    """XGBoostを5分割で学習する(分け方・Target Encodingの扱いは cross_validate_lightgbm と同じ)。"""
    oof, test_prediction = np.zeros(len(y)), None if X_test is None else np.zeros(len(X_test))
    start = time.time()
    for k, (fit_rows, valid_rows) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=cv_seed).split(X, y)):
        X_fit, encoded = features.target_encode_in_folds(X.iloc[fit_rows], y[fit_rows], [X.iloc[valid_rows]] + ([] if X_test is None else [X_test]), seed=k)
        model = xgb.XGBClassifier(**params)
        model.fit(X_fit, y[fit_rows], eval_set=[(encoded[0], y[valid_rows])], verbose=False)
        oof[valid_rows] = model.predict_proba(encoded[0])[:, 1]
        if X_test is not None:
            test_prediction += model.predict_proba(encoded[1])[:, 1] / N_SPLITS
        if verbose:
            print(f"fold{k}: AUC {roc_auc_score(y[valid_rows], oof[valid_rows]):.6f} 木{int(model.best_iteration)}本 ({time.time() - start:.0f}秒)")
    return oof, test_prediction


def rank_correlation_table(predictions):
    """モデルの名前 → OOF の辞書から、予測の順位相関(1に近いほど似た予測)の表を作る。"""
    names = list(predictions)
    return pd.DataFrame([[spearmanr(predictions[a], predictions[b])[0] for b in names] for a in names], index=names, columns=names)


# Optunaのハイパラ探索(5分割のOOF AUCを最大にする。見込みの薄い試行はfoldの途中で打ち切る)

def prepare_tuning_folds(X, y):
    """探索のたびにTarget Encodingを計算し直さないよう、5つのfoldを先に作っておく。"""
    folds = []
    for k, (fit_rows, valid_rows) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=CV_SEED).split(X, y)):
        X_fit, (X_valid,) = features.target_encode_in_folds(X.iloc[fit_rows], y[fit_rows], [X.iloc[valid_rows]], seed=k)
        folds.append((X_fit, y[fit_rows], X_valid, valid_rows))
    return folds


def suggest_lightgbm_params(trial):
    return dict(num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 500, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
                max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511, 1023]),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),  # 行の間引き
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0))  # 特徴量の間引き


def suggest_xgboost_params(trial):
    return dict(max_depth=trial.suggest_int("max_depth", 3, 12),
                min_child_weight=trial.suggest_float("min_child_weight", 0.1, 100, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                gamma=trial.suggest_float("gamma", 0.0, 1.0),
                max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511, 1023]),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0))


def run_optuna_search(model_name, folds, y, n_trials, storage, study_name):
    """model_name("lightgbm" か "xgboost")のハイパラをTPEで探す。1試行目は既定の設定。結果はstorage(SQLite)に保存し、続きから再開できる。"""
    defaults = {"lightgbm": dict(num_leaves=31, min_child_samples=20, reg_lambda=1e-3, reg_alpha=1e-3, min_split_gain=0.0, max_bin=255,
                                 subsample=1.0, colsample_bytree=1.0),
                "xgboost": dict(max_depth=6, min_child_weight=1.0, reg_lambda=1.0, reg_alpha=1e-3, gamma=0.0, max_bin=255,
                                subsample=1.0, colsample_bytree=1.0)}[model_name]

    def objective(trial):
        params = suggest_lightgbm_params(trial) if model_name == "lightgbm" else suggest_xgboost_params(trial)
        oof, fold_aucs = np.zeros(len(y)), []
        for k, (X_fit, y_fit, X_valid, valid_rows) in enumerate(folds):
            if model_name == "lightgbm":
                model = lgb.LGBMClassifier(**DEFAULT_LIGHTGBM_PARAMS, **params, subsample_freq=1)
                model.fit(X_fit, y_fit, eval_set=[(X_valid, y[valid_rows])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
            else:
                model = xgb.XGBClassifier(**DEFAULT_XGBOOST_PARAMS, **params)
                model.fit(X_fit, y_fit, eval_set=[(X_valid, y[valid_rows])], verbose=False)
            oof[valid_rows] = model.predict_proba(X_valid)[:, 1]
            fold_aucs.append(roc_auc_score(y[valid_rows], oof[valid_rows]))
            trial.report(float(np.mean(fold_aucs)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return roc_auc_score(y, oof)

    study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    if len(study.trials) == 0:
        study.enqueue_trial(defaults)
    finished = len([t for t in study.trials if t.state.is_finished()])
    study.optimize(objective, n_trials=max(0, n_trials - finished))
    return study


# 2段目(スタッキング): 1段目のOOFのlogitから、答えに一番合う重みを学ぶ

def to_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def standardize(v):
    return (v - v.mean()) / v.std()


def cross_validate_logistic_stack(logits, y, C=1.0, meta_seed=7, max_iter=1000):
    """2段目のロジスティック回帰を「4つで重みを学び、残り1つで採点」×5回で評価し、(スタックのOOF, foldごとの係数)を返す。
    重みを決めた行で採点すると点数が甘くなるので、提出用とは別にこの方法で測る。"""
    stack_oof, coefficients = np.zeros(len(y)), []
    for fit_rows, valid_rows in StratifiedKFold(5, shuffle=True, random_state=meta_seed).split(logits, y):
        stacker = LogisticRegression(C=C, max_iter=max_iter).fit(logits[fit_rows], y[fit_rows])
        stack_oof[valid_rows] = stacker.decision_function(logits[valid_rows])
        coefficients.append(stacker.coef_[0])
    return stack_oof, np.array(coefficients)


def cross_validate_rank_blend(predictions, y, weight_step=0.05, meta_seed=7):
    """比較用: 予測を順位に直し、重み(合計1、weight_step刻みの全組み合わせ)を学習側の行で選んで平均する。"""
    ranks = np.column_stack([rankdata(predictions[:, j]) / len(y) for j in range(predictions.shape[1])])
    grid = [w for w in itertools.product(np.arange(0, 1 + 1e-9, weight_step), repeat=predictions.shape[1] - 1) if sum(w) <= 1 + 1e-9]
    grid = [np.array(list(w) + [1 - sum(w)]) for w in grid]
    blend_oof = np.zeros(len(y))
    for fit_rows, valid_rows in StratifiedKFold(5, shuffle=True, random_state=meta_seed).split(ranks, y):
        best = grid[int(np.argmax([roc_auc_score(y[fit_rows], ranks[fit_rows] @ w) for w in grid]))]
        blend_oof[valid_rows] = ranks[valid_rows] @ best
    return blend_oof


STACKER_LIGHTGBM_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
                               bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0)


def train_lightgbm_stacker(X_fit, y_fit):
    """2段目の浅いLightGBM。木の本数は学習側の1割で決める(採点するfoldの答えは使わない)。"""
    X_train, X_stop, y_train, y_stop = train_test_split(X_fit, y_fit, test_size=0.1, stratify=y_fit, random_state=0)
    return lgb.train(STACKER_LIGHTGBM_PARAMS, lgb.Dataset(X_train, y_train), 5000, valid_sets=[lgb.Dataset(X_stop, y_stop)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def cross_validate_final_stack(logits, y, C=0.003, lightgbm_weight=0.2, meta_seed=7):
    """最終の2段目: ロジスティック回帰8割+浅いLightGBM2割(それぞれ平均0・標準偏差1にそろえて足す)を5分割で評価する。"""
    logistic_oof, lightgbm_oof, best_iterations = np.zeros(len(y)), np.zeros(len(y)), []
    for fit_rows, valid_rows in StratifiedKFold(5, shuffle=True, random_state=meta_seed).split(logits, y):
        logistic_oof[valid_rows] = LogisticRegression(C=C, max_iter=2000).fit(logits[fit_rows], y[fit_rows]).decision_function(logits[valid_rows])
        stacker = train_lightgbm_stacker(logits[fit_rows], y[fit_rows])
        lightgbm_oof[valid_rows] = stacker.predict(logits[valid_rows], raw_score=True)
        best_iterations.append(stacker.best_iteration)
    blended = (1 - lightgbm_weight) * standardize(logistic_oof) + lightgbm_weight * standardize(lightgbm_oof)
    return {"logistic": logistic_oof, "lightgbm": lightgbm_oof, "blended": blended, "best_iterations": best_iterations}


def fit_final_stack_and_predict(oof_logits, y, test_logits, best_iterations, C=0.003, lightgbm_weight=0.2, lightgbm_seeds=5):
    """提出用: trainの全行で2段目を学び直し、testの満足の確率を返す。LightGBMの木の本数は評価時の平均×1.1、seedを変えた平均。"""
    logistic_test = LogisticRegression(C=C, max_iter=2000).fit(oof_logits, y).decision_function(test_logits)
    n_trees = int(np.mean(best_iterations) * 1.1)
    lightgbm_test = np.mean([lgb.train({**STACKER_LIGHTGBM_PARAMS, "seed": seed}, lgb.Dataset(oof_logits, y), n_trees).predict(test_logits, raw_score=True)
                             for seed in range(lightgbm_seeds)], axis=0)
    score = (1 - lightgbm_weight) * standardize(logistic_test) + lightgbm_weight * standardize(lightgbm_test)
    return 1 / (1 + np.exp(-score))
