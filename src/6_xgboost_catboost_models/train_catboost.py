"""CatBoostをKaggleのGPUで学習し、OOFとtestの予測を保存する。

目的: RealMLP+LightGBM(+XGBoost)のアンサンブルに、木のモデルとしてCatBoostを足す価値があるかを調べる。
ひとことで言うと: LightGBMと同じ36列を、カテゴリ列の扱い方が違うCatBoostに学習させ、違う予測になるかを見る。
やり方:
  - 特徴量はfeature_definitions.pyの採用済みの列(`python src/2_lightgbm_baseline_and_feature_engineering/train_lightgbm.py --feature none --export-fe`でdata/fe36_*.csvに書き出す。
    Kaggleで動かすときは、このCSVをデータセットとして読み込ませる)。TEの2列(距離・距離×Class)だけは
    キーのまま受け取り、foldごとに計算する(target_encode_in_folds・target_encodeはfeature_definitions.pyと同一)。
  - Class・Type of Travel・Customer TypeはCatBoostのカテゴリ列として渡す(CatBoost独自の、答えを順番に使う集計で数値化される)。
  - 学習率0.05、木の本数は検証foldでの早期停止(100本)で決める(LightGBM・XGBoostの判定と同じ条件)。ほかは既定(depth 6等)。
  - RUNSの各行(タグ, fold分割のseed, 元の特徴量20個だけか, 上書きするハイパラ, 距離の扱い)を順に学習し、oof_cat_<タグ>.npy・test_cat_<タグ>.npyを保存。--run タグ で1行だけ学習できる。
  - 距離の扱い: None=数値のまま。"add"=数値の列は残し、同じ値の文字列コピーdist_catをカテゴリ列として足す。
    "rep"=Flight Distanceの数値列そのものをカテゴリ列に置き換える。このコンペで一番効いたのは距離を路線IDとして扱うこと
    (TEの2列)なので、CatBoost独自のカテゴリ処理(答えを行の順番に使う集計+カテゴリ同士の組み合わせ)で距離を扱えば、
    LightGBMのTEとは違う予測になる可能性がある。
    "allcat"=元の特徴量20個(数値・評価・カテゴリ)を全て文字列にしてカテゴリ列に置き換える(作り込み16列・TEはそのまま)。
    公開ノートブックで全列カテゴリ化+自前TEなしが単体+0.00015。まず「全列カテゴリ化」だけを切り分ける。
    "allcat_note"=allcatに加えてTEの2列を外す(allcatとの差がTEを外した効果)。
  - ローカル実行: /kaggle/inputが無ければdata/fe36_*.csvを読み、CPU(8スレッド)で学習してdata/に保存する。
    GPUとCPUではCatBoostの計算(境界の作り方等)が違うので、比較の基準もCPUで学習し直す。
なぜ: 木のモデル同士は予測が似やすい(XGBoostは「もう1本のLightGBM」程度)。CatBoostはカテゴリ列の扱いと木の形
  (左右対称の木)が違うため、XGBoostより違う予測になる可能性がある。
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Optunaの最良(試行#21)。3seedの確認(調整前のallcatとseed0・1で比べる)と、アンサンブル用のtest予測を作る
TUNED_PARAMS = dict(depth=7, l2_leaf_reg=24.86317046031032, border_count=254, random_strength=0.17142049132377946,
                    bagging_temperature=0.4171977018905056, max_ctr_complexity=4)
RUNS = [("allcat_t21_cv42", 42, False, TUNED_PARAMS, "allcat"), ("allcat_t21_cv0", 0, False, TUNED_PARAMS, "allcat"),
        ("allcat_t21_cv1", 1, False, TUNED_PARAMS, "allcat"), ("allcat_cv0", 0, False, {}, "allcat"), ("allcat_cv1", 1, False, {}, "allcat")]
ap = argparse.ArgumentParser()
ap.add_argument("--run", help="RUNSのうち、このタグの1行だけを学習する(例: allcat_t21_cv42 = 最終提出で使った設定)")
args, _ = ap.parse_known_args()  # Kaggleのスクリプト実行で余分な引数が付いても止まらないように
if args.run:
    RUNS = [run for run in RUNS if run[0] == args.run]
    assert RUNS, f"RUNSに{args.run}がない"
CATEGORICAL_COLUMNS = ["Customer Type", "Type of Travel", "Class"]
TARGET_ENCODING_SPECS = {"te_distance_m3": ("Flight Distance", 3), "te_dist_class": (["Flight Distance", "Class"], 3)}
N_BASE_FEATURES = 20  # 先頭20列が元の特徴量(feature_definitions.pyのBASE_FEATURES)

kaggle_files = glob.glob("/kaggle/input/**/fe36_train.csv", recursive=True)
KAGGLE = bool(kaggle_files)
data_dir, out_dir = (os.path.dirname(kaggle_files[0]), ".") if KAGGLE else ("data", "data")
train = pd.read_csv(f"{data_dir}/fe36_train.csv"); test = pd.read_csv(f"{data_dir}/fe36_test.csv")
y = train.pop("satisfaction").values
ids = test["id"].values
train, test = train.drop(columns="id"), test.drop(columns="id")


def target_encode(key_fit, y_fit, key_apply, model):  # feature_definitions.pyと同一
    prior = y_fit.mean()
    stats = pd.DataFrame({"key": key_fit, "y": y_fit}).groupby("key")["y"].agg(["sum", "count"])
    return pd.Series(key_apply).map((stats["sum"] + model * prior) / (stats["count"] + model)).fillna(prior).values


def target_encode_in_folds(X_fit, y_fit, others, seed):  # feature_definitions.pyと同一
    X_fit = X_fit.copy(); others = [other.copy() for other in others]
    for name, (_, model) in TARGET_ENCODING_SPECS.items():
        if name not in X_fit:
            continue
        key = X_fit[name].values; encoded = np.zeros(len(X_fit))
        for inner_fit, inner_apply in StratifiedKFold(5, shuffle=True, random_state=seed).split(key, y_fit):
            encoded[inner_apply] = target_encode(key[inner_fit], y_fit[inner_fit], key[inner_apply], model)
        for other in others:
            other[name] = target_encode(key, y_fit, other[name].values, model)
        X_fit[name] = encoded
    return X_fit, others


summary = []
for tag, cv_seed, raw_only, param_overrides, distance_mode in RUNS:
    X, X_test = (train.iloc[:, :N_BASE_FEATURES], test.iloc[:, :N_BASE_FEATURES]) if raw_only else (train, test)
    cat_features = [c for c in CATEGORICAL_COLUMNS if c in X]
    if distance_mode == "allcat_note":  # 全列カテゴリ化に加え、自前のTE2列を外してCatBoost独自の集計だけに任せる
        X, X_test = X.drop(columns=list(TARGET_ENCODING_SPECS)), X_test.drop(columns=list(TARGET_ENCODING_SPECS))
    if distance_mode in ("allcat", "allcat_note"):  # 元の特徴量20個を全てカテゴリ列に(欠損は"-1"の文字列。pandasのstr型は欠損をNaNのまま残すため先に埋める)
        base_cols = list(X.columns[:N_BASE_FEATURES])
        X = X.assign(**{c: X[c].fillna(-1).astype(str) for c in base_cols})
        X_test = X_test.assign(**{c: X_test[c].fillna(-1).astype(str) for c in base_cols})
        cat_features = base_cols
    elif distance_mode:  # 距離をカテゴリ列にする(CatBoostのカテゴリ列は文字列か整数で渡す)
        col = "dist_cat" if distance_mode == "add" else "Flight Distance"
        X = X.assign(**{col: X["Flight Distance"].astype(str)})
        X_test = X_test.assign(**{col: X_test["Flight Distance"].astype(str)})
        cat_features.append(col)
    cat_params = dict(iterations=20000, learning_rate=0.05, eval_metric="AUC", random_seed=42,
                      **({"task_type": "GPU"} if KAGGLE else {"task_type": "CPU", "thread_count": 8, "allow_writing_files": False}),
                      od_type="Iter", od_wait=100, use_best_model=True, verbose=0)
    cat_params.update(param_overrides)
    oof = np.zeros(len(y)); test_pred = np.zeros(len(X_test)); aucs, best_iterations = [], []; start_time = time.time()
    for k, (train_rows, valid_rows) in enumerate(StratifiedKFold(5, shuffle=True, random_state=cv_seed).split(X, y)):
        X_fit, (X_valid, X_test_fold) = target_encode_in_folds(X.iloc[train_rows], y[train_rows], [X.iloc[valid_rows], X_test], seed=k)
        model = CatBoostClassifier(**cat_params, cat_features=cat_features)
        model.fit(X_fit, y[train_rows], eval_set=(X_valid, y[valid_rows]))
        oof[valid_rows] = model.predict_proba(X_valid)[:, 1]; test_pred += model.predict_proba(X_test_fold)[:, 1] / 5
        aucs.append(roc_auc_score(y[valid_rows], oof[valid_rows])); best_iterations.append(int(model.get_best_iteration()))
        print(f"{tag} fold{k}: va {aucs[-1]:.5f} iter {best_iterations[-1]} {time.time() - start_time:.0f}s", flush=True)
    result = {"tag": tag, "cv_seed": cv_seed, "raw_only": raw_only, "over": param_overrides, "dist": distance_mode, "oof": float(roc_auc_score(y, oof)),
              "va_folds": aucs, "iters": best_iterations, "sec": round(time.time() - start_time)}
    print(json.dumps(result), flush=True); summary.append(result)
    np.save(f"{out_dir}/oof_cat_{tag}.npy", oof); np.save(f"{out_dir}/test_cat_{tag}.npy", test_pred)
    if not KAGGLE:  # ローカルは実行ごとに追記
        open(f"{out_dir}/cat_results.jsonl", "a").write(json.dumps(result) + "\n")
if KAGGLE:
    json.dump(summary, open("cat_results.json", "w"))
    pd.DataFrame({"id": ids}).to_csv("test_ids.csv", index=False)
