"""CatBoostをKaggleのGPUで学習し、OOFとtestの予測を保存する。

目的: RealMLP+LightGBMのアンサンブルに、3つ目の木のモデルとしてCatBoostを足す価値があるかを調べる。
ひとことで言うと: LightGBMと同じ36列を、カテゴリ列の扱い方が違うCatBoostに学習させ、違う予測になるかを見る。
やり方:
  - 特徴量はlgb_fe_check.pyの採用済みの列(`python 2_feature_engineering/lgb_fe_check.py --fe none --export-fe`でdata/fe36_*.csvに書き出す。
    Kaggleで動かすときは、このCSVをデータセットとして読み込ませる)。TEの2列(距離・距離×Class)だけは
    キーのまま受け取り、foldごとに計算する(te_fold・te_mapはlgb_fe_check.pyと同一)。
  - Class・Type of Travel・Customer TypeはCatBoostのカテゴリ列として渡す(CatBoost独自の、答えを順番に使う集計で数値化される)。
  - 学習率0.05、木の本数は検証foldでの早期停止(100本)で決める(LightGBM・XGBoostの判定と同じ条件)。ほかは既定(depth 6等)。
  - RUNSの各行(タグ, fold分割のseed, 生の20列だけか, 上書きするハイパラ, 距離の扱い)を順に学習し、oof_cat_<タグ>.npy・test_cat_<タグ>.npyを保存。
  - 距離の扱い: None=数値のまま。"add"=数値の列は残し、同じ値の文字列コピーdist_catをカテゴリ列として足す。
    "rep"=Flight Distanceの数値列そのものをカテゴリ列に置き換える。このコンペで一番効いたのは距離を路線IDとして扱うこと
    (TEの2列)なので、CatBoost独自のカテゴリ処理(答えを行の順番に使う集計+カテゴリ同士の組み合わせ)で距離を扱えば、
    LightGBMのTEとは違う予測になる可能性がある。
    "allcat"=生の20列(数値・評価・カテゴリ)を全て文字列にしてカテゴリ列に置き換える(作り込み16列・TEはそのまま)。
    公開ノートブックで全列カテゴリ化+自前TEなしが単体+0.00015。まず「全列カテゴリ化」だけを切り分ける。
    "allcat_note"=allcatに加えてTEの2列を外す(allcatとの差がTEを外した効果)。
  - ローカル実行: /kaggle/inputが無ければdata/fe36_*.csvを読み、CPU(8スレッド)で学習してdata/に保存する。
    GPUとCPUではCatBoostの計算(境界の作り方等)が違うので、比較の基準もCPUで学習し直す。
なぜ: 木のモデル同士は予測が似やすい(XGBoostは「もう1本のLightGBM」程度)。CatBoostはカテゴリ列の扱いと木の形
  (左右対称の木)が違うため、XGBoostより違う予測になる可能性がある。
"""
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
T21 = dict(depth=7, l2_leaf_reg=24.86317046031032, border_count=254, random_strength=0.17142049132377946,
           bagging_temperature=0.4171977018905056, max_ctr_complexity=4)
RUNS = [("allcat_t21_cv42", 42, False, T21, "allcat"), ("allcat_t21_cv0", 0, False, T21, "allcat"),
        ("allcat_t21_cv1", 1, False, T21, "allcat"), ("allcat_cv0", 0, False, {}, "allcat"), ("allcat_cv1", 1, False, {}, "allcat")]
CAT = ["Customer Type", "Type of Travel", "Class"]
TE = {"te_distance_m3": ("Flight Distance", 3), "te_dist_class": (["Flight Distance", "Class"], 3)}
RAW_N = 20  # 先頭20列が生の列(lgb_fe_check.pyのBASE)

hit = glob.glob("/kaggle/input/**/fe36_train.csv", recursive=True)
KAGGLE = bool(hit)
d, OUT = (os.path.dirname(hit[0]), ".") if KAGGLE else ("data", "data")
train = pd.read_csv(f"{d}/fe36_train.csv"); test = pd.read_csv(f"{d}/fe36_test.csv")
y = train.pop("satisfaction").values
ids = test["id"].values
train, test = train.drop(columns="id"), test.drop(columns="id")


def te_map(key_fit, y_fit, key_apply, m):  # lgb_fe_check.pyと同一
    p = y_fit.mean()
    g = pd.DataFrame({"k": key_fit, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
    return pd.Series(key_apply).map((g["sum"] + m * p) / (g["count"] + m)).fillna(p).values


def te_fold(Xtr, ytr, others, seed):  # lgb_fe_check.pyと同一
    Xtr = Xtr.copy(); others = [o.copy() for o in others]
    for name, (_, m) in TE.items():
        if name not in Xtr:
            continue
        key = Xtr[name].values; v = np.zeros(len(Xtr))
        for itr, iva in StratifiedKFold(5, shuffle=True, random_state=seed).split(key, ytr):
            v[iva] = te_map(key[itr], ytr[itr], key[iva], m)
        for o in others:
            o[name] = te_map(key, ytr, o[name].values, m)
        Xtr[name] = v
    return Xtr, others


summary = []
for tag, cv_seed, raw_only, over, dist in RUNS:
    X, Xt = (train.iloc[:, :RAW_N], test.iloc[:, :RAW_N]) if raw_only else (train, test)
    cats = [c for c in CAT if c in X]
    if dist == "allcat_note":  # 全列カテゴリ化に加え、自前のTE2列を外してCatBoost独自の集計だけに任せる
        X, Xt = X.drop(columns=list(TE)), Xt.drop(columns=list(TE))
    if dist in ("allcat", "allcat_note"):  # 生の20列を全てカテゴリ列に(欠損は"-1"の文字列。pandasのstr型は欠損をNaNのまま残すため先に埋める)
        raw = list(X.columns[:RAW_N])
        X = X.assign(**{c: X[c].fillna(-1).astype(str) for c in raw})
        Xt = Xt.assign(**{c: Xt[c].fillna(-1).astype(str) for c in raw})
        cats = raw
    elif dist:  # 距離をカテゴリ列にする(CatBoostのカテゴリ列は文字列か整数で渡す)
        col = "dist_cat" if dist == "add" else "Flight Distance"
        X = X.assign(**{col: X["Flight Distance"].astype(str)})
        Xt = Xt.assign(**{col: Xt["Flight Distance"].astype(str)})
        cats.append(col)
    P = dict(iterations=20000, learning_rate=0.05, eval_metric="AUC", random_seed=42,
             **({"task_type": "GPU"} if KAGGLE else {"task_type": "CPU", "thread_count": 8, "allow_writing_files": False}),
             od_type="Iter", od_wait=100, use_best_model=True, verbose=0)
    P.update(over)
    oof = np.zeros(len(y)); pred = np.zeros(len(Xt)); aucs, iters = [], []; t0 = time.time()
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=cv_seed).split(X, y)):
        Xtr, (Xva, Xte) = te_fold(X.iloc[tr], y[tr], [X.iloc[va], Xt], seed=k)
        m = CatBoostClassifier(**P, cat_features=cats)
        m.fit(Xtr, y[tr], eval_set=(Xva, y[va]))
        oof[va] = m.predict_proba(Xva)[:, 1]; pred += m.predict_proba(Xte)[:, 1] / 5
        aucs.append(roc_auc_score(y[va], oof[va])); iters.append(int(m.get_best_iteration()))
        print(f"{tag} fold{k}: va {aucs[-1]:.5f} iter {iters[-1]} {time.time() - t0:.0f}s", flush=True)
    res = {"tag": tag, "cv_seed": cv_seed, "raw_only": raw_only, "over": over, "dist": dist, "oof": float(roc_auc_score(y, oof)),
           "va_folds": aucs, "iters": iters, "sec": round(time.time() - t0)}
    print(json.dumps(res), flush=True); summary.append(res)
    np.save(f"{OUT}/oof_cat_{tag}.npy", oof); np.save(f"{OUT}/test_cat_{tag}.npy", pred)
    if not KAGGLE:  # ローカルは実行ごとに追記
        open(f"{OUT}/cat_results.jsonl", "a").write(json.dumps(res) + "\n")
if KAGGLE:
    json.dump(summary, open("cat_results.json", "w"))
    pd.DataFrame({"id": ids}).to_csv("test_ids.csv", index=False)
