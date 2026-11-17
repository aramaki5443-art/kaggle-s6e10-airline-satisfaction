"""LightGBMの学習と、特徴量の採否の判定: 今の土台に特徴量の候補を1種類だけ足して、OOF AUCの変化を測る。

目的: 特徴量エンジニアリングの施策を1つずつ、同じ条件でベースラインと比べる。最終のLightGBM(--predict)もこのスクリプト。
ひとことで言うと: 「いつもの列」+「試したい列」で学習し直し、ベースラインと同じ行どうしで成績を比べる。
やり方:
  - 特徴量の定義(BASE・FE・ADOPTED・TE)は src/3_lgb_baseline_fe/features.py から読み込む。
  - ベースラインは生の20列(lgb_forward_add.py --n 20と同一)+採用済みのFE(ADOPTED)。`--fe 名前`でFEに登録した列を足す。
    比較相手のOOFは、最後に採用したFEを足したときのOOF(BASE_OOF)。FEを採用したらfeatures.pyのADOPTEDとここのBASE_OOFを更新する。
  - 比較は対応のあるブートストラップ(同じ行を復元抽出して、2つのOOFのAUC差を100回計算)で、差の揺れも出す。
    feature_fractionは既定の1.0なので、列を足しても既存の木の列の選ばれ方は変わらない(公平比較)。
なぜ: 1施策1要素の原則。足した列以外の条件をそろえれば、差はその列の効果とみなせる。
  - `--orig`: 元データ(人工データの元になった実在の満足度データ、約13万行、data/original/)を各foldの学習用の行にだけ足す。
    検証用の行はコンペのデータのままなので、OOFは今までと同じ物差しで比べられる。元データの行もTEの満足率の計算に入る。
  - `--shap N`: fold0だけ学習し、検証foldからN行を抜き出してSHAP値(各列が予測をどれだけ押し上げ/押し下げたか)を
    LightGBM内蔵のTreeSHAP(pred_contrib)で計算し、data/shap_lgb_<tag>.npzに保存して終わる(図はshap_report.py)。
実行: python src/3_lgb_baseline_fe/lgb_check.py --fe zero_flags
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# 特徴量の定義を読み込む(フォルダ名が数字で始まりパッケージにできないため、フォルダを検索パスに足す)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402
from features import *  # noqa: E402,F401,F403  BASE・CAT・FE・ADOPTED・TE・te_fold・PREPROC・DROP・noneなど

BASE_OOF = "night_rsd_wifi_cv42"  # 比較相手 = data/oof_lgb_fe_<これ>.npy(最後に採用したFEを--cv-seed 42で足したときのOOF。3seedで比べるときはseed0・1も同名_cv0・_cv1)


def fast_auc(yy, p):
    r = rankdata(p)
    npos = yy.sum()
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(yy) - npos))


ap = argparse.ArgumentParser()
ap.add_argument("--fe", required=True, choices=FE)
ap.add_argument("--adopted", default=None, help="ADOPTEDを一時的に置き換える(カンマ区切り。切り分け実験用)")
ap.add_argument("--base-oof", default=None, help="BASE_OOFを一時的に置き換える")
ap.add_argument("--n-trees", type=int, default=0, help="0より大きいと早期停止を使わず木の本数をこの値に固定する")
ap.add_argument("--es-holdout", type=float, default=0.0, help="0より大きいと、早期停止を検証foldではなく学習側のこの割合で行う(例0.05。OOFが甘くならない)")
ap.add_argument("--cv-seed", type=int, default=None, help="fold分割のseedを変える(結果が1seedの偶然でないかの再確認用)。OOFは<fe>_cv<seed>の名前で保存")
ap.add_argument("--predict", action="store_true", help="各foldのモデルでtestを予測して平均し data/submission.csv を作る(提出用)")
ap.add_argument("--orig", action="store_true", help="元データの行を各foldの学習用の行に足す")
ap.add_argument("--params", default=None, help="PARAMSの一部を一時的に置き換える(JSON、例 {\"num_leaves\": 63}。ハイパラを1つずつ調べる用。num_leavesは31のまま)")
ap.add_argument("--shap", type=int, default=0, help="fold0だけ学習し、検証foldのこの行数でSHAP値を保存して終わる")
ap.add_argument("--export-fe", action="store_true", help="学習せず、特徴量をdata/fe36_train.csv・fe36_test.csvに書き出して終わる(catboost_kaggle.pyの入力。TEの列はキーのまま)")
ap.add_argument("--tag", default=None, help="OOF・test予測の保存名を指定する(提出用の構成が既存の実験のファイルを上書きしないため)")
args = ap.parse_args()
if args.adopted is not None:
    ADOPTED = [a for a in args.adopted.split(",") if a]
DROPS = [c for a in ADOPTED + [args.fe] for c in DROP.get(a, [])]  # 採用済みの「外す」候補も外す
if args.base_oof is not None:
    BASE_OOF = args.base_oof
if args.params:
    PARAMS = {**PARAMS, **json.loads(args.params)}
    F.PARAMS = PARAMS
    print("PARAMS", PARAMS)

# データの置き場所・列名・foldの分け方(リポジトリのルートで実行する前提)
cfg = {"train_csv": "data/train.csv", "test_csv": "data/test.csv", "id_col": "id", "target_col": "satisfaction",
       "cv": {"n_splits": 5, "seed": 42}}
train = pd.read_csv(cfg["train_csv"])
prep = PREPROC.get(args.fe, lambda d: d)
train = prep(train)
y = train[cfg["target_col"]].astype(int).values
base_cols = pd.concat([train[BASE]] + [FE[a](train) for a in ADOPTED], axis=1)
X = pd.concat([base_cols, FE[args.fe](train)], axis=1).drop(columns=DROPS)
for c in CAT:
    if c in X:
        X[c] = X[c].astype("category")
if args.orig:
    o = pd.concat([pd.read_csv(f"data/original/{f}.csv") for f in ("train", "test")], ignore_index=True)
    o = prep(o)
    yo = (o["satisfaction"] == "satisfied").astype(int).values  # 元データは文字列のラベル
    Xo = pd.concat([o[BASE]] + [FE[a](o) for a in ADOPTED] + [FE[args.fe](o)], axis=1).drop(columns=DROPS)
    for c in X.columns[X.dtypes == "category"]:
        Xo[c] = pd.Categorical(Xo[c], categories=X[c].cat.categories)
    print(f"元データ {len(Xo)}行(満足率{yo.mean():.3f})を学習用に追加")
if args.predict:
    test = prep(pd.read_csv(cfg["test_csv"]))
    Xt = pd.concat([test[BASE]] + [FE[a](test) for a in ADOPTED] + [FE[args.fe](test)], axis=1).drop(columns=DROPS)
    for c in X.columns[X.dtypes == "category"]:
        Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)  # trainと同じカテゴリ番号にそろえる
    pred = np.zeros(len(Xt))
if args.export_fe:
    test = prep(pd.read_csv(cfg["test_csv"]))
    Xt = pd.concat([test[BASE]] + [FE[a](test) for a in ADOPTED] + [FE[args.fe](test)], axis=1).drop(columns=DROPS)
    pd.concat([train[[cfg["id_col"]]], X, pd.Series(y, index=X.index, name=cfg["target_col"])], axis=1).to_csv("data/fe36_train.csv", index=False)
    pd.concat([test[[cfg["id_col"]]], Xt], axis=1).to_csv("data/fe36_test.csv", index=False)
    print(f"data/fe36_train.csv・fe36_test.csvに{X.shape[1]}列を書き出した")
    raise SystemExit
print(f"{args.fe}: 外した列{DROPS} +{X.shape[1] - base_cols.shape[1]}列 {list(X.columns[base_cols.shape[1]:])}")

oof = np.zeros(len(X))
tr_aucs, va_aucs, iters = [], [], []
cv_seed = cfg["cv"]["seed"] if args.cv_seed is None else args.cv_seed
tag = args.fe if args.cv_seed is None else f"{args.fe}_cv{args.cv_seed}"  # 既定seedのOOFを上書きしないため名前を分ける
if args.orig:
    tag = "orig_" + tag
if args.tag:
    tag = args.tag
skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cv_seed)
t0 = time.time()
for k, (tr, va) in enumerate(skf.split(X, y)):
    Xfit, yfit = (pd.concat([X.iloc[tr], Xo], ignore_index=True), np.r_[y[tr], yo]) if args.orig else (X.iloc[tr], y[tr])
    Xtr, (Xva, Xte) = te_fold(Xfit, yfit, [X.iloc[va], Xt if args.predict else X.iloc[:1]], seed=k)
    m = lgb.LGBMClassifier(**PARAMS)
    if args.n_trees:
        # 早期停止を使わず木の本数を固定する(検証foldの答えを見ず、学習側の行も減らさない)
        m.set_params(n_estimators=args.n_trees)
        m.fit(Xtr, yfit)
    elif args.es_holdout:
        # 木の本数を学習側の一部(層化抽出)で決める。検証foldは採点だけに使い、答えを見て本数を選ばない
        es = np.zeros(len(yfit), bool)
        es[next(StratifiedShuffleSplit(1, test_size=args.es_holdout, random_state=k).split(Xtr, yfit))[1]] = True
        m.fit(Xtr[~es], yfit[~es], eval_X=(Xtr[es],), eval_y=(yfit[es],), eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(Xtr, yfit, eval_X=(Xva,), eval_y=(y[va],), eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[va] = m.predict_proba(Xva)[:, 1]
    if args.shap:
        sm = Xva.sample(min(args.shap, len(Xva)), random_state=0)
        ct = m.booster_.predict(sm, pred_contrib=True)  # 列ごとの寄与(logitの単位)、最後の列は全行共通の基準値
        xs = sm.copy()
        for c in xs.columns[xs.dtypes == "category"]:
            xs[c] = xs[c].cat.codes  # 図で色を付けるため数値にする
        np.savez(f"data/shap_lgb_{tag}.npz", shap=ct[:, :-1], base=ct[:, -1], X=xs.values.astype(float), cols=np.array(X.columns),
                 y=y[va][Xva.index.get_indexer(sm.index)], gain=m.booster_.feature_importance("gain"),
                 split=m.booster_.feature_importance("split"), va_auc=roc_auc_score(y[va], oof[va]), iters=m.best_iteration_)
        m.booster_.save_model(f"data/shap_lgb_{tag}.txt")  # 列の組み合わせ効果(交互作用)を後で計算するため
        sm.to_pickle(f"data/shap_lgb_{tag}_X.pkl")
        print(f"SHAP保存 data/shap_lgb_{tag}.npz ({len(sm)}行、fold0 va {roc_auc_score(y[va], oof[va]):.5f}、木{m.best_iteration_}本)")
        raise SystemExit
    if args.predict:
        pred += m.predict_proba(Xte)[:, 1] / skf.get_n_splits()
    tr_aucs.append(roc_auc_score(yfit, m.predict_proba(Xtr)[:, 1]))
    va_aucs.append(roc_auc_score(y[va], oof[va]))
    iters.append(args.n_trees or m.best_iteration_)
    # 新しい列が分岐に何回使われたか(0回なら情報が既存の列と完全に重なっている)
    imp = dict(zip(X.columns, m.booster_.feature_importance("split")))
    print(f"fold{k}: tr {tr_aucs[-1]:.5f} va {va_aucs[-1]:.5f} iter {iters[-1]} "
          f"新列の分岐回数 {sum(imp[c] for c in X.columns[base_cols.shape[1]:])}")

if BASE_OOF:
    base = np.load(f"data/oof_lgb_fe_{BASE_OOF}.npy")
    base_res = [json.loads(l) for l in open("data/lgb_fe_results.jsonl", encoding="utf-8")
                if json.loads(l)["fe"] == BASE_OOF][-1]
else:
    base = np.load("data/oof_lgb_forward_n20.npy")
    base_res = next(json.loads(l) for l in open("data/lgb_forward_results.jsonl", encoding="utf-8")
                    if json.loads(l)["n"] == 20)
rng = np.random.default_rng(0)
diffs = []
for _ in range(100):
    i = rng.integers(0, len(y), len(y))
    diffs.append(fast_auc(y[i], oof[i]) - fast_auc(y[i], base[i]))
diffs = np.array(diffs)
res = {"fe": tag, "params": json.loads(args.params) if args.params else {}, "adopted": ADOPTED, "base": BASE_OOF, "tr": float(np.mean(tr_aucs)), "oof": float(roc_auc_score(y, oof)),
       "base_oof": float(roc_auc_score(y, base)), "va_folds": [float(v) for v in va_aucs],
       "folds_improved": int(sum(a > b for a, b in zip(va_aucs, base_res["va_folds"]))),
       "boot_sd": float(diffs.std()), "boot_p_le0": float((diffs <= 0).mean()),
       "iters": [int(i) for i in iters], "sec": round(time.time() - t0)}
print(json.dumps(res, ensure_ascii=False))
np.save(f"data/oof_lgb_fe_{tag}.npy", oof)
if args.predict:
    pd.DataFrame({cfg["id_col"]: test[cfg["id_col"]], cfg["target_col"]: pred}).to_csv("data/submission.csv", index=False)
    np.save(f"data/test_lgb_fe_{tag}.npy", pred)
with open("data/lgb_fe_results.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
