"""LightGBMの学習と、特徴量の採否の判定: 今の土台に特徴量の候補を1種類だけ足して、OOF AUCの変化を測る。

目的: 特徴量エンジニアリングの施策を1つずつ、同じ条件でベースラインと比べる。最終のLightGBM(--predict)もこのスクリプト。
ひとことで言うと: 今の土台の特徴量に候補を1つ足して学習し直し、土台だけのときと同じ行どうしでOOF AUCを比べる。
やり方:
  - 特徴量の定義(BASE_FEATURES・FEATURE_BUILDERS・ADOPTED_FEATURES・TARGET_ENCODING_SPECS)は、不採用の候補(rejected_feature_candidates.py)と合わせて src/3_lightgbm_baseline_and_feature_engineering/feature_definitions.py から読み込む。
  - ベースラインは元の特徴量20個(select_baseline_features.py --n-features 20と同一)+採用済みの特徴量(ADOPTED_FEATURES)。`--feature 名前`でFEATURE_BUILDERSに登録した列を足す。
    比較相手のOOFは、最後に採用した特徴量を足したときのOOF(COMPARISON_OOF)。採用したらfeature_definitions.pyのADOPTED_FEATURESとここのCOMPARISON_OOFを更新する。
  - 比較は対応のあるブートストラップ(同じ行を復元抽出して、2つのOOFのAUC差を100回計算)で、差の揺れも出す。
    feature_fractionは既定の1.0なので、列を足しても既存の木の列の選ばれ方は変わらない(公平比較)。
なぜ: 1施策1要素の原則。足した列以外の条件をそろえれば、差はその列の効果とみなせる。
  - `--add-original-data`: 元データ(人工データの元になった実在の満足度データ、約13万行、data/original/)を各foldの学習用の行にだけ足す。
    検証用の行はコンペのデータのままなので、OOFは今までと同じ物差しで比べられる。元データの行もTEの満足率の計算に入る。
  - `--shap N`: fold0だけ学習し、検証foldからN行を抜き出してSHAP値(各列が予測をどれだけ押し上げ/押し下げたか)を
    LightGBM内蔵のTreeSHAP(pred_contrib)で計算し、data/shap_lgb_<save-name>.npzに保存して終わる(図はanalyze_shap.py)。
実行: python src/3_lightgbm_baseline_and_feature_engineering/train_lightgbm.py --feature zero_flags
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
import feature_definitions as features  # noqa: E402
import rejected_feature_candidates as rejected  # noqa: E402

# 判定できる候補 = 採用した特徴量 + 試したが採用しなかった候補(判定をやり直せるように両方を読む)
CANDIDATE_BUILDERS = {**features.FEATURE_BUILDERS, **rejected.REJECTED_FEATURE_BUILDERS}

COMPARISON_OOF = "night_rsd_wifi_cv42"  # 比較相手 = data/oof_lgb_fe_<これ>.npy(最後に採用した特徴量を--cv-seed 42で足したときのOOF。3seedで比べるときはseed0・1も同名_cv0・_cv1)


def rank_auc(labels, scores):
    """AUCを順位から直接計算する(ブートストラップで100回計算するので、roc_auc_scoreより速い書き方)。"""
    ranks = rankdata(scores)
    n_pos = labels.sum()
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(labels) - n_pos))


ap = argparse.ArgumentParser()
ap.add_argument("--feature", required=True, choices=CANDIDATE_BUILDERS, help="足す特徴量の候補の名前(何も足さないときは none)")
ap.add_argument("--adopted", default=None, help="ADOPTED_FEATURESを一時的に置き換える(カンマ区切り。切り分け実験用)")
ap.add_argument("--comparison-oof", default=None, help="COMPARISON_OOFを一時的に置き換える")
ap.add_argument("--n-trees", type=int, default=0, help="0より大きいと早期停止を使わず木の本数をこの値に固定する")
ap.add_argument("--early-stop-holdout", type=float, default=0.0, help="0より大きいと、早期停止を検証foldではなく学習側のこの割合で行う(例0.05。OOFが甘くならない)")
ap.add_argument("--cv-seed", type=int, default=None, help="fold分割のseedを変える(結果が1seedの偶然でないかの再確認用)。OOFは<feature>_cv<seed>の名前で保存")
ap.add_argument("--predict", action="store_true", help="各foldのモデルでtestを予測して平均し data/submission.csv を作る(提出用)")
ap.add_argument("--add-original-data", action="store_true", help="元データの行を各foldの学習用の行に足す")
ap.add_argument("--params", default=None, help="ハイパラの一部を一時的に置き換える(JSON、例 {\"num_leaves\": 63}。ハイパラを1つずつ調べる用)")
ap.add_argument("--shap", type=int, default=0, help="fold0だけ学習し、検証foldのこの行数でSHAP値を保存して終わる")
ap.add_argument("--export-fe", action="store_true", help="学習せず、特徴量をdata/fe36_train.csv・fe36_test.csvに書き出して終わる(train_catboost.pyの入力。TEの列はキーのまま)")
ap.add_argument("--tuned", action="store_true", help="5のハイパラ調整で採用した設定(TUNED_PARAMS、学習率0.02)で学習する")
ap.add_argument("--save-name", default=None, help="OOF・test予測の保存名を指定する(提出用の構成が既存の実験のファイルを上書きしないため)")
args = ap.parse_args()
adopted = features.ADOPTED_FEATURES if args.adopted is None else [a for a in args.adopted.split(",") if a]
dropped_cols = [c for a in adopted + [args.feature] for c in rejected.DROP_CANDIDATES.get(a, [])]  # 採用済みの「外す」候補も外す
comparison_oof = COMPARISON_OOF if args.comparison_oof is None else args.comparison_oof
# 5のハイパラ調整(tune_lightgbm.py)で採用した設定: Optunaの試行14の値に、学習率を0.05→0.02に下げたもの
TUNED_PARAMS = {"num_leaves": 61, "min_child_samples": 5, "reg_lambda": 29.265955275846228, "reg_alpha": 0.3399216318523539,
                "min_split_gain": 0.34853053056853356, "max_bin": 511, "subsample": 0.9039977941161244, "subsample_freq": 1,
                "colsample_bytree": 0.5146287416154858, "learning_rate": 0.02}
lgb_params = dict(features.DEFAULT_LGB_PARAMS)
if args.tuned:
    lgb_params.update(TUNED_PARAMS)
if args.params:
    lgb_params.update(json.loads(args.params))
    print("ハイパラ", lgb_params)
features.DEFAULT_LGB_PARAMS = lgb_params  # 元データの先生役(不採用の候補orig_teacher)も同じハイパラで学習するため

# データの置き場所・列名・foldの分け方(リポジトリのルートで実行する前提)
config = {"train_csv": "data/train.csv", "test_csv": "data/test.csv", "id_col": "id", "target_col": "satisfaction",
          "cv": {"n_splits": 5, "seed": 42}}
preprocess = rejected.PREPROCESSING.get(args.feature, lambda df: df)


def build_features(df):
    """元の特徴量20個+採用済みの特徴量+今回の候補を並べ、外す候補の列を落とす。"""
    parts = [df[features.BASE_FEATURES]] + [CANDIDATE_BUILDERS[a](df) for a in adopted] + [CANDIDATE_BUILDERS[args.feature](df)]
    return pd.concat(parts, axis=1).drop(columns=dropped_cols)


train = preprocess(pd.read_csv(config["train_csv"]))
y = train[config["target_col"]].astype(int).values
base_features = pd.concat([train[features.BASE_FEATURES]] + [CANDIDATE_BUILDERS[a](train) for a in adopted], axis=1)  # 今の土台
X = pd.concat([base_features, CANDIDATE_BUILDERS[args.feature](train)], axis=1).drop(columns=dropped_cols)
for c in features.CATEGORICAL_COLUMNS:
    if c in X:
        X[c] = X[c].astype("category")
if args.add_original_data:
    orig = preprocess(pd.concat([pd.read_csv(f"data/original/{f}.csv") for f in ("train", "test")], ignore_index=True))
    y_orig = (orig["satisfaction"] == "satisfied").astype(int).values  # 元データは文字列のラベル
    X_orig = build_features(orig)
    for c in X.columns[X.dtypes == "category"]:
        X_orig[c] = pd.Categorical(X_orig[c], categories=X[c].cat.categories)
    print(f"元データ {len(X_orig)}行(満足率{y_orig.mean():.3f})を学習用に追加")
if args.predict or args.export_fe:
    test = preprocess(pd.read_csv(config["test_csv"]))
    X_test = build_features(test)
if args.export_fe:
    pd.concat([train[[config["id_col"]]], X, pd.Series(y, index=X.index, name=config["target_col"])], axis=1).to_csv("data/fe36_train.csv", index=False)
    pd.concat([test[[config["id_col"]]], X_test], axis=1).to_csv("data/fe36_test.csv", index=False)
    print(f"data/fe36_train.csv・fe36_test.csvに{X.shape[1]}列を書き出した")
    raise SystemExit
if args.predict:
    for c in X.columns[X.dtypes == "category"]:
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)  # trainと同じカテゴリ番号にそろえる
    test_pred = np.zeros(len(X_test))
new_cols = list(X.columns[base_features.shape[1]:])
print(f"{args.feature}: 外した列{dropped_cols} +{X.shape[1] - base_features.shape[1]}列 {new_cols}")

oof = np.zeros(len(X))
train_aucs, valid_aucs, best_iterations = [], [], []
cv_seed = config["cv"]["seed"] if args.cv_seed is None else args.cv_seed
tag = args.feature if args.cv_seed is None else f"{args.feature}_cv{args.cv_seed}"  # 既定seedのOOFを上書きしないため名前を分ける
if args.add_original_data:
    tag = "orig_" + tag
if args.save_name:
    tag = args.save_name
folds = StratifiedKFold(n_splits=config["cv"]["n_splits"], shuffle=True, random_state=cv_seed)
start_time = time.time()
for k, (train_rows, valid_rows) in enumerate(folds.split(X, y)):
    if args.add_original_data:
        X_fit, y_fit = pd.concat([X.iloc[train_rows], X_orig], ignore_index=True), np.r_[y[train_rows], y_orig]
    else:
        X_fit, y_fit = X.iloc[train_rows], y[train_rows]
    # TEの列を、答えを見ない満足率に置き換える(testを使わないときはダミーの1行を渡す)
    X_fit, (X_valid, X_test_fold) = features.target_encode_in_folds(
        X_fit, y_fit, [X.iloc[valid_rows], X_test if args.predict else X.iloc[:1]], seed=k)
    model = lgb.LGBMClassifier(**lgb_params)
    if args.n_trees:
        # 早期停止を使わず木の本数を固定する(検証foldの答えを見ず、学習側の行も減らさない)
        model.set_params(n_estimators=args.n_trees)
        model.fit(X_fit, y_fit)
    elif args.early_stop_holdout:
        # 木の本数を学習側の一部(層化抽出)で決める。検証foldは採点だけに使い、答えを見て本数を選ばない
        stop_rows = np.zeros(len(y_fit), bool)
        stop_rows[next(StratifiedShuffleSplit(1, test_size=args.early_stop_holdout, random_state=k).split(X_fit, y_fit))[1]] = True
        model.fit(X_fit[~stop_rows], y_fit[~stop_rows], eval_X=(X_fit[stop_rows],), eval_y=(y_fit[stop_rows],), eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X_fit, y_fit, eval_X=(X_valid,), eval_y=(y[valid_rows],), eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[valid_rows] = model.predict_proba(X_valid)[:, 1]
    if args.shap:
        shap_rows = X_valid.sample(min(args.shap, len(X_valid)), random_state=0)
        contrib = model.booster_.predict(shap_rows, pred_contrib=True)  # 列ごとの寄与(logitの単位)、最後の列は全行共通の基準値
        shap_X = shap_rows.copy()
        for c in shap_X.columns[shap_X.dtypes == "category"]:
            shap_X[c] = shap_X[c].cat.codes  # 図で色を付けるため数値にする
        np.savez(f"data/shap_lgb_{tag}.npz", shap=contrib[:, :-1], base=contrib[:, -1], X=shap_X.values.astype(float), cols=np.array(X.columns),
                 y=y[valid_rows][X_valid.index.get_indexer(shap_rows.index)], gain=model.booster_.feature_importance("gain"),
                 split=model.booster_.feature_importance("split"), va_auc=roc_auc_score(y[valid_rows], oof[valid_rows]), iters=model.best_iteration_)
        model.booster_.save_model(f"data/shap_lgb_{tag}.txt")  # 列の組み合わせ効果(交互作用)を後で計算するため
        shap_rows.to_pickle(f"data/shap_lgb_{tag}_X.pkl")
        print(f"SHAP保存 data/shap_lgb_{tag}.npz ({len(shap_rows)}行、fold0 va {roc_auc_score(y[valid_rows], oof[valid_rows]):.5f}、木{model.best_iteration_}本)")
        raise SystemExit
    if args.predict:
        test_pred += model.predict_proba(X_test_fold)[:, 1] / folds.get_n_splits()
    train_aucs.append(roc_auc_score(y_fit, model.predict_proba(X_fit)[:, 1]))
    valid_aucs.append(roc_auc_score(y[valid_rows], oof[valid_rows]))
    best_iterations.append(args.n_trees or model.best_iteration_)
    # 新しい列が分岐に何回使われたか(0回なら情報が既存の列と完全に重なっている)
    split_counts = dict(zip(X.columns, model.booster_.feature_importance("split")))
    print(f"fold{k}: tr {train_aucs[-1]:.5f} va {valid_aucs[-1]:.5f} iter {best_iterations[-1]} "
          f"新列の分岐回数 {sum(split_counts[c] for c in new_cols)}")

result = {"fe": tag, "params": json.loads(args.params) if args.params else {}, "adopted": adopted, "base": comparison_oof,
          "tr": float(np.mean(train_aucs)), "oof": float(roc_auc_score(y, oof)),
          "va_folds": [float(v) for v in valid_aucs], "iters": [int(i) for i in best_iterations], "sec": round(time.time() - start_time)}
# 比較相手(土台)のOOFがあれば、同じ行で比べる。cloneした直後など無いときは、比べずにOOFを保存だけする
base_oof_path, results_path = ((f"data/oof_lgb_fe_{comparison_oof}.npy", "data/lgb_fe_results.jsonl") if comparison_oof
                               else ("data/oof_lgb_forward_n20.npy", "data/lgb_forward_results.jsonl"))
base_result = None
if os.path.exists(base_oof_path) and os.path.exists(results_path):
    past_results = [json.loads(line) for line in open(results_path, encoding="utf-8")]
    hits = [r for r in past_results if (r.get("fe") == comparison_oof if comparison_oof else r.get("n") == 20)]
    base_result = hits[-1] if hits else None
if base_result is None:
    print(f"比較相手のOOF({base_oof_path})が無いので、比べずにOOFを保存する")
else:
    base_oof = np.load(base_oof_path)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(100):  # 対応のあるブートストラップ: 同じ行を復元抽出して、2つのOOFのAUC差を測る
        rows = rng.integers(0, len(y), len(y))
        diffs.append(rank_auc(y[rows], oof[rows]) - rank_auc(y[rows], base_oof[rows]))
    diffs = np.array(diffs)
    result.update({"base_oof": float(roc_auc_score(y, base_oof)),
                   "folds_improved": int(sum(a > b for a, b in zip(valid_aucs, base_result["va_folds"]))),
                   "boot_sd": float(diffs.std()), "boot_p_le0": float((diffs <= 0).mean())})
print(json.dumps(result, ensure_ascii=False))
np.save(f"data/oof_lgb_fe_{tag}.npy", oof)
if args.predict:
    pd.DataFrame({config["id_col"]: test[config["id_col"]], config["target_col"]: test_pred}).to_csv("data/submission.csv", index=False)
    np.save(f"data/test_lgb_fe_{tag}.npy", test_pred)
with open("data/lgb_fe_results.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
