"""XGBoostをLightGBMと同じ特徴量で学習し、OOFとtestの予測を保存する。

目的: 調整後のLightGBM+RealMLPのブレンドに、3つ目のモデルとしてXGBoostを足すと伸びるかを、特徴量選びに時間をかける前に確かめる。
ひとことで言うと: LightGBMと同じ36列をXGBoostに学習させ、LightGBM・RealMLPと違う予測になるかを見る。
やり方:
  - 特徴量はfeature_definitions.pyのADOPTED_FEATURES(36列)とTEをそのまま使う(定義・TEの計算はfeature_definitions.pyと同一)。
  - ハイパラは、--tunedを付けると6のハイパラ調整で採用した設定(TUNED_PARAMS)、付けなければXGBoostの既定(max_depth 6等)。ただし学習率だけ既定の0.3を0.05にし、木の本数は検証foldでの早期停止(100本)で
    決める(LightGBMの判定と同じ条件。0.3だと1本ごとの変化が大きく、判定がぶれやすいため)。
  - fold分割はseed42の5fold(--cv-seedで変更可)。OOFをdata/oof_xgb_<save-name>_cv<seed>.npyに保存し、ブレンドの診断は別に行う。
  - --predictで各foldのモデルでtestを予測して平均し、data/test_xgb_<save-name>_cv<seed>.npyに保存する(アンサンブル用、train_lightgbm.pyと同じやり方)。
なぜ: GBDT同士は予測が似やすく(前コンペでGBDT3種の相関0.998以上)、単体が良くてもブレンドが伸びないことがある。
"""
import argparse
import sys
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 特徴量の定義を読み込む(フォルダ名が数字で始まりパッケージにできないため、そのフォルダを検索パスに足す)
sys.path.insert(0, "src/3_lightgbm_baseline_and_feature_engineering")
import feature_definitions as features  # noqa: E402
import rejected_feature_candidates as rejected  # noqa: E402

CANDIDATE_BUILDERS = {**features.FEATURE_BUILDERS, **rejected.REJECTED_FEATURE_BUILDERS}  # --featureで試せる候補

ap = argparse.ArgumentParser()
ap.add_argument("--cv-seed", type=int, default=42)
ap.add_argument("--save-name", default="base")
ap.add_argument("--params", default=None, help="既定の一部を置き換える(JSON)")
ap.add_argument("--adopted", default=None, help="使う特徴量をカンマ区切りで置き換える(空文字なら元の特徴量20個だけ。特徴量の違いで多様性が増えるかの確認用)")
ap.add_argument("--predict", action="store_true", help="testも予測して保存する")
ap.add_argument("--tuned", action="store_true", help="6のハイパラ調整(tune_xgboost.py)で採用した設定(TUNED_PARAMS)で学習する")
ap.add_argument("--feature", default=None, help="ADOPTED_FEATURESに1つ足す特徴量(feature_definitions.pyのFEATURE_BUILDERSの名前。XGBoost向けの特徴量選び)")
args = ap.parse_args()
adopted = features.ADOPTED_FEATURES if args.adopted is None else [a for a in args.adopted.split(",") if a]
if args.feature:
    adopted = adopted + [args.feature]

# データの置き場所・列名・foldの分け方(リポジトリのルートで実行する前提)
config = {"train_csv": "data/train.csv", "test_csv": "data/test.csv", "id_col": "id", "target_col": "satisfaction",
          "cv": {"n_splits": 5, "seed": 42}}
train = pd.read_csv(config["train_csv"])
y = train[config["target_col"]].astype(int).values
X = pd.concat([train[features.BASE_FEATURES]] + [CANDIDATE_BUILDERS[a](train) for a in adopted], axis=1)
for c in features.CATEGORICAL_COLUMNS:
    X[c] = X[c].astype("category")
print(f"{X.shape[1]}列", flush=True)
if args.predict:
    test = pd.read_csv(config["test_csv"])
    X_test = pd.concat([test[features.BASE_FEATURES]] + [CANDIDATE_BUILDERS[a](test) for a in adopted], axis=1)
    for c in features.CATEGORICAL_COLUMNS:
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)  # trainと同じカテゴリ番号にそろえる
    test_pred = np.zeros(len(X_test))

xgb_params = dict(learning_rate=0.05, n_estimators=10000, early_stopping_rounds=100, tree_method="hist", enable_categorical=True,
                  eval_metric="auc", random_state=42, n_jobs=-1)
# 6のハイパラ調整(tune_xgboost.py)で採用した設定: Optunaの試行35の値
TUNED_PARAMS = {"max_depth": 10, "min_child_weight": 23.27750908295225, "reg_lambda": 17.154422085714742, "reg_alpha": 0.10595012492749806,
                "gamma": 0.09813717336201243, "max_bin": 1023, "subsample": 0.9653056625598389, "colsample_bytree": 0.4517043587579834}
if args.tuned:
    xgb_params.update(TUNED_PARAMS)
if args.params:
    xgb_params.update(json.loads(args.params))
oof = np.zeros(len(y)); train_aucs, valid_aucs, best_iterations = [], [], []; start_time = time.time()
for k, (train_rows, valid_rows) in enumerate(StratifiedKFold(5, shuffle=True, random_state=args.cv_seed).split(X, y)):
    X_fit, encoded_others = features.target_encode_in_folds(X.iloc[train_rows], y[train_rows], [X.iloc[valid_rows]] + ([X_test] if args.predict else []), seed=k)  # feature_definitions.pyと同じTE
    X_valid = encoded_others[0]
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_fit, y[train_rows], eval_set=[(X_valid, y[valid_rows])], verbose=False)
    oof[valid_rows] = model.predict_proba(X_valid)[:, 1]
    if args.predict:
        test_pred += model.predict_proba(encoded_others[1])[:, 1] / 5
    train_aucs.append(roc_auc_score(y[train_rows], model.predict_proba(X_fit)[:, 1])); valid_aucs.append(roc_auc_score(y[valid_rows], oof[valid_rows]))
    best_iterations.append(int(model.best_iteration))
    print(f"fold{k}: tr {train_aucs[-1]:.5f} va {valid_aucs[-1]:.5f} iter {best_iterations[-1]} {time.time() - start_time:.0f}s", flush=True)
tag = f"{args.save_name}_cv{args.cv_seed}"
result = {"tag": tag, "params": xgb_params, "tr": float(np.mean(train_aucs)), "oof": float(roc_auc_score(y, oof)),
          "va_folds": [float(v) for v in valid_aucs], "iters": best_iterations, "sec": round(time.time() - start_time)}
print(json.dumps(result, ensure_ascii=False))
np.save(f"data/oof_xgb_{tag}.npy", oof)
if args.predict:
    np.save(f"data/test_xgb_{tag}.npy", test_pred)
with open("data/xgb_results.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
