"""特徴量の定義: データの読み込み、元の特徴量の一覧、採用した13種類(16個)の特徴量の作り方、Target Encoding。

ノートブック(notebooks/)から読み込んで使う。採用した特徴量を考えた順に、仮説・確認・結果と一緒に追うには
notebooks/02_feature_engineering.ipynb を見る(最後のセルで、このファイルと同じ値になることを照合している)。

- FEATURE_BUILDERS: 特徴量の名前 → 関数(データを受け取り、足す列を返す)。
- ADOPTED_FEATURES: 採用した順の名前。モデルは元の特徴量20個(BASE_FEATURES)+ADOPTED_FEATURESで学習する。
- 値ごとの満足率(Target Encoding)は答えを使うので、FEATURE_BUILDERSでは元の値(キー)だけを置き、
  学習のfoldごとに target_encode_in_folds で満足率に置き換える。
"""
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

# データの置き場所。既定はリポジトリ直下の data/(環境変数 AIRLINE_DATA_DIR で変えられる)
DATA_DIR = Path(os.environ.get("AIRLINE_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
TARGET = "satisfaction"

ORIGINAL_FEATURES_BY_UNIVARIATE_AUC = [  # 元の特徴量21個。EDA 6節の「値ごとの満足率」の単変量AUCの高い順
    "Online boarding", "Class", "Inflight entertainment", "Inflight wifi service", "Seat comfort",
    "Flight Distance", "Type of Travel", "On-board service", "Cleanliness", "Leg room service", "Age",
    "Ease of Online booking", "Baggage handling", "Checkin service", "Food and drink", "Gate location",
    "Customer Type", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
    "Departure Delay in Minutes", "Gender",
]
BASE_FEATURES = ORIGINAL_FEATURES_BY_UNIVARIATE_AUC[:20]  # ベースラインで選んだ20個(性別を除く)
CATEGORICAL_COLUMNS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SERVICE_RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
                   "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
                   "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]  # サービス評価13項目(0〜5点)
COMFORT_RATINGS = ["Seat comfort", "Leg room service", "Inflight entertainment"]  # 機内の快適さ


def load_train_and_test():
    """train.csv・test.csvを読む。"""
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


def read_train_and_test_columns(columns):
    """train+testを縦に並べ、指定した列だけ返す(答えを使わない集計用)。"""
    return pd.concat([pd.read_csv(DATA_DIR / f"{part}.csv", usecols=columns)[columns] for part in ("train", "test")], ignore_index=True)


# ① 条件で絞る(ゲート特徴量): 点数が効く客層だけに値を持たせ、ほかの客は空欄(欠損)にする

def comfort(df):
    """機内の快適さ = 座席・足元の広さ・機内エンタメの点数の平均(0点は「評価なし」の可能性があるので除く)。"""
    return df[COMFORT_RATINGS].replace(0, np.nan).mean(axis=1)


def comfort_business_travel(df):
    """出張客だけ快適さを持ち、私用客(快適さに関係なく満足率約10%)は空欄。"""
    return pd.DataFrame({"comfort_business_travel": comfort(df).where(df["Type of Travel"] == "Business travel")})


def comfort_loyal_business_travel(df):
    """常連かつ出張客だけ快適さを持つ。常連でない客は快適さが効かない(EDA 8節)。"""
    loyal_business = (df["Type of Travel"] == "Business travel") & (df["Customer Type"] == "Loyal Customer")
    return pd.DataFrame({"comfort_loyal_business_travel": comfort(df).where(loyal_business)})


def onboard_service_business_travel(df):
    """出張客だけ機内サービスの点数を持つ(単変量AUC 出張客0.759、私用客0.509)。"""
    return pd.DataFrame({"onboard_service_business_travel": df["On-board service"].where(df["Type of Travel"] == "Business travel")})


# ② 似た乗客で数える・平均する

# Target Encoding: 特徴量の名前 → (キーにする元の列, 強さm)。mは人数が少ない値の満足率を全体の満足率にどれだけ寄せるか
TARGET_ENCODING_SPECS = {"distance_satisfaction_rate": ("Flight Distance", 3),
                         "distance_class_satisfaction_rate": (["Flight Distance", "Class"], 3)}


def target_encoding_key(name):
    """Target Encodingのキー(元の値)を置く。満足率への置き換えは学習のfoldごとに行う。"""
    def build(df):
        columns = TARGET_ENCODING_SPECS[name][0]
        if isinstance(columns, list):  # 組み合わせは値を文字列でつないだキーにする(例: "1235|Business")
            return pd.DataFrame({name: df[columns].fillna(-1).astype(str).agg("|".join, axis=1)})
        return pd.DataFrame({name: df[columns]})
    return build


def target_encode(key_fit, y_fit, key_apply, m):
    """値ごとの満足率を、人数が少ないほど全体の満足率に寄せて計算する。学習側に無い値は全体の満足率。"""
    prior = y_fit.mean()
    stats = pd.DataFrame({"key": key_fit, "y": y_fit}).groupby("key")["y"].agg(["sum", "count"])
    return pd.Series(key_apply).map((stats["sum"] + m * prior) / (stats["count"] + m)).fillna(prior).values


def target_encode_in_folds(X_fit, y_fit, others, seed):
    """学習用の行は、その中でさらに5分割して自分を含まない満足率を入れる(自分の答えが混ざるのを防ぐ)。
    検証用・testの行(others)には、学習用の行全体から計算した満足率を入れる。"""
    X_fit = X_fit.copy()
    others = [other.copy() for other in others]
    for name, (_, m) in TARGET_ENCODING_SPECS.items():
        if name not in X_fit:
            continue
        key = X_fit[name].values
        encoded = np.zeros(len(X_fit))
        for inner_fit, inner_apply in StratifiedKFold(5, shuffle=True, random_state=seed).split(key, y_fit):
            encoded[inner_apply] = target_encode(key[inner_fit], y_fit[inner_fit], key[inner_apply], m)
        for other in others:
            other[name] = target_encode(key, y_fit, other[name].values, m)
        X_fit[name] = encoded
    return X_fit, others


def distance_travel_type_count(df):
    """同じ飛行距離・同じ旅行目的の乗客が、train+testに何人いるか(答えは使わない)。"""
    columns = ["Flight Distance", "Type of Travel"]
    counts = read_train_and_test_columns(columns).fillna(-1).astype(str).agg("|".join, axis=1).value_counts()
    return pd.DataFrame({"distance_travel_type_count": df[columns].fillna(-1).astype(str).agg("|".join, axis=1).map(counts).values}, index=df.index)


def route_statistic(name, column, statistic):
    """飛行距離の値を路線とみなし、train+testで同じ距離の乗客の column の平均(mean)または標準偏差(std)を入れる。"""
    def build(df):
        train_and_test = read_train_and_test_columns(["Flight Distance", column])
        by_route = train_and_test.groupby("Flight Distance")[column].agg(statistic)
        return pd.DataFrame({name: df["Flight Distance"].map(by_route).values}, index=df.index)
    return build


GROUP_KEYS = ["Class", "Type of Travel", "Customer Type"]  # 座席クラス×旅行目的×常連かどうかの12グループ
GROUP_DEVIATION_COLUMNS = {"age": "Age", "flight_distance": "Flight Distance",
                           "departure_delay": "Departure Delay in Minutes", "arrival_delay": "Arrival Delay in Minutes"}


def group_deviation(df):
    """自分の値 − 同じ客層グループの平均(train+testで計算)を、年齢・飛行距離・遅延2つの4列で入れる。"""
    train_and_test = read_train_and_test_columns(GROUP_KEYS + list(GROUP_DEVIATION_COLUMNS.values()))
    new_columns = {}
    for short_name, column in GROUP_DEVIATION_COLUMNS.items():
        group_mean = train_and_test.groupby(GROUP_KEYS)[column].mean().rename("group_mean")
        row_mean = df[GROUP_KEYS].merge(group_mean, left_on=GROUP_KEYS, right_index=True, how="left")["group_mean"]
        new_columns[f"group_deviation_{short_name}"] = df[column].values - row_mean.values
    return pd.DataFrame(new_columns, index=df.index)


# ③ モデルで予想する: サービス評価の期待値(ほかの項目への答え方から、この人ならこの項目に何点を付けそうか)

EXPECTED_RATINGS = {"expected_online_boarding": "Online boarding", "expected_legroom": "Leg room service",
                    "expected_checkin": "Checkin service", "expected_time_convenient": "Departure/Arrival time convenient"}


def make_expected_rating(rating_column):
    """rating_column の点数を、ほかの元の特徴量21個からLightGBMで予想した期待値(train+testの全行)。
    目的変数(満足かどうか)は使わない。5分割して、予想する行自身は学習に使わない。"""
    train_and_test = read_train_and_test_columns(ORIGINAL_FEATURES_BY_UNIVARIATE_AUC)
    X = train_and_test.drop(columns=rating_column)
    for c in X.select_dtypes(exclude="number").columns:
        X[c] = X[c].astype("category")
    y = train_and_test[rating_column].astype(int).values
    expected = np.zeros(len(X))
    for k, (fit_rows, apply_rows) in enumerate(KFold(5, shuffle=True, random_state=0).split(X)):
        model = lgb.LGBMClassifier(objective="multiclass" if len(np.unique(y)) > 2 else "binary", learning_rate=0.1,
                                   n_estimators=2000, random_state=0, verbose=-1)
        stop_rows = np.random.default_rng(k).random(len(fit_rows)) < 0.1  # 木の本数を決める行(学習側の1割)
        model.fit(X.iloc[fit_rows[~stop_rows]], y[fit_rows[~stop_rows]], eval_X=(X.iloc[fit_rows[stop_rows]],),
                  eval_y=(y[fit_rows[stop_rows]],), callbacks=[lgb.early_stopping(50, verbose=False)])
        expected[apply_rows] = model.predict_proba(X.iloc[apply_rows]) @ model.classes_  # 点数の確率×点数の合計
    return expected


def expected_rating_path(name):
    return DATA_DIR / f"{name}.npy"


def expected_rating(name):
    """make_expected_rating で前もって保存した期待値(train+testの順)から、渡された行の分を切り出す。"""
    def build(df):
        expected = np.load(expected_rating_path(name))
        n_train = len(pd.read_csv(DATA_DIR / "train.csv", usecols=["id"]))
        return pd.DataFrame({name: expected[:n_train] if len(df) == n_train else expected[n_train:]}, index=df.index)
    return build


FEATURE_BUILDERS = {
    "comfort_business_travel": comfort_business_travel,
    "onboard_service_business_travel": onboard_service_business_travel,
    "distance_satisfaction_rate": target_encoding_key("distance_satisfaction_rate"),
    "distance_class_satisfaction_rate": target_encoding_key("distance_class_satisfaction_rate"),
    "distance_travel_type_count": distance_travel_type_count,
    "comfort_loyal_business_travel": comfort_loyal_business_travel,
    "route_mean_legroom": route_statistic("route_mean_legroom", "Leg room service", "mean"),
    **{name: expected_rating(name) for name in EXPECTED_RATINGS},
    "group_deviation": group_deviation,
    "route_std_wifi": route_statistic("route_std_wifi", "Inflight wifi service", "std"),
}
ADOPTED_FEATURES = list(FEATURE_BUILDERS)  # 採用した順(3seedで確認済み)


def build_feature_table(df, feature_names=ADOPTED_FEATURES, base_features=BASE_FEATURES):
    """元の特徴量+指定した特徴量を並べ、カテゴリ列をcategory型にする(Target Encodingの列はキーのまま)。"""
    X = pd.concat([df[base_features]] + [FEATURE_BUILDERS[name](df) for name in feature_names], axis=1)
    for c in CATEGORICAL_COLUMNS:
        if c in X:
            X[c] = X[c].astype("category")
    return X


def build_train_and_test_tables(train, test, feature_names=ADOPTED_FEATURES, base_features=BASE_FEATURES):
    """trainとtestの特徴量の表を作り、testのカテゴリ番号をtrainにそろえる。"""
    X, X_test = build_feature_table(train, feature_names, base_features), build_feature_table(test, feature_names, base_features)
    for c in X.columns[X.dtypes == "category"]:
        X_test[c] = pd.Categorical(X_test[c], categories=X[c].cat.categories)
    return X, X_test
