"""特徴量の定義: 元の特徴量の一覧、採用した特徴量(ADOPTED_FEATURES)の作り方、Target Encoding。

目的: 採用した特徴量の作り方を1か所にまとめ、どのモデルのスクリプトからも同じ特徴量を使えるようにする。
ひとことで言うと: 「材料の下ごしらえ」の手順書。学習や採点はしない(それは同じフォルダの train_lightgbm.py などが行う)。
やり方:
  - FEATURE_BUILDERS: 名前 → 関数(データを受け取り、足す列を返す)。ここには採用した特徴量だけを置く。
    試したが採用しなかった候補は rejected_feature_candidates.py にあり、train_lightgbm.py --feature 名前 で判定し直せる。
  - ADOPTED_FEATURES: 判定で採用した特徴量の名前。モデルのスクリプトは元の特徴量20個(BASE_FEATURES)+ADOPTED_FEATURESで学習する。
  - TARGET_ENCODING_SPECS: 値ごとの満足率(Target Encoding)の列。答えを使うので、FEATURE_BUILDERSでは元の値だけを置き、
    学習ループの中でfoldごとに満足率へ置き換える(target_encode_in_folds)。
  - 考え方とコードを順に追うには feature_engineering.ipynb を見る(同じ特徴量になることをノートブックの最後で照合している)。
なぜ: LightGBM・XGBoost・ハイパラ探索で特徴量の定義が食い違わないようにするため。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_FEATURES = [  # select_baseline_features.pyのFEATURE_ORDER[:20](Genderを除く元の特徴量20個)
    "Online boarding", "Class", "Inflight entertainment", "Inflight wifi service", "Seat comfort",
    "Flight Distance", "Type of Travel", "On-board service", "Cleanliness", "Leg room service", "Age",
    "Ease of Online booking", "Baggage handling", "Checkin service", "Food and drink", "Gate location",
    "Customer Type", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
    "Departure Delay in Minutes",
]
CATEGORICAL_COLUMNS = ["Customer Type", "Type of Travel", "Class"]
SERVICE_RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
                   "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
                   "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]  # サービス評価0〜5
DEFAULT_LGB_PARAMS = dict(objective="binary", learning_rate=0.05, n_estimators=10000, random_state=42, verbose=-1)  # LightGBMの既定ハイパラ(元データの先生役でも使う。train_lightgbm.pyの--paramsで置き換わる)


COMFORT_RATINGS = ["Seat comfort", "Leg room service", "Inflight entertainment"]  # 機内の快適さ


def comfort(df):
    # 機内の快適さ = 座席・足元・機内エンタメの平均(0は未回答の可能性があるため除く)
    return df[COMFORT_RATINGS].replace(0, np.nan).mean(axis=1)


def comfort_gate_business(df):
    # Type of Travelをゲートにする: 出張客だけ快適さの値を持ち、私用客(快適さに関係なく満足率約10%)は欠損
    return pd.DataFrame({"comfort_business": comfort(df).where(df["Type of Travel"] == "Business travel")})


def comfort_gate_loyal_business(df):
    # 常連かつ出張客だけ快適さ、他は欠損。disloyal客(99.9%が出張客)は快適さが効かない(出張客内AUC 0.51、
    # 常連は0.80〜0.86、EDA 8節)ため、comfort_gate_businessから外す。comfort_gate_businessと置き換えて比べる
    mask = (df["Type of Travel"] == "Business travel") & (df["Customer Type"] == "Loyal Customer")
    return pd.DataFrame({"comfort_loyal_business": comfort(df).where(mask)})


# Target Encoding(TE): 値ごとの満足率を列にする。答え(satisfaction)を使うため、FEATURE_BUILDERSでは元の値だけを
# 置いておき、学習ループの中でfoldごとに満足率へ置き換える(target_encode_in_folds)。列名 -> (元の列, smoothingの強さm)
# 不採用のTEの候補は rejected_feature_candidates.py が読み込まれたときにここへ足される。
TARGET_ENCODING_SPECS = {"te_distance_m3": ("Flight Distance", 3), "te_dist_class": (["Flight Distance", "Class"], 3)}

def target_encoding_key(name):
    # TEのキー(元の値)を作る。満足率への置き換えは学習ループの中で行う(target_encode_in_folds)
    def build(df):
        cols = TARGET_ENCODING_SPECS[name][0]
        if isinstance(cols, list):  # 組み合わせは各列の値を文字列でつないだキーにする
            return pd.DataFrame({name: df[cols].fillna(-1).astype(str).agg("|".join, axis=1)})
        return pd.DataFrame({name: df[cols]})
    return build


def target_encode(key_fit, y_fit, key_apply, m):
    # 値ごとの満足率を、人数が少ないほど全体の満足率に寄せて計算する(smoothing)。trainに無い値は全体の満足率
    prior = y_fit.mean()
    stats = pd.DataFrame({"key": key_fit, "y": y_fit}).groupby("key")["y"].agg(["sum", "count"])
    return pd.Series(key_apply).map((stats["sum"] + m * prior) / (stats["count"] + m)).fillna(prior).values


def target_encode_in_folds(X_fit, y_fit, others, seed):
    # 学習用の行は、その中でさらに5分割して自分を含まない満足率を入れる(自分の答えが混ざるのを防ぐ)。
    # 検証用・testの行(others)には、学習用の行全体の満足率を入れる
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


def frequency_encode(name, cols):
    # Frequency Encoding: colsの値(の組み合わせ)がtrain+testに何回出てくるか(目的変数は使わない)。
    # 残差との相関はほぼ0でも学習すると効くことがあるため、事前確認で落とさず学習で確かめる
    def build(df):
        # usecolsはCSVの列順で返すため、colsの順に並べ直してキーの文字列をそろえる
        train_and_test = pd.concat([pd.read_csv(fn, usecols=cols)[cols] for fn in ("data/train.csv", "data/test.csv")])
        counts = train_and_test.fillna(-1).astype(str).agg("|".join, axis=1).value_counts()
        return pd.DataFrame({name: df[cols].fillna(-1).astype(str).agg("|".join, axis=1).map(counts).values}, index=df.index)
    return build


def route_mean(name, col):
    # ルートプロファイル(公開ノートブック・Discussion): 飛行距離の値を「路線」とみなし、train+testで同じ距離の乗客の
    # colの平均を列にする(目的変数は使わない)。距離のTE(採用済み)は路線の満足率そのものだが、こちらは路線の
    # 「乗客の顔ぶれ・評価の傾向」で、少人数の路線でもラベルのぶれを含まない。路線の人数(=freq_dist)は不採用済み
    def build(df):
        train_and_test = pd.concat([pd.read_csv(fn, usecols=["Flight Distance", col]) for fn in ("data/train.csv", "data/test.csv")])
        return pd.DataFrame({name: df["Flight Distance"].map(train_and_test.groupby("Flight Distance")[col].mean()).values}, index=df.index)
    return build


def route_std(name, col):
    # 路線ごとの評価の標準偏差(train+test、Wi-Fiを採用)。路線平均(rp_)が同じでも、評価がそろった路線とばらけた路線を区別する
    def build(df):
        train_and_test = pd.concat([pd.read_csv(fn, usecols=["Flight Distance", col]) for fn in ("data/train.csv", "data/test.csv")])
        return pd.DataFrame({name: df["Flight Distance"].map(train_and_test.groupby("Flight Distance")[col].std()).values}, index=df.index)
    return build


def group_deviation(name, keys, nums=None, z=False):
    # 背景グループ統計(公開ノートブックでClass×旅行目的×顧客タイプ版を採用)の「行の値−グループ平均」だけを取り出した版。keysで乗客をグループ分けし
    # (例: Class×旅行目的×顧客タイプの12グループ)、年齢・距離・遅延2列がグループの中で高いか低いかを4列で渡す。
    # グループ平均・標準偏差そのものはグループ(=既存の列の組み合わせ)ごとに一定で木が作れるため、ここでは偏差だけ。
    # 平均はtrain+testで計算(目的変数は使わない)
    # nums: 偏差を取る列(既定は年齢・距離・遅延2列)。z=True: グループの標準偏差で割る(グループごとに散らばりが違っても同じ物差しにする、不採用)
    nums = nums or {"age": "Age", "dist": "Flight Distance", "dep": "Departure Delay in Minutes", "arr": "Arrival Delay in Minutes"}
    def build(df):
        train_and_test = pd.concat([pd.read_csv(fn, usecols=keys + list(nums.values())) for fn in ("data/train.csv", "data/test.csv")])
        new_columns = {}
        for short, c in nums.items():
            group_stats = train_and_test.groupby(keys)[c].agg(m="mean", s="std")
            row_stats = df[keys].merge(group_stats, left_on=keys, right_index=True, how="left")
            new_columns[f"{name}_{short}"] = (df[c].values - row_stats["m"].values) / (row_stats["s"].values if z else 1)
        return pd.DataFrame(new_columns, index=df.index)
    return build


def expected_rating(name):
    # 期待値の列(公開ノートブックのアイデア): make_expected_rating.pyで前もって作った「他の列から予想したこの項目の点数」(train+testの順に保存)。
    # trainとtestで行数が違うので、渡された行数で切り分ける
    def build(df):
        expected = np.load(f"data/aux_ev_{name[3:]}.npy")
        n_train = len(pd.read_csv("data/train.csv", usecols=["id"]))
        return pd.DataFrame({name: expected[:n_train] if len(df) == n_train else expected[n_train:]}, index=df.index)
    return build


def none(df):
    # 何も足さない(ADOPTED_FEATURESだけのベースラインを、--cv-seedで別のfold分割でも学習するため)
    return pd.DataFrame(index=df.index)


def gate_business(col):
    # 出張客だけ元の評価(0も含む)、私用客は欠損。私用客ではほとんどの評価項目が満足度にほぼ効かない
    # (出張客内の単変量AUC 0.70〜0.79に対し私用客内は0.51前後)ため、その区別を1列で渡す
    def build(df):
        return pd.DataFrame({f"{col}_business": df[col].where(df["Type of Travel"] == "Business travel")})
    return build


# 採用した特徴量(① 条件で絞る ② 似た乗客で数える・平均する ③ モデルで予想する)
FEATURE_BUILDERS = {
    "none": none,  # 何も足さない(採用した特徴量だけで学習するとき)
    # ① 条件で絞る(ゲート特徴量)
    "comfort_gate_business": comfort_gate_business,
    "comfort_gate_loyal_business": comfort_gate_loyal_business,
    "gate_onboard": gate_business("On-board service"),
    # ② 似た乗客で数える・平均する
    "te_distance_m3": target_encoding_key("te_distance_m3"),
    "te_dist_class": target_encoding_key("te_dist_class"),
    "freq_dist_travel": frequency_encode("freq_dist_travel", ["Flight Distance", "Type of Travel"]),
    "rp_legroom": route_mean("rp_legroom", "Leg room service"),
    "rsd_wifi": route_std("rsd_wifi", "Inflight wifi service"),
    "ctx_ctc": group_deviation("ctx_ctc", ["Class", "Type of Travel", "Customer Type"]),
    # ③ モデルで予想する(評価の期待値。make_expected_rating.pyで前もって作る)
    "ev_boarding": expected_rating("ev_boarding"),
    "ev_legroom": expected_rating("ev_legroom"),
    "ev_checkin": expected_rating("ev_checkin"),
    "ev_time": expected_rating("ev_time"),
}
ADOPTED_FEATURES = ["comfort_gate_business", "gate_onboard", "te_distance_m3", "te_dist_class", "freq_dist_travel", "comfort_gate_loyal_business",
                    "rp_legroom", "ev_boarding", "ev_legroom", "ev_checkin", "ev_time", "ctx_ctc", "rsd_wifi"]  # 採用した順(3seedで確認済み)
