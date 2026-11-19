"""試したが採用しなかった特徴量の候補(約90個)。採否の判定の記録として残す。

目的: 採用した特徴量(feature_definitions.py)と分けて置き、本体を短く読みやすくする。どれを試したかの証拠でもある。
ひとことで言うと: 「試作したが使わなかった下ごしらえ」の手順書。
やり方:
  - REJECTED_FEATURE_BUILDERS: 名前 → 関数。train_lightgbm.py・train_xgboost.py は feature_definitions.FEATURE_BUILDERS と
    合わせて読み込むので、`train_lightgbm.py --feature 名前` で判定をやり直せる。
  - 読み込まれたとき、不採用のTEの候補(REJECTED_TARGET_ENCODING_SPECS)を feature_definitions.TARGET_ENCODING_SPECS に足す
    (TEの置き換えは feature_definitions.target_encode_in_folds が行うため)。
  - 各候補のコメントに、着想と結果(不採用・保留)を書いている。採否の基準はREADMEの2の節。
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import feature_definitions as features
from feature_definitions import (BASE_FEATURES, CATEGORICAL_COLUMNS, SERVICE_RATINGS, comfort, comfort_gate_business,
                                 comfort_gate_loyal_business, expected_rating, frequency_encode, gate_business, group_deviation, none,
                                 route_mean, route_std, target_encoding_key)


# Target Encoding(TE): 値ごとの満足率を列にする。答え(satisfaction)を使うため、FE関数では元の値だけを
# 置いておき、学習ループの中でfoldごとに満足率へ置き換える(target_encode_in_folds)。列名 -> (元の列, smoothingの強さm)
REJECTED_TARGET_ENCODING_SPECS = {"te_distance": ("Flight Distance", 20),
                         "te_distance_m5": ("Flight Distance", 5), "te_distance_m50": ("Flight Distance", 50), "te_distance_m1": ("Flight Distance", 1),
                         "te_distance_m10": ("Flight Distance", 10),
                         "te_age": ("Age", 3),
                         "te_age_dist": (["Age", "Flight Distance"], 3),
                         "te_dist_travel": (["Flight Distance", "Type of Travel"], 3),
      # 評価2つの組み合わせ(6×6=36通り)。78ペアのうち「組み合わせのTEの単変量AUC − 単独で強い方」が大きい上位3つ
                         "te_wifi_gate": (["Inflight wifi service", "Gate location"], 3),
                         "te_time_gate": (["Departure/Arrival time convenient", "Gate location"], 3),
                         "te_wifi_time": (["Inflight wifi service", "Departure/Arrival time convenient"], 3),
                         "te_depdelay": ("Departure Delay in Minutes", 3), "te_arrdelay": ("Arrival Delay in Minutes", 3),  # 欠損はTEで全体の満足率  # 回数(freq_dist_travel)と同じキーの満足率
                         "te_svc13": (SERVICE_RATINGS, 3),
                         "te_age_class": (["Age", "Class"], 3), "te_age_travel": (["Age", "Type of Travel"], 3),
                         "te_ctc": (["Class", "Type of Travel", "Customer Type"], 3), "te_class_wifi": (["Class", "Inflight wifi service"], 3),
                         "te_wifi_boarding": (["Inflight wifi service", "Online boarding"], 3)}  # 年齢×カテゴリ(不採用)  # 組み合わせのTE: 複数の列の値をつなげた1つのキーごとの満足率  # mの比較用
features.TARGET_ENCODING_SPECS.update(REJECTED_TARGET_ENCODING_SPECS)

def zero_flags(df):
    # 評価が0かどうかのフラグを項目ごとに作る(0が1件も無いBaggage handlingは除く)
    return pd.DataFrame({f"{c}_is0": (df[c] == 0).astype(int) for c in SERVICE_RATINGS if c != "Baggage handling"})


def zero_count(df):
    # 13項目のうち評価0の個数(複数の列にまたがる情報なので、木が1回の分岐では作れない)
    return pd.DataFrame({"svc_zero_count": (df[SERVICE_RATINGS] == 0).sum(axis=1)})


def count_score(k):
    # 13項目のうち評価がkの項目数(0点の個数=zero_countと同じ数え方)。5点の個数は満足、1点の個数は不満の多さ
    def build(df):
        return pd.DataFrame({f"svc_count{k}": (df[SERVICE_RATINGS] == k).sum(axis=1)})
    return build


def svc_mean(df):
    # 評価の平均。0は未利用・該当なしの項目があるため(EDA 7節)、0を除いて平均する
    return pd.DataFrame({"svc_mean": df[SERVICE_RATINGS].replace(0, np.nan).mean(axis=1)})


def svc_min(df):
    # 最も低い評価(0を除く)。1つでも強い不満があるかを表す
    return pd.DataFrame({"svc_min": df[SERVICE_RATINGS].replace(0, np.nan).min(axis=1)})


def svc_std(df):
    # 評価のばらつき(標準偏差、0を除く)。平均が同じでも「全部3点」と「5点と1点が混在」を区別する
    return pd.DataFrame({"svc_std": df[SERVICE_RATINGS].replace(0, np.nan).std(axis=1)})


def svc_std_business(df):
    # 出張客だけ評価の標準偏差、私用客は欠損(集計値にゲートを付けた型。ゲートなしは不採用)
    return pd.DataFrame({"svc_std_business": svc_std(df)["svc_std"].where(df["Type of Travel"] == "Business travel")})


def svc_max(df):
    return pd.DataFrame({"svc_max": df[SERVICE_RATINGS].max(axis=1)})


def noise_control(df):
    # 対照実験用: 満足度と無関係な乱数の列。「列を1本足すだけで成績がどれだけ動くか」の目安にする
    return pd.DataFrame({"noise": np.random.default_rng(0).random(len(df))})


def dist_x_comfort(df):
    # 快適なら長距離ほど満足が上がり、不快なら上がらない(むしろ下がる)交互作用を1列で表す
    return pd.DataFrame({"dist_x_comfort": df["Flight Distance"] * comfort(df)})


def dist_div_comfort(df):
    # 比率の形。「長距離なのに不快」ほど大きくなる
    return pd.DataFrame({"dist_div_comfort": df["Flight Distance"] / comfort(df)})


def dist_x_comfort_business(df):
    # 出張客だけ飛行距離×快適さ、私用客は欠損(ゲートなしの掛け算は効かなかったため、ゲートと組み合わせて再確認)
    product = df["Flight Distance"] * comfort(df)
    return pd.DataFrame({"dist_x_comfort_business": product.where(df["Type of Travel"] == "Business travel")})


def dist_x_baggage(df):
    # 飛行距離×手荷物の評価(0なし)。距離5分位で手荷物の単変量AUCが0.570→0.789と最も大きく上がった項目
    return pd.DataFrame({"dist_x_baggage": df["Flight Distance"] * df["Baggage handling"]})


def dist_x_baggage_business(df):
    # 出張客だけ飛行距離×手荷物、私用客は欠損(全員の掛け算は-0.00005。私用客の値が混ざるのを除く)
    product = df["Flight Distance"] * df["Baggage handling"]
    return pd.DataFrame({"dist_x_baggage_business": product.where(df["Type of Travel"] == "Business travel")})


def baggage_longhaul(df):
    # 長距離(1189マイル超)だけ手荷物の評価、それ以外は欠損。距離との関係が段差型(1189マイル以下は手荷物の
    # 単変量AUC 0.60前後で横ばい、超えると0.72〜0.82)なので、掛け算ではなく段差の形で渡す
    return pd.DataFrame({"baggage_longhaul": df["Baggage handling"].where(df["Flight Distance"] > 1189)})


def comfort_mean(df):
    # ゲートなしの快適さ(comfort_gate_businessの改善がゲート由来かを切り分けるため)
    return pd.DataFrame({"comfort_mean": comfort(df)})


def gate_loyal_business(col):
    # 常連かつ出張客だけ元の評価(0も含む)、他は欠損。快適さ(comfort_gate_loyal_business、採用)と同じ型を機内サービスの
    # 単項目に広げる。出張客内の単変量AUCが常連とdisloyalで大きく違う項目(清潔さ0.799/0.508・機内サービス0.811/0.632・
    # 食事0.682/0.510)を選んだ
    def build(df):
        mask = (df["Type of Travel"] == "Business travel") & (df["Customer Type"] == "Loyal Customer")
        return pd.DataFrame({f"{col}_loyal_business": df[col].where(mask)})
    return build


def svc3_gate_loyal_business(df):
    # 常連かつ出張客だけ、清潔さ・機内サービス・食事の平均(0を除く)、他は欠損。単項目のゲートは木が分岐で
    # 作れて不採用だったが、平均(集計値)は作りにくく、快適さの平均(comfort_gate_loyal_business)では効いたため同じ型で試す
    mask = (df["Type of Travel"] == "Business travel") & (df["Customer Type"] == "Loyal Customer")
    mean3 = df[["Cleanliness", "On-board service", "Food and drink"]].replace(0, np.nan).mean(axis=1)
    return pd.DataFrame({"svc3_loyal_business": mean3.where(mask)})


def gate_mean_business(name, cols):
    # 出張客だけ、評価項目cols(0を除く)の平均、私用客は欠損。comfort_gate_businessの項目の選び方を見直すための版
    def build(df):
        mean = df[cols].replace(0, np.nan).mean(axis=1)
        return pd.DataFrame({name: mean.where(df["Type of Travel"] == "Business travel")})
    return build


# 出張客内の順位相関で、互いに0.54〜0.69と強く連動する「機内の環境」のまとまり(足元は機内サービス・手荷物側に近い)
CABIN_RATINGS = ["Inflight entertainment", "Seat comfort", "Cleanliness", "Food and drink"]

STAFF_RATINGS = ["On-board service", "Baggage handling", "Leg room service"]  # 「スタッフ・サービス」のまとまり(相関0.51〜0.63)


def gate_class_business(col):
    # Businessクラスだけ元の評価、Eco・Eco Plusは欠損。出張客内でもBusinessクラスとEcoで効き方が違う
    # (例: 足元の単変量AUC 0.730/0.553、EDA 8節)。Businessクラスの99%は出張客
    def build(df):
        return pd.DataFrame({f"{col}_bizclass": df[col].where(df["Class"] == "Business")})
    return build


def wifi_eco(df):
    # Eco・Eco PlusだけWi-Fiの評価(0も含む)、Businessクラスは欠損。Wi-Fiは他の項目と逆にEco側でよく効く
    # (単変量AUC Business 0.618・Eco 0.800・Eco Plus 0.819、EDA 8節)
    return pd.DataFrame({"wifi_eco": df["Inflight wifi service"].where(df["Class"] != "Business")})


def gate_where(name, col, gate_col, gate_val):
    # gate_colがgate_valの行だけcolの値、他は欠損(逆向きのゲート用。私用客・disloyal客だけのWi-Fi・オンライン予約等)
    def build(df):
        return pd.DataFrame({name: df[col].where(df[gate_col] == gate_val)})
    return build


def disloyal_age31_36(df):
    # disloyal客かつ31〜36歳のフラグ。この帯だけ満足率2〜4%と極端に低く(前後は13〜19%)、
    # ゲートの網羅チェックで唯一|z|>3の残差(予測が0.8ポイント高い)が残っていた
    mask = (df["Customer Type"] == "disloyal Customer") & df["Age"].between(31, 36)
    return pd.DataFrame({"disloyal_age31_36": mask.astype(int)})


def dist_cat(df):
    # Exact Value Encoding: 飛行距離の値そのものをカテゴリ(category型)として渡す。LightGBMは値ごとの満足率の順に
    # 並べ替えて分岐するため、TEと同じ「値ごとのクセ」を、満足率を自分で計算せずに木へ直接渡せる
    return pd.DataFrame({"dist_cat": df["Flight Distance"].astype("category")})


def delay_combo(name):
    # 比例尺度どうし(遅延・飛行距離)の差・比・和。木は2列の差や比を何度も分岐して階段状に近似するしかないため、
    # 1列にまとめて渡す。単独の集計では「遅れを取り戻すと満足」は見えなかった(出発30分以上遅れた7,193人で31〜33%と横ばい)
    def build(df):
        dep, arr, dist = df["Departure Delay in Minutes"], df["Arrival Delay in Minutes"], df["Flight Distance"]
        values = {"delay_diff": arr - dep,  # マイナス=飛行中に遅れを取り戻した
                  "arr_delay_per_dist": arr / dist,  # 距離のわりに遅れが大きいか
                  "delay_ratio": (arr / dep).where(dep > 0),  # 出発遅延0では割れないため欠損
                  "delay_sum": dep + arr}[name]
        return pd.DataFrame({name: values})
    return build


# 列を足さずに元の列を書き換える施策(--featureで選ぶと、列を作る前にtrain・test・元データへ適用し、追加列は0本)
PREPROCESSING = {
    # 到着遅延の欠損(train 292行・test 130行)を同じ人の出発遅延で埋める。欠損でない行の87%は到着=出発で、相関0.84
    "fill_arrdelay": lambda df: df.assign(**{"Arrival Delay in Minutes": df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])}),
}


_teacher_cache = {}  # 元データで学習したモデル(設定ごとに1回だけ作って使い回す)
# 公開ノートブックの教師の設定。600本固定・深さ8・列の抽選0.5・正則化(L2 5、L1 0.05)。
# subsampleはsubsample_freqが無いとLightGBMでは効かないため、同ノートブックと同じく実質なし
PUBLIC_TEACHER_PARAMS = dict(objective="binary", n_estimators=600, learning_rate=0.05, max_depth=8, min_child_samples=20,
                             subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.05, random_state=123, verbose=-1)


def orig_teacher_fe(name, params=None):
    # 元データ(ノイズが少なく、元の特徴量20個だけでAUC 0.995)だけで学習したLightGBMの予測を1列として渡す。元データを
    # 行として足す方法は悪化したが、公開ノートブック3件で予測を列にする方法は
    # +0.00055〜+0.00072。教師モデルはコンペのラベルを一切見ないため、foldの外で作ってもリークにならない。
    # params=None: 既定ハイパラ(features.DEFAULT_LGB_PARAMS)で、元データの20%で早期終了して本数を決め全体に学習し直す
    def build(df):
        if name not in _teacher_cache:
            orig = pd.concat([pd.read_csv(f"data/original/{f}.csv") for f in ("train", "test")], ignore_index=True)
            y_orig = (orig["satisfaction"] == "satisfied").astype(int).values  # 元データは文字列のラベル
            X_orig = orig[BASE_FEATURES].copy()
            for c in CATEGORICAL_COLUMNS:
                X_orig[c] = X_orig[c].astype("category")
            if params is None:
                tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X_orig, y_orig))
                model = lgb.LGBMClassifier(**features.DEFAULT_LGB_PARAMS).fit(X_orig.iloc[tr], y_orig[tr], eval_set=[(X_orig.iloc[va], y_orig[va])], eval_metric="auc",
                                                                     callbacks=[lgb.early_stopping(100, verbose=False)])
                print(f"教師モデル: 元データ内AUC {roc_auc_score(y_orig[va], model.predict_proba(X_orig.iloc[va])[:, 1]):.4f}、木{model.best_iteration_}本")
                model = lgb.LGBMClassifier(**{**features.DEFAULT_LGB_PARAMS, "n_estimators": model.best_iteration_}).fit(X_orig, y_orig)
            else:
                model = lgb.LGBMClassifier(**params).fit(X_orig, y_orig)
            _teacher_cache[name] = (model, X_orig.dtypes)
        model, train_dtypes = _teacher_cache[name]
        X = df[BASE_FEATURES].copy()
        for c in CATEGORICAL_COLUMNS:
            X[c] = pd.Categorical(X[c], categories=train_dtypes[c].categories)
        return pd.DataFrame({name: model.predict_proba(X)[:, 1]}, index=df.index)
    return build


def orig_rate(df, m=10, cols=None):
    # 元の特徴量20個それぞれについて、元データでの「その値の人の満足率」を1列ずつ渡す(+20列、公開ノートブックの`org`)。
    # 先生役(1つのモデルの確信)と違い、列ごとの単純な平均なので強い確信を持ち込みにくい。先生役と同じくPublicで
    # 下がるかを確かめる。少数の値の極端な率を抑えるため全体の満足率へm件分寄せる。元データに無い値は欠損
    orig = pd.concat([pd.read_csv(f"data/original/{f}.csv") for f in ("train", "test")], ignore_index=True)
    y_orig = (orig["satisfaction"] == "satisfied").astype(int)
    new_columns = {}
    for c in cols or BASE_FEATURES:
        stats = y_orig.groupby(orig[c]).agg(["sum", "count"])
        rate = (stats["sum"] + m * y_orig.mean()) / (stats["count"] + m)
        new_columns[f"org_{c}"] = df[c].map(rate).astype(float)
    return pd.DataFrame(new_columns, index=df.index)
# 26列の上では20列版は悪化。評価・カテゴリは値の種類が少なく木が元の列で分けられ、飛行距離は既存のTEと重なるため、
# 値の種類が多く既存のFEと重ならない年齢・遅延の3列だけに絞る
ORIGINAL_RATE_COLUMNS = ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def eq_flag(name, pairs):
    # 評価の一致フラグ(4種とも不採用): 2つの評価の点数が等しいか。時刻の便利さとゲートの場所は、点数が等しいと満足率54〜69%、
    # 違うと7〜36%。木は元の2つの特徴量から『等しい』を作るのに対角線を何度も分ける必要があるので、1列で渡す。
    # pairsが複数なら、等しいペアの数
    def build(df):
        return pd.DataFrame({name: sum((df[a] == df[b]).astype(int) for a, b in pairs)}, index=df.index)
    return build


def route_share(name, col, val):
    # 路線(飛行距離の値)ごとの、colがvalである乗客の割合(train+test、目的変数不使用)(3つとも不採用)。距離×Class・距離×旅行目的のTE・回数は
    # 採用済みだが、こちらは路線の客層(ビジネス客の多い路線か)
    def build(df):
        train_and_test = pd.concat([pd.read_csv(fn, usecols=["Flight Distance", col]) for fn in ("data/train.csv", "data/test.csv")])
        share = (train_and_test[col] == val).groupby(train_and_test["Flight Distance"]).mean()
        return pd.DataFrame({name: df["Flight Distance"].map(share).values}, index=df.index)
    return build


def route_zero(name, col, mode):
    # 路線の0の割合(mode="zero")・0を除いた路線平均(mode="nz")。評価の0は「該当なし」に近く(0の人の満足率はWi-Fi 88.7%・
    # オンライン予約70.8%・搭乗62.0%と全体44%から外れる)、0込みの路線平均(route_mean)では「0が多い路線」と「評価が低い路線」が
    # 混ざる。分けて渡せば、路線の利用のされ方(0の割合)と評価の高さ(0除外平均)を別々に使える。公開ノートブックは全て0込み
    def build(df):
        train_and_test = pd.concat([pd.read_csv(fn, usecols=["Flight Distance", col]) for fn in ("data/train.csv", "data/test.csv")])
        values = (train_and_test[col] == 0).astype(float) if mode == "zero" else train_and_test[col].replace(0, np.nan)
        return pd.DataFrame({name: df["Flight Distance"].map(values.groupby(train_and_test["Flight Distance"]).mean()).values}, index=df.index)
    return build


def route_diff(name, col):
    # 自分の評価−路線平均(公開ノートブックのアイデア): 路線平均(route_mean、rp_legroomで採用)は路線の乗客の傾向だが、こちらは
    # 「この路線の乗客の中で、この人は甘いか辛いか」。木は元の評価と路線平均の2列を分岐で引き算できないので新しい情報になりうる
    def build(df):
        route_avg = route_mean("_m", col)(df)["_m"]
        return pd.DataFrame({name: df[col].values - route_avg.values}, index=df.index)
    return build


def digits(name, col):
    # 桁の特徴量(公開ノートブックのアイデア): 値の各桁 floor(x/10^k)%10(k=0〜3、値が1種類しかない桁は除く)。人工データの生成で
    # 「末尾が0の距離」のような桁のクセが残っていれば、木が元の値の大小の分岐では作れない区別(例: 末尾が0の値をまとめる)を渡せる。
    # 欠損(到着遅延)は欠損のまま(0で埋めると遅延0分と混ざるため)
    def build(df):
        values = df[col].astype(float)
        new_columns = {f"{name}_d{k}": np.floor(values / 10 ** k) % 10 for k in range(4)}
        return pd.DataFrame({k: x for k, x in new_columns.items() if x.nunique() > 1}, index=df.index)
    return build


def digit1(name, col, k):
    # 桁を1つだけ足す版(元の列ごとにまとめて入れたら全て不採用だったため、効く桁が他の桁の入場料で
    # 打ち消されていないかを1桁ずつ確かめる)。定義はdigitsと同じ
    def build(df):
        return pd.DataFrame({name: np.floor(df[col].astype(float) / 10 ** k) % 10}, index=df.index)
    return build


def gpt2_token(name, col):
    # GPT-2トークンキー(公開ノートブックのアイデア): 値を' '+文字列にしてGPT-2の区切り方でトークンに分ける(例: 694→' 6'+'94'、460→' 460')。
    # 人工データの生成器が数値を文字のかたまりごとに作っていれば、区切れ方のクセ(例: 末尾が'94'の値)が満足率に残る可能性がある。
    # 列: 先頭トークンID・末尾トークンID+100000×トークン数・トークン数(投稿者の定義どおり)
    def build(df):
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        values = df[col]
        toks = {u: enc.encode(" " + str(int(u))) for u in values.dropna().unique()}  # 値の種類ごとに1回だけ区切る
        first = values.map(lambda u: toks[u][0] if u in toks else np.nan)
        n_tokens = values.map(lambda u: len(toks[u]) if u in toks else np.nan)
        last = values.map(lambda u: toks[u][-1] + 100000 * len(toks[u]) if u in toks else np.nan)
        return pd.DataFrame({f"{name}_first": first, f"{name}_last": last, f"{name}_n": n_tokens}, index=df.index).astype(float)
    return build


def age_div_dist(df):
    # 年齢÷飛行距離(不採用)。到着遅延÷距離は保留(+0.000018)
    return pd.DataFrame({"age_div_dist": df["Age"] / df["Flight Distance"]}, index=df.index)


def gender(df):
    # 元のGenderを戻す。前向き追加では効かなかったが、当時は21列。TEの事前確認で今のモデルの残差との
    # 相関+0.0037(組み合わせのTEと同程度)だったため、事前確認で落とさず学習で確かめる
    return pd.DataFrame({"Gender": df["Gender"].astype("category")})


REJECTED_FEATURE_BUILDERS = {"none": none, "ev_boarding": expected_rating("ev_boarding"), "ev_legroom": expected_rating("ev_legroom"), **{f"ev_{n}": expected_rating(f"ev_{n}") for n in ("checkin", "onboard", "baggage", "food", "wifi", "clean", "seat", "booking", "ent", "gate", "time", "class", "travel", "cust")}, **{f"{k}_{n}": route_zero(f"{k}_{n}", c, mode) for n, c in (
                         ("wifi", "Inflight wifi service"), ("booking", "Ease of Online booking"), ("boarding", "Online boarding"))
                         for k, mode in (("rz", "zero"), ("rnz", "nz"))},
                    "tok_dist": gpt2_token("tok_dist", "Flight Distance"), **{f"dg1_{n}_d{k}": digit1(f"dg1_{n}_d{k}", c, k) for n, c, ks in (
                         ("dist", "Flight Distance", range(4)), ("age", "Age", range(2)), ("depdelay", "Departure Delay in Minutes", range(3)),
                         ("arrdelay", "Arrival Delay in Minutes", range(3))) for k in ks},
                    **{n: digits(n, c) for n, c in (("dg_dist", "Flight Distance"), ("dg_age", "Age"), ("dg_depdelay", "Departure Delay in Minutes"), ("dg_arrdelay", "Arrival Delay in Minutes"))}, "rpd_legroom": route_diff("rpd_legroom", "Leg room service"), "rp_baggage": route_mean("rp_baggage", "Baggage handling"), "rp_legroom": route_mean("rp_legroom", "Leg room service"),
                    "rp_onboard": route_mean("rp_onboard", "On-board service"), **{n: route_mean(n, c) for n, c in (
                        ("rp_wifi", "Inflight wifi service"), ("rp_time", "Departure/Arrival time convenient"), ("rp_booking", "Ease of Online booking"),
                        ("rp_gate", "Gate location"), ("rp_food", "Food and drink"), ("rp_boarding", "Online boarding"), ("rp_seat", "Seat comfort"),
                        ("rp_ent", "Inflight entertainment"), ("rp_checkin", "Checkin service"), ("rp_clean", "Cleanliness"), ("rp_age", "Age"),
                        ("rp_depdelay", "Departure Delay in Minutes"), ("rp_arrdelay", "Arrival Delay in Minutes"))},
                    "orig_rate": orig_rate, "orig_rate_num3": lambda df: orig_rate(df, cols=ORIGINAL_RATE_COLUMNS), "orig_teacher": orig_teacher_fe("orig_teacher"), "orig_teacher_pub": orig_teacher_fe("orig_teacher_pub", PUBLIC_TEACHER_PARAMS), **{k: none for k in PREPROCESSING}, **{n: delay_combo(n) for n in ("delay_diff", "arr_delay_per_dist", "delay_ratio", "delay_sum")}, "noise_control": noise_control, "gender": gender, "freq_dist_travel": frequency_encode("freq_dist_travel", ["Flight Distance", "Type of Travel"]),
                    "freq_dist": frequency_encode("freq_dist", ["Flight Distance"]), "freq_age": frequency_encode("freq_age", ["Age"]),
                    "freq_depdelay": frequency_encode("freq_depdelay", ["Departure Delay in Minutes"]), "freq_arrdelay": frequency_encode("freq_arrdelay", ["Arrival Delay in Minutes"]),
                    "freq_dist_class": frequency_encode("freq_dist_class", ["Flight Distance", "Class"]), "freq_age_dist": frequency_encode("freq_age_dist", ["Age", "Flight Distance"]),
                    "freq_svc13": frequency_encode("freq_svc13", SERVICE_RATINGS), "freq_age_class_travel": frequency_encode("freq_age_class_travel", ["Age", "Class", "Type of Travel"]),
                    "freq_boarding": frequency_encode("freq_boarding", ["Online boarding"]),
                    "freq_dist_cust": frequency_encode("freq_dist_cust", ["Flight Distance", "Customer Type"]),
                    "freq_dist_class_travel": frequency_encode("freq_dist_class_travel", ["Flight Distance", "Class", "Type of Travel"]),
                    "freq_dist_travel_cust": frequency_encode("freq_dist_travel_cust", ["Flight Distance", "Type of Travel", "Customer Type"]),
                    "gate_onboard": gate_business("On-board service"), "loyal_clean": gate_loyal_business("Cleanliness"), "loyal_onboard": gate_loyal_business("On-board service"), "loyal_food": gate_loyal_business("Food and drink"), "svc3_loyal_business": svc3_gate_loyal_business, "gate_clean": gate_business("Cleanliness"),
                    "gate_baggage": gate_business("Baggage handling"), "gate_checkin": gate_business("Checkin service"),
                    "gate_ent": gate_business("Inflight entertainment"), "gate_seat": gate_business("Seat comfort"),
                    "gate_legroom": gate_business("Leg room service"), "gate_food": gate_business("Food and drink"),
                    "gate_boarding": gate_business("Online boarding"),
                    "cls_onboard": gate_class_business("On-board service"), "cls_legroom": gate_class_business("Leg room service"),
                    "cls_baggage": gate_class_business("Baggage handling"), "cls_checkin": gate_class_business("Checkin service"), "wifi_eco": wifi_eco, "dist_cat": dist_cat, "te_distance": target_encoding_key("te_distance"),
                    "te_distance_m5": target_encoding_key("te_distance_m5"), "te_distance_m50": target_encoding_key("te_distance_m50"), "te_distance_m1": target_encoding_key("te_distance_m1"), "te_distance_m3": target_encoding_key("te_distance_m3"), "te_distance_m10": target_encoding_key("te_distance_m10"), "te_age": target_encoding_key("te_age"), "te_dist_class": target_encoding_key("te_dist_class"), "te_dist_travel": target_encoding_key("te_dist_travel"), "te_wifi_gate": target_encoding_key("te_wifi_gate"), "te_time_gate": target_encoding_key("te_time_gate"), "te_wifi_time": target_encoding_key("te_wifi_time"), "te_depdelay": target_encoding_key("te_depdelay"), "te_arrdelay": target_encoding_key("te_arrdelay"),
                    "te_age_dist": target_encoding_key("te_age_dist"), "te_svc13": target_encoding_key("te_svc13"), "disloyal_age31_36": disloyal_age31_36,
                    "wifi_personal": gate_where("wifi_personal", "Inflight wifi service", "Type of Travel", "Personal Travel"),
                    "booking_personal": gate_where("booking_personal", "Ease of Online booking", "Type of Travel", "Personal Travel"),
                    "wifi_disloyal": gate_where("wifi_disloyal", "Inflight wifi service", "Customer Type", "disloyal Customer"),
                    "booking_disloyal": gate_where("booking_disloyal", "Ease of Online booking", "Customer Type", "disloyal Customer"),
                    "cabin_gate_business": gate_mean_business("cabin_business", CABIN_RATINGS),
                    "cabin_staff_gate_business": gate_mean_business("cabin_staff_business", CABIN_RATINGS + STAFF_RATINGS), "dist_x_comfort": dist_x_comfort, "dist_div_comfort": dist_div_comfort,
                    "comfort_gate_business": comfort_gate_business, "comfort_mean": comfort_mean, "comfort_gate_loyal_business": comfort_gate_loyal_business,
                    "dist_x_comfort_business": dist_x_comfort_business, "dist_x_baggage": dist_x_baggage, "baggage_longhaul": baggage_longhaul, "dist_x_baggage_business": dist_x_baggage_business, "zero_flags": zero_flags, "zero_count": zero_count, "svc_mean": svc_mean, "svc_min": svc_min,
                    "svc_max": svc_max, "svc_std": svc_std, "svc_std_business": svc_std_business, "count5": count_score(5), "count1": count_score(1)}
REJECTED_FEATURE_BUILDERS.update({"ctx_ctc": group_deviation("ctx_ctc", ["Class", "Type of Travel", "Customer Type"]),
                         "ctx_bct": group_deviation("ctx_bct", ["Online boarding", "Class", "Type of Travel"]),
                         "ctx_wct": group_deviation("ctx_wct", ["Inflight wifi service", "Class", "Type of Travel"])})
REJECTED_FEATURE_BUILDERS.update({"eq_time_gate": eq_flag("eq_time_gate", [("Departure/Arrival time convenient", "Gate location")]),
                         "eq_wifi_time": eq_flag("eq_wifi_time", [("Inflight wifi service", "Departure/Arrival time convenient")]),
                         "eq_wifi_gate": eq_flag("eq_wifi_gate", [("Inflight wifi service", "Gate location")]),
                         "eq_count4": eq_flag("eq_count4", [("Inflight wifi service", "Departure/Arrival time convenient"), ("Inflight wifi service", "Ease of Online booking"), ("Inflight wifi service", "Gate location"), ("Departure/Arrival time convenient", "Ease of Online booking"), ("Departure/Arrival time convenient", "Gate location"), ("Ease of Online booking", "Gate location")]),
                         "rs_business": route_share("rs_business", "Class", "Business"),
                         "rs_travel": route_share("rs_travel", "Type of Travel", "Business travel"),
                         "rs_loyal": route_share("rs_loyal", "Customer Type", "Loyal Customer"),
                         "rsd_legroom": route_std("rsd_legroom", "Leg room service"),
                         "rsd_boarding": route_std("rsd_boarding", "Online boarding"),
                         "rsd_wifi": route_std("rsd_wifi", "Inflight wifi service"),
                         "te_age_class": target_encoding_key("te_age_class"), "te_age_travel": target_encoding_key("te_age_travel")})
ROUTE_STD_REST = {"rsd_booking": "Ease of Online booking", "rsd_time": "Departure/Arrival time convenient", "rsd_gate": "Gate location",
                  "rsd_food": "Food and drink", "rsd_seat": "Seat comfort", "rsd_ent": "Inflight entertainment", "rsd_onboard": "On-board service",
                  "rsd_baggage": "Baggage handling", "rsd_checkin": "Checkin service", "rsd_clean": "Cleanliness"}  # 残り10項目の路線標準偏差(全て不採用、予約は保留)
REJECTED_FEATURE_BUILDERS.update({k: route_std(k, c) for k, c in ROUTE_STD_REST.items()})
REJECTED_FEATURE_BUILDERS.update({"rsd_age": route_std("rsd_age", "Age"), "rsd_dep": route_std("rsd_dep", "Departure Delay in Minutes"),
                         "rsd_arr": route_std("rsd_arr", "Arrival Delay in Minutes"),  # 年齢・遅延の路線標準偏差(不採用)。客層内偏差の拡張・組み合わせTE・年齢÷距離も不採用
                         "ctx_ctc_wifi": group_deviation("ctx_ctc_wifi", ["Class", "Type of Travel", "Customer Type"], {"v": "Inflight wifi service"}),
                         "ctx_ctc_boarding": group_deviation("ctx_ctc_boarding", ["Class", "Type of Travel", "Customer Type"], {"v": "Online boarding"}),
                         "ctx_ctc_z": group_deviation("ctx_ctc_z", ["Class", "Type of Travel", "Customer Type"], z=True),
                         "te_ctc": target_encoding_key("te_ctc"), "te_class_wifi": target_encoding_key("te_class_wifi"), "te_wifi_boarding": target_encoding_key("te_wifi_boarding"),
                         "age_div_dist": age_div_dist})


def route_class_std(name, col):
    # 路線×Classごとの評価の標準偏差(追加候補B、不採用)。同じ路線でもBusinessとEcoで客層が違うため細かく分ける。人数が減る分ぶれやすい
    def build(df):
        keys = ["Flight Distance", "Class"]
        train_and_test = pd.concat([pd.read_csv(fn, usecols=keys + [col]) for fn in ("data/train.csv", "data/test.csv")])
        route_class_std_values = train_and_test.groupby(keys)[col].std().rename("s")
        return pd.DataFrame({name: df[keys].merge(route_class_std_values, left_on=keys, right_index=True, how="left")["s"].values}, index=df.index)
    return build


REJECTED_FEATURE_BUILDERS.update({"rs_wifi5": route_share("rs_wifi5", "Inflight wifi service", 5),  # 追加候補A: 路線のWi-Fi 5点の割合(rsd_wifiと順位相関0.60)
                         "rcsd_wifi": route_class_std("rcsd_wifi", "Inflight wifi service"),  # 追加候補B
           # 追加候補C: 路線平均との差(年齢・遅延)。足元の評価の差は不採用、客層グループ内の差(ctx_ctc)は採用
                         "rdev_age": group_deviation("rdev_age", ["Flight Distance"], {"v": "Age"}),
                         "rdev_dep": group_deviation("rdev_dep", ["Flight Distance"], {"v": "Departure Delay in Minutes"}),
                         "rdev_arr": group_deviation("rdev_arr", ["Flight Distance"], {"v": "Arrival Delay in Minutes"})})
# 追加候補D: 飛行距離を10刻み・100刻みに丸めた値のTE(公開ノートブック、不採用)。人数の少ない路線を近い距離とまとめ、満足率のぶれを抑える。
# 丸めたキーはFE関数で作り、TARGET_ENCODING_SPECSに名前を登録して学習ループ(target_encode_in_folds)で満足率に置き換える
features.TARGET_ENCODING_SPECS.update({"te_dist10": (None, 3), "te_dist100": (None, 3)})
REJECTED_FEATURE_BUILDERS.update({"te_dist10": lambda df: pd.DataFrame({"te_dist10": df["Flight Distance"] // 10}, index=df.index),
                         "te_dist100": lambda df: pd.DataFrame({"te_dist100": df["Flight Distance"] // 100}, index=df.index)})
DROP_CANDIDATES = {"drop_gate": ["Gate location"], "drop_dep": ["Departure Delay in Minutes"], "drop_arr": ["Arrival Delay in Minutes"]}  # 足すのではなく、元の特徴量を外して効果を見る候補(公開ノートブック、ゲート・遅延2列とも不採用)
REJECTED_FEATURE_BUILDERS.update({k: none for k in DROP_CANDIDATES})

# 採用した特徴量とnoneは feature_definitions.py にある(上の一覧は試した順のまま書いたので、ここで取り除く)
for _name in features.FEATURE_BUILDERS:
    REJECTED_FEATURE_BUILDERS.pop(_name, None)
