"""特徴量の定義: 生の列の一覧、特徴量の候補(FE)、採用した特徴量(ADOPTED)、Target Encoding(TE)。

目的: 特徴量の作り方を1か所にまとめ、どのモデルのスクリプトからも同じ特徴量を使えるようにする。
ひとことで言うと: 「材料の下ごしらえ」の手順書。学習や採点はしない(それは同じフォルダの lgb_check.py などが行う)。
やり方:
  - FE: 名前 → 関数(データを受け取り、足す列を返す)。候補は src/3_feature_engineering/lgb_check.py --fe 名前 で1つずつ判定する。
  - ADOPTED: 判定で採用した特徴量の名前。モデルのスクリプトは生の20列(BASE)+ADOPTEDで学習する。
  - TE: 値ごとの満足率の列。答えを使うので、FEでは元の値だけを置き、学習ループの中でfoldごとに満足率へ置き換える(te_fold)。
なぜ: LightGBM・XGBoost・ハイパラ探索で特徴量の定義が食い違わないようにするため。
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = [  # lgb_forward_add.pyのORDER[:20](Genderを除く生の20列)
    "Online boarding", "Class", "Inflight entertainment", "Inflight wifi service", "Seat comfort",
    "Flight Distance", "Type of Travel", "On-board service", "Cleanliness", "Leg room service", "Age",
    "Ease of Online booking", "Baggage handling", "Checkin service", "Food and drink", "Gate location",
    "Customer Type", "Departure/Arrival time convenient", "Arrival Delay in Minutes",
    "Departure Delay in Minutes",
]
CAT = ["Customer Type", "Type of Travel", "Class"]
SVC = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
       "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment", "On-board service",
       "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]  # サービス評価0〜5
PARAMS = dict(objective="binary", learning_rate=0.05, n_estimators=10000, random_state=42, verbose=-1)  # LightGBMの既定ハイパラ(元データの先生役でも使う。lgb_check.pyの--paramsで置き換わる)


def zero_flags(d):
    # 評価が0かどうかのフラグを項目ごとに作る(0が1件も無いBaggage handlingは除く)
    return pd.DataFrame({f"{c}_is0": (d[c] == 0).astype(int) for c in SVC if c != "Baggage handling"})


def zero_count(d):
    # 13項目のうち評価0の個数(複数の列にまたがる情報なので、木が1回の分岐では作れない)
    return pd.DataFrame({"svc_zero_count": (d[SVC] == 0).sum(axis=1)})


def count_score(k):
    # 13項目のうち評価がkの項目数(0点の個数=zero_countと同じ数え方)。5点の個数は満足、1点の個数は不満の多さ
    def f(d):
        return pd.DataFrame({f"svc_count{k}": (d[SVC] == k).sum(axis=1)})
    return f


def svc_mean(d):
    # 評価の平均。0は未利用・該当なしの項目があるため(EDA 7節)、0を除いて平均する
    return pd.DataFrame({"svc_mean": d[SVC].replace(0, np.nan).mean(axis=1)})


def svc_min(d):
    # 最も低い評価(0を除く)。1つでも強い不満があるかを表す
    return pd.DataFrame({"svc_min": d[SVC].replace(0, np.nan).min(axis=1)})


def svc_std(d):
    # 評価のばらつき(標準偏差、0を除く)。平均が同じでも「全部3点」と「5点と1点が混在」を区別する
    return pd.DataFrame({"svc_std": d[SVC].replace(0, np.nan).std(axis=1)})


def svc_std_business(d):
    # 出張客だけ評価の標準偏差、私用客は欠損(集計値にゲートを付けた型。ゲートなしは不採用)
    return pd.DataFrame({"svc_std_business": svc_std(d)["svc_std"].where(d["Type of Travel"] == "Business travel")})


def svc_max(d):
    return pd.DataFrame({"svc_max": d[SVC].max(axis=1)})


def noise_control(d):
    # 対照実験用: 満足度と無関係な乱数の列。「列を1本足すだけで成績がどれだけ動くか」の目安にする
    return pd.DataFrame({"noise": np.random.default_rng(0).random(len(d))})


COMFORT = ["Seat comfort", "Leg room service", "Inflight entertainment"]  # 機内の快適さ


def comfort(d):
    # 機内の快適さ = 座席・足元・機内エンタメの平均(0は未回答の可能性があるため除く)
    return d[COMFORT].replace(0, np.nan).mean(axis=1)


def dist_x_comfort(d):
    # 快適なら長距離ほど満足が上がり、不快なら上がらない(むしろ下がる)交互作用を1列で表す
    return pd.DataFrame({"dist_x_comfort": d["Flight Distance"] * comfort(d)})


def dist_div_comfort(d):
    # 比率の形。「長距離なのに不快」ほど大きくなる
    return pd.DataFrame({"dist_div_comfort": d["Flight Distance"] / comfort(d)})


def dist_x_comfort_business(d):
    # 出張客だけ飛行距離×快適さ、私用客は欠損(ゲートなしの掛け算は効かなかったため、ゲートと組み合わせて再確認)
    v = d["Flight Distance"] * comfort(d)
    return pd.DataFrame({"dist_x_comfort_business": v.where(d["Type of Travel"] == "Business travel")})


def dist_x_baggage(d):
    # 飛行距離×手荷物の評価(0なし)。距離5分位で手荷物の単変量AUCが0.570→0.789と最も大きく上がった項目
    return pd.DataFrame({"dist_x_baggage": d["Flight Distance"] * d["Baggage handling"]})


def dist_x_baggage_business(d):
    # 出張客だけ飛行距離×手荷物、私用客は欠損(全員の掛け算は-0.00005。私用客の値が混ざるのを除く)
    v = d["Flight Distance"] * d["Baggage handling"]
    return pd.DataFrame({"dist_x_baggage_business": v.where(d["Type of Travel"] == "Business travel")})


def baggage_longhaul(d):
    # 長距離(1189マイル超)だけ手荷物の評価、それ以外は欠損。距離との関係が段差型(1189マイル以下は手荷物の
    # 単変量AUC 0.60前後で横ばい、超えると0.72〜0.82)なので、掛け算ではなく段差の形で渡す
    return pd.DataFrame({"baggage_longhaul": d["Baggage handling"].where(d["Flight Distance"] > 1189)})


def comfort_mean(d):
    # ゲートなしの快適さ(comfort_gate_businessの改善がゲート由来かを切り分けるため)
    return pd.DataFrame({"comfort_mean": comfort(d)})


def comfort_gate_business(d):
    # Type of Travelをゲートにする: 出張客だけ快適さの値を持ち、私用客(快適さに関係なく満足率約10%)は欠損
    return pd.DataFrame({"comfort_business": comfort(d).where(d["Type of Travel"] == "Business travel")})


def comfort_gate_loyal_business(d):
    # 常連かつ出張客だけ快適さ、他は欠損。disloyal客(99.9%が出張客)は快適さが効かない(出張客内AUC 0.51、
    # 常連は0.80〜0.86、EDA 8節)ため、comfort_gate_businessから外す。comfort_gate_businessと置き換えて比べる
    m = (d["Type of Travel"] == "Business travel") & (d["Customer Type"] == "Loyal Customer")
    return pd.DataFrame({"comfort_loyal_business": comfort(d).where(m)})


def gate_loyal_business(col):
    # 常連かつ出張客だけ元の評価(0も含む)、他は欠損。快適さ(comfort_gate_loyal_business、採用)と同じ型を機内サービスの
    # 単項目に広げる。出張客内の単変量AUCが常連とdisloyalで大きく違う項目(清潔さ0.799/0.508・機内サービス0.811/0.632・
    # 食事0.682/0.510)を選んだ
    def f(d):
        m = (d["Type of Travel"] == "Business travel") & (d["Customer Type"] == "Loyal Customer")
        return pd.DataFrame({f"{col}_loyal_business": d[col].where(m)})
    return f


def svc3_gate_loyal_business(d):
    # 常連かつ出張客だけ、清潔さ・機内サービス・食事の平均(0を除く)、他は欠損。単項目のゲートは木が分岐で
    # 作れて不採用だったが、平均(集計値)は作りにくく、快適さの平均(comfort_gate_loyal_business)では効いたため同じ型で試す
    m = (d["Type of Travel"] == "Business travel") & (d["Customer Type"] == "Loyal Customer")
    v = d[["Cleanliness", "On-board service", "Food and drink"]].replace(0, np.nan).mean(axis=1)
    return pd.DataFrame({"svc3_loyal_business": v.where(m)})


def gate_mean_business(name, cols):
    # 出張客だけ、評価項目cols(0を除く)の平均、私用客は欠損。comfort_gate_businessの項目の選び方を見直すための版
    def f(d):
        v = d[cols].replace(0, np.nan).mean(axis=1)
        return pd.DataFrame({name: v.where(d["Type of Travel"] == "Business travel")})
    return f


# 出張客内の順位相関で、互いに0.54〜0.69と強く連動する「機内の環境」のまとまり(足元は機内サービス・手荷物側に近い)
CABIN = ["Inflight entertainment", "Seat comfort", "Cleanliness", "Food and drink"]

STAFF = ["On-board service", "Baggage handling", "Leg room service"]  # 「スタッフ・サービス」のまとまり(相関0.51〜0.63)


def gate_class_business(col):
    # Businessクラスだけ元の評価、Eco・Eco Plusは欠損。出張客内でもBusinessクラスとEcoで効き方が違う
    # (例: 足元の単変量AUC 0.730/0.553、EDA 8節)。Businessクラスの99%は出張客
    def f(d):
        return pd.DataFrame({f"{col}_bizclass": d[col].where(d["Class"] == "Business")})
    return f


def wifi_eco(d):
    # Eco・Eco PlusだけWi-Fiの評価(0も含む)、Businessクラスは欠損。Wi-Fiは他の項目と逆にEco側でよく効く
    # (単変量AUC Business 0.618・Eco 0.800・Eco Plus 0.819、EDA 8節)
    return pd.DataFrame({"wifi_eco": d["Inflight wifi service"].where(d["Class"] != "Business")})


def gate_where(name, col, gate_col, gate_val):
    # gate_colがgate_valの行だけcolの値、他は欠損(逆向きのゲート用。私用客・disloyal客だけのWi-Fi・オンライン予約等)
    def f(d):
        return pd.DataFrame({name: d[col].where(d[gate_col] == gate_val)})
    return f


def disloyal_age31_36(d):
    # disloyal客かつ31〜36歳のフラグ。この帯だけ満足率2〜4%と極端に低く(前後は13〜19%)、
    # ゲートの網羅チェックで唯一|z|>3の残差(予測が0.8ポイント高い)が残っていた
    m = (d["Customer Type"] == "disloyal Customer") & d["Age"].between(31, 36)
    return pd.DataFrame({"disloyal_age31_36": m.astype(int)})


# Target Encoding(TE): 値ごとの満足率を列にする。答え(satisfaction)を使うため、FE関数では元の値だけを
# 置いておき、学習ループの中でfoldごとに満足率へ置き換える(te_fold)。列名 -> (元の列, smoothingの強さm)
TE = {"te_distance": ("Flight Distance", 20),
      "te_distance_m5": ("Flight Distance", 5), "te_distance_m50": ("Flight Distance", 50), "te_distance_m1": ("Flight Distance", 1),
      "te_distance_m3": ("Flight Distance", 3), "te_distance_m10": ("Flight Distance", 10),
      "te_age": ("Age", 3),
      "te_dist_class": (["Flight Distance", "Class"], 3), "te_age_dist": (["Age", "Flight Distance"], 3),
      "te_dist_travel": (["Flight Distance", "Type of Travel"], 3),
      # 評価2つの組み合わせ(6×6=36通り)。78ペアのうち「組み合わせのTEの単変量AUC − 単独で強い方」が大きい上位3つ
      "te_wifi_gate": (["Inflight wifi service", "Gate location"], 3),
      "te_time_gate": (["Departure/Arrival time convenient", "Gate location"], 3),
      "te_wifi_time": (["Inflight wifi service", "Departure/Arrival time convenient"], 3),
      "te_depdelay": ("Departure Delay in Minutes", 3), "te_arrdelay": ("Arrival Delay in Minutes", 3),  # 欠損はTEで全体の満足率  # 回数(freq_dist_travel)と同じキーの満足率
      "te_svc13": (SVC, 3),
      "te_age_class": (["Age", "Class"], 3), "te_age_travel": (["Age", "Type of Travel"], 3),
      "te_ctc": (["Class", "Type of Travel", "Customer Type"], 3), "te_class_wifi": (["Class", "Inflight wifi service"], 3),
      "te_wifi_boarding": (["Inflight wifi service", "Online boarding"], 3)}  # 年齢×カテゴリ(不採用)  # 組み合わせのTE: 複数の列の値をつなげた1つのキーごとの満足率  # mの比較用


def te_key(name):
    def f(d):
        cols = TE[name][0]
        if isinstance(cols, list):  # 組み合わせは各列の値を文字列でつないだキーにする
            return pd.DataFrame({name: d[cols].fillna(-1).astype(str).agg("|".join, axis=1)})
        return pd.DataFrame({name: d[cols]})
    return f


def te_map(key_fit, y_fit, key_apply, m):
    # 値ごとの満足率を、人数が少ないほど全体の満足率に寄せて計算する(smoothing)。trainに無い値は全体の満足率
    p = y_fit.mean()
    g = pd.DataFrame({"k": key_fit, "y": y_fit}).groupby("k")["y"].agg(["sum", "count"])
    return pd.Series(key_apply).map((g["sum"] + m * p) / (g["count"] + m)).fillna(p).values


def te_fold(Xtr, ytr, others, seed):
    # 学習用の行は、その中でさらに5分割して自分を含まない満足率を入れる(自分の答えが混ざるのを防ぐ)。
    # 検証用・testの行(others)には、学習用の行全体の満足率を入れる
    Xtr = Xtr.copy()
    others = [o.copy() for o in others]
    for name, (_, m) in TE.items():
        if name not in Xtr:
            continue
        key = Xtr[name].values
        v = np.zeros(len(Xtr))
        for itr, iva in StratifiedKFold(5, shuffle=True, random_state=seed).split(key, ytr):
            v[iva] = te_map(key[itr], ytr[itr], key[iva], m)
        for o in others:
            o[name] = te_map(key, ytr, o[name].values, m)
        Xtr[name] = v
    return Xtr, others


def dist_cat(d):
    # Exact Value Encoding: 飛行距離の値そのものをカテゴリ(category型)として渡す。LightGBMは値ごとの満足率の順に
    # 並べ替えて分岐するため、TEと同じ「値ごとのクセ」を、満足率を自分で計算せずに木へ直接渡せる
    return pd.DataFrame({"dist_cat": d["Flight Distance"].astype("category")})


def freq(name, cols):
    # Frequency Encoding: colsの値(の組み合わせ)がtrain+testに何回出てくるか(目的変数は使わない)。
    # 残差との相関はほぼ0でも学習すると効くことがあるため、事前確認で落とさず学習で確かめる
    def f(d):
        # usecolsはCSVの列順で返すため、colsの順に並べ直してキーの文字列をそろえる
        both = pd.concat([pd.read_csv(fn, usecols=cols)[cols] for fn in ("data/train.csv", "data/test.csv")])
        cnt = both.fillna(-1).astype(str).agg("|".join, axis=1).value_counts()
        return pd.DataFrame({name: d[cols].fillna(-1).astype(str).agg("|".join, axis=1).map(cnt).values}, index=d.index)
    return f


def delay_combo(name):
    # 比例尺度どうし(遅延・飛行距離)の差・比・和。木は2列の差や比を何度も分岐して階段状に近似するしかないため、
    # 1列にまとめて渡す。単独の集計では「遅れを取り戻すと満足」は見えなかった(出発30分以上遅れた7,193人で31〜33%と横ばい)
    def f(d):
        dep, arr, dist = d["Departure Delay in Minutes"], d["Arrival Delay in Minutes"], d["Flight Distance"]
        v = {"delay_diff": arr - dep,  # マイナス=飛行中に遅れを取り戻した
             "arr_delay_per_dist": arr / dist,  # 距離のわりに遅れが大きいか
             "delay_ratio": (arr / dep).where(dep > 0),  # 出発遅延0では割れないため欠損
             "delay_sum": dep + arr}[name]
        return pd.DataFrame({name: v})
    return f


# 列を足さずに元の列を書き換える施策(--feで選ぶと、列を作る前にtrain・test・元データへ適用し、追加列は0本)
PREPROC = {
    # 到着遅延の欠損(train 292行・test 130行)を同じ人の出発遅延で埋める。欠損でない行の87%は到着=出発で、相関0.84
    "fill_arrdelay": lambda d: d.assign(**{"Arrival Delay in Minutes": d["Arrival Delay in Minutes"].fillna(d["Departure Delay in Minutes"])}),
}


_TEACHER = {}  # 元データで学習したモデル(設定ごとに1回だけ作って使い回す)
# 公開ノートブックの教師の設定。600本固定・深さ8・列の抽選0.5・正則化(L2 5、L1 0.05)。
# subsampleはsubsample_freqが無いとLightGBMでは効かないため、同ノートブックと同じく実質なし
TEACHER_PUB = dict(objective="binary", n_estimators=600, learning_rate=0.05, max_depth=8, min_child_samples=20,
                   subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.05, random_state=123, verbose=-1)


def orig_teacher_fe(name, params=None):
    # 元データ(ノイズが少なく、生の20列だけでAUC 0.995)だけで学習したLightGBMの予測を1列として渡す。元データを
    # 行として足す方法は悪化したが、公開ノートブック3件で予測を列にする方法は
    # +0.00055〜+0.00072。教師モデルはコンペのラベルを一切見ないため、foldの外で作ってもリークにならない。
    # params=None: 既定ハイパラ(PARAMS)で、元データの20%で早期終了して本数を決め全体に学習し直す
    def f(d):
        if name not in _TEACHER:
            o = pd.concat([pd.read_csv(f"data/original/{f}.csv") for f in ("train", "test")], ignore_index=True)
            yo = (o["satisfaction"] == "satisfied").astype(int).values  # 元データは文字列のラベル
            Xo = o[BASE].copy()
            for c in CAT:
                Xo[c] = Xo[c].astype("category")
            if params is None:
                tr, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(Xo, yo))
                m = lgb.LGBMClassifier(**PARAMS).fit(Xo.iloc[tr], yo[tr], eval_set=[(Xo.iloc[va], yo[va])], eval_metric="auc",
                                                     callbacks=[lgb.early_stopping(100, verbose=False)])
                print(f"教師モデル: 元データ内AUC {roc_auc_score(yo[va], m.predict_proba(Xo.iloc[va])[:, 1]):.4f}、木{m.best_iteration_}本")
                m = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": m.best_iteration_}).fit(Xo, yo)
            else:
                m = lgb.LGBMClassifier(**params).fit(Xo, yo)
            _TEACHER[name] = (m, Xo.dtypes)
        m, dt = _TEACHER[name]
        X = d[BASE].copy()
        for c in CAT:
            X[c] = pd.Categorical(X[c], categories=dt[c].categories)
        return pd.DataFrame({name: m.predict_proba(X)[:, 1]}, index=d.index)
    return f


def orig_rate(d, m=10, cols=None):
    # 生の20列それぞれについて、元データでの「その値の人の満足率」を1列ずつ渡す(+20列、公開ノートブックの`org`)。
    # 先生役(1つのモデルの確信)と違い、列ごとの単純な平均なので強い確信を持ち込みにくい。先生役と同じくPublicで
    # 下がるかを確かめる。少数の値の極端な率を抑えるため全体の満足率へm件分寄せる。元データに無い値は欠損
    o = pd.concat([pd.read_csv(f"data/original/{f}.csv") for f in ("train", "test")], ignore_index=True)
    yo = (o["satisfaction"] == "satisfied").astype(int)
    out = {}
    for c in cols or BASE:
        s = yo.groupby(o[c]).agg(["sum", "count"])
        rate = (s["sum"] + m * yo.mean()) / (s["count"] + m)
        out[f"org_{c}"] = d[c].map(rate).astype(float)
    return pd.DataFrame(out, index=d.index)
# 26列の上では20列版は悪化。評価・カテゴリは値の種類が少なく木が元の列で分けられ、飛行距離は既存のTEと重なるため、
# 値の種類が多く既存のFEと重ならない年齢・遅延の3列だけに絞る
ORIG_RATE_NUM3 = ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def route_mean(name, col):
    # ルートプロファイル(公開ノートブック・Discussion): 飛行距離の値を「路線」とみなし、train+testで同じ距離の乗客の
    # colの平均を列にする(目的変数は使わない)。距離のTE(採用済み)は路線の満足率そのものだが、こちらは路線の
    # 「乗客の顔ぶれ・評価の傾向」で、少人数の路線でもラベルのぶれを含まない。路線の人数(=freq_dist)は不採用済み
    def f(d):
        both = pd.concat([pd.read_csv(fn, usecols=["Flight Distance", col]) for fn in ("data/train.csv", "data/test.csv")])
        return pd.DataFrame({name: d["Flight Distance"].map(both.groupby("Flight Distance")[col].mean()).values}, index=d.index)
    return f


def eq_flag(name, pairs):
    # 評価の一致フラグ(4種とも不採用): 2つの評価の点数が等しいか。時刻の便利さとゲートの場所は、点数が等しいと満足率54〜69%、
    # 違うと7〜36%。木は生の2列から『等しい』を作るのに対角線を何度も分ける必要があるので、1列で渡す。
    # pairsが複数なら、等しいペアの数
    def f(d):
        return pd.DataFrame({name: sum((d[a] == d[b]).astype(int) for a, b in pairs)}, index=d.index)
    return f


def route_share(name, col, val):
    # 路線(飛行距離の値)ごとの、colがvalである乗客の割合(train+test、目的変数不使用)(3つとも不採用)。距離×Class・距離×旅行目的のTE・回数は
    # 採用済みだが、こちらは路線の客層(ビジネス客の多い路線か)
    def f(d):
        both = pd.concat([pd.read_csv(fn, usecols=["Flight Distance", col]) for fn in ("data/train.csv", "data/test.csv")])
        share = (both[col] == val).groupby(both["Flight Distance"]).mean()
        return pd.DataFrame({name: d["Flight Distance"].map(share).values}, index=d.index)
    return f


def route_std(name, col):
    # 路線ごとの評価の標準偏差(train+test、Wi-Fiを採用)。路線平均(rp_)が同じでも、評価がそろった路線とばらけた路線を区別する
    def f(d):
        both = pd.concat([pd.read_csv(fn, usecols=["Flight Distance", col]) for fn in ("data/train.csv", "data/test.csv")])
        return pd.DataFrame({name: d["Flight Distance"].map(both.groupby("Flight Distance")[col].std()).values}, index=d.index)
    return f


def ctx_dev(name, keys, nums=None, z=False):
    # 背景グループ統計(公開ノートブックでClass×旅行目的×顧客タイプ版を採用)の「行の値−グループ平均」だけを取り出した版。keysで乗客をグループ分けし
    # (例: Class×旅行目的×顧客タイプの12グループ)、年齢・距離・遅延2列がグループの中で高いか低いかを4列で渡す。
    # グループ平均・標準偏差そのものはグループ(=既存の列の組み合わせ)ごとに一定で木が作れるため、ここでは偏差だけ。
    # 平均はtrain+testで計算(目的変数は使わない)
    # nums: 偏差を取る列(既定は年齢・距離・遅延2列)。z=True: グループの標準偏差で割る(グループごとに散らばりが違っても同じ物差しにする、不採用)
    nums = nums or {"age": "Age", "dist": "Flight Distance", "dep": "Departure Delay in Minutes", "arr": "Arrival Delay in Minutes"}
    def f(d):
        both = pd.concat([pd.read_csv(fn, usecols=keys + list(nums.values())) for fn in ("data/train.csv", "data/test.csv")])
        out = {}
        for short, c in nums.items():
            st = both.groupby(keys)[c].agg(m="mean", s="std")
            g = d[keys].merge(st, left_on=keys, right_index=True, how="left")
            out[f"{name}_{short}"] = (d[c].values - g["m"].values) / (g["s"].values if z else 1)
        return pd.DataFrame(out, index=d.index)
    return f


def route_zero(name, col, kind):
    # 路線の0の割合(kind="zero")・0を除いた路線平均(kind="nz")。評価の0は「該当なし」に近く(0の人の満足率はWi-Fi 88.7%・
    # オンライン予約70.8%・搭乗62.0%と全体44%から外れる)、0込みの路線平均(route_mean)では「0が多い路線」と「評価が低い路線」が
    # 混ざる。分けて渡せば、路線の利用のされ方(0の割合)と評価の高さ(0除外平均)を別々に使える。公開ノートブックは全て0込み
    def f(d):
        both = pd.concat([pd.read_csv(fn, usecols=["Flight Distance", col]) for fn in ("data/train.csv", "data/test.csv")])
        v = (both[col] == 0).astype(float) if kind == "zero" else both[col].replace(0, np.nan)
        return pd.DataFrame({name: d["Flight Distance"].map(v.groupby(both["Flight Distance"]).mean()).values}, index=d.index)
    return f


def aux_ev(name):
    # 期待値の列(公開ノートブックのアイデア): make_expected_value.pyで前もって作った「他の列から予想したこの項目の点数」(train+testの順に保存)。
    # trainとtestで行数が違うので、渡された行数で切り分ける
    def f(d):
        ev = np.load(f"data/aux_ev_{name[3:]}.npy")
        n_tr = len(pd.read_csv("data/train.csv", usecols=["id"]))
        return pd.DataFrame({name: ev[:n_tr] if len(d) == n_tr else ev[n_tr:]}, index=d.index)
    return f


def route_diff(name, col):
    # 自分の評価−路線平均(公開ノートブックのアイデア): 路線平均(route_mean、rp_legroomで採用)は路線の乗客の傾向だが、こちらは
    # 「この路線の乗客の中で、この人は甘いか辛いか」。木は生の評価と路線平均の2列を分岐で引き算できないので新しい情報になりうる
    def f(d):
        m = route_mean("_m", col)(d)["_m"]
        return pd.DataFrame({name: d[col].values - m.values}, index=d.index)
    return f


def digits(name, col):
    # 桁の特徴量(公開ノートブックのアイデア): 値の各桁 floor(x/10^k)%10(k=0〜3、値が1種類しかない桁は除く)。人工データの生成で
    # 「末尾が0の距離」のような桁のクセが残っていれば、木が生の値の大小の分岐では作れない区別(例: 末尾が0の値をまとめる)を渡せる。
    # 欠損(到着遅延)は欠損のまま(0で埋めると遅延0分と混ざるため)
    def f(d):
        v = d[col].astype(float)
        out = {f"{name}_d{k}": np.floor(v / 10 ** k) % 10 for k in range(4)}
        return pd.DataFrame({k: x for k, x in out.items() if x.nunique() > 1}, index=d.index)
    return f


def digit1(name, col, k):
    # 桁を1つだけ足す版(元の列ごとにまとめて入れたら全て不採用だったため、効く桁が他の桁の入場料で
    # 打ち消されていないかを1桁ずつ確かめる)。定義はdigitsと同じ
    def f(d):
        return pd.DataFrame({name: np.floor(d[col].astype(float) / 10 ** k) % 10}, index=d.index)
    return f


def gpt2_token(name, col):
    # GPT-2トークンキー(公開ノートブックのアイデア): 値を' '+文字列にしてGPT-2の区切り方でトークンに分ける(例: 694→' 6'+'94'、460→' 460')。
    # 人工データの生成器が数値を文字のかたまりごとに作っていれば、区切れ方のクセ(例: 末尾が'94'の値)が満足率に残る可能性がある。
    # 列: 先頭トークンID・末尾トークンID+100000×トークン数・トークン数(投稿者の定義どおり)
    def f(d):
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        v = d[col]
        toks = {u: enc.encode(" " + str(int(u))) for u in v.dropna().unique()}  # 値の種類ごとに1回だけ区切る
        first = v.map(lambda u: toks[u][0] if u in toks else np.nan)
        n = v.map(lambda u: len(toks[u]) if u in toks else np.nan)
        last = v.map(lambda u: toks[u][-1] + 100000 * len(toks[u]) if u in toks else np.nan)
        return pd.DataFrame({f"{name}_first": first, f"{name}_last": last, f"{name}_n": n}, index=d.index).astype(float)
    return f


def age_div_dist(d):
    # 年齢÷飛行距離(不採用)。到着遅延÷距離は保留(+0.000018)
    return pd.DataFrame({"age_div_dist": d["Age"] / d["Flight Distance"]}, index=d.index)


def none(d):
    # 何も足さない(ADOPTEDだけのベースラインを、--cv-seedで別のfold分割でも学習するため)
    return pd.DataFrame(index=d.index)


def gender(d):
    # 生のGender列を戻す。前向き追加では効かなかったが、当時は21列。TEの事前確認で今のモデルの残差との
    # 相関+0.0037(組み合わせのTEと同程度)だったため、事前確認で落とさず学習で確かめる
    return pd.DataFrame({"Gender": d["Gender"].astype("category")})


def gate_business(col):
    # 出張客だけ元の評価(0も含む)、私用客は欠損。私用客ではほとんどの評価項目が満足度にほぼ効かない
    # (出張客内の単変量AUC 0.70〜0.79に対し私用客内は0.51前後)ため、その区別を1列で渡す
    def f(d):
        return pd.DataFrame({f"{col}_business": d[col].where(d["Type of Travel"] == "Business travel")})
    return f


FE = {"none": none, "ev_boarding": aux_ev("ev_boarding"), "ev_legroom": aux_ev("ev_legroom"), **{f"ev_{n}": aux_ev(f"ev_{n}") for n in ("checkin", "onboard", "baggage", "food", "wifi", "clean", "seat", "booking", "ent", "gate", "time", "class", "travel", "cust")}, **{f"{k}_{n}": route_zero(f"{k}_{n}", c, kind) for n, c in (
           ("wifi", "Inflight wifi service"), ("booking", "Ease of Online booking"), ("boarding", "Online boarding"))
           for k, kind in (("rz", "zero"), ("rnz", "nz"))},
      "tok_dist": gpt2_token("tok_dist", "Flight Distance"), **{f"dg1_{n}_d{k}": digit1(f"dg1_{n}_d{k}", c, k) for n, c, ks in (
           ("dist", "Flight Distance", range(4)), ("age", "Age", range(2)), ("depdelay", "Departure Delay in Minutes", range(3)),
           ("arrdelay", "Arrival Delay in Minutes", range(3))) for k in ks},
      **{n: digits(n, c) for n, c in (("dg_dist", "Flight Distance"), ("dg_age", "Age"), ("dg_depdelay", "Departure Delay in Minutes"), ("dg_arrdelay", "Arrival Delay in Minutes"))}, "rpd_legroom": route_diff("rpd_legroom", "Leg room service"), "rp_baggage": route_mean("rp_baggage", "Baggage handling"), "rp_legroom": route_mean("rp_legroom", "Leg room service"),
      "rp_onboard": route_mean("rp_onboard", "On-board service"), **{n: route_mean(n, c) for n, c in (
          ("rp_wifi", "Inflight wifi service"), ("rp_time", "Departure/Arrival time convenient"), ("rp_booking", "Ease of Online booking"),
          ("rp_gate", "Gate location"), ("rp_food", "Food and drink"), ("rp_boarding", "Online boarding"), ("rp_seat", "Seat comfort"),
          ("rp_ent", "Inflight entertainment"), ("rp_checkin", "Checkin service"), ("rp_clean", "Cleanliness"), ("rp_age", "Age"),
          ("rp_depdelay", "Departure Delay in Minutes"), ("rp_arrdelay", "Arrival Delay in Minutes"))},
      "orig_rate": orig_rate, "orig_rate_num3": lambda d: orig_rate(d, cols=ORIG_RATE_NUM3), "orig_teacher": orig_teacher_fe("orig_teacher"), "orig_teacher_pub": orig_teacher_fe("orig_teacher_pub", TEACHER_PUB), **{k: none for k in PREPROC}, **{n: delay_combo(n) for n in ("delay_diff", "arr_delay_per_dist", "delay_ratio", "delay_sum")}, "noise_control": noise_control, "gender": gender, "freq_dist_travel": freq("freq_dist_travel", ["Flight Distance", "Type of Travel"]),
      "freq_dist": freq("freq_dist", ["Flight Distance"]), "freq_age": freq("freq_age", ["Age"]),
      "freq_depdelay": freq("freq_depdelay", ["Departure Delay in Minutes"]), "freq_arrdelay": freq("freq_arrdelay", ["Arrival Delay in Minutes"]),
      "freq_dist_class": freq("freq_dist_class", ["Flight Distance", "Class"]), "freq_age_dist": freq("freq_age_dist", ["Age", "Flight Distance"]),
      "freq_svc13": freq("freq_svc13", SVC), "freq_age_class_travel": freq("freq_age_class_travel", ["Age", "Class", "Type of Travel"]),
      "freq_boarding": freq("freq_boarding", ["Online boarding"]),
      "freq_dist_cust": freq("freq_dist_cust", ["Flight Distance", "Customer Type"]),
      "freq_dist_class_travel": freq("freq_dist_class_travel", ["Flight Distance", "Class", "Type of Travel"]),
      "freq_dist_travel_cust": freq("freq_dist_travel_cust", ["Flight Distance", "Type of Travel", "Customer Type"]),
      "gate_onboard": gate_business("On-board service"), "loyal_clean": gate_loyal_business("Cleanliness"), "loyal_onboard": gate_loyal_business("On-board service"), "loyal_food": gate_loyal_business("Food and drink"), "svc3_loyal_business": svc3_gate_loyal_business, "gate_clean": gate_business("Cleanliness"),
      "gate_baggage": gate_business("Baggage handling"), "gate_checkin": gate_business("Checkin service"),
      "gate_ent": gate_business("Inflight entertainment"), "gate_seat": gate_business("Seat comfort"),
      "gate_legroom": gate_business("Leg room service"), "gate_food": gate_business("Food and drink"),
      "gate_boarding": gate_business("Online boarding"),
      "cls_onboard": gate_class_business("On-board service"), "cls_legroom": gate_class_business("Leg room service"),
      "cls_baggage": gate_class_business("Baggage handling"), "cls_checkin": gate_class_business("Checkin service"), "wifi_eco": wifi_eco, "dist_cat": dist_cat, "te_distance": te_key("te_distance"),
      "te_distance_m5": te_key("te_distance_m5"), "te_distance_m50": te_key("te_distance_m50"), "te_distance_m1": te_key("te_distance_m1"), "te_distance_m3": te_key("te_distance_m3"), "te_distance_m10": te_key("te_distance_m10"), "te_age": te_key("te_age"), "te_dist_class": te_key("te_dist_class"), "te_dist_travel": te_key("te_dist_travel"), "te_wifi_gate": te_key("te_wifi_gate"), "te_time_gate": te_key("te_time_gate"), "te_wifi_time": te_key("te_wifi_time"), "te_depdelay": te_key("te_depdelay"), "te_arrdelay": te_key("te_arrdelay"),
      "te_age_dist": te_key("te_age_dist"), "te_svc13": te_key("te_svc13"), "disloyal_age31_36": disloyal_age31_36,
      "wifi_personal": gate_where("wifi_personal", "Inflight wifi service", "Type of Travel", "Personal Travel"),
      "booking_personal": gate_where("booking_personal", "Ease of Online booking", "Type of Travel", "Personal Travel"),
      "wifi_disloyal": gate_where("wifi_disloyal", "Inflight wifi service", "Customer Type", "disloyal Customer"),
      "booking_disloyal": gate_where("booking_disloyal", "Ease of Online booking", "Customer Type", "disloyal Customer"),
      "cabin_gate_business": gate_mean_business("cabin_business", CABIN),
      "cabin_staff_gate_business": gate_mean_business("cabin_staff_business", CABIN + STAFF), "dist_x_comfort": dist_x_comfort, "dist_div_comfort": dist_div_comfort,
      "comfort_gate_business": comfort_gate_business, "comfort_mean": comfort_mean, "comfort_gate_loyal_business": comfort_gate_loyal_business,
      "dist_x_comfort_business": dist_x_comfort_business, "dist_x_baggage": dist_x_baggage, "baggage_longhaul": baggage_longhaul, "dist_x_baggage_business": dist_x_baggage_business, "zero_flags": zero_flags, "zero_count": zero_count, "svc_mean": svc_mean, "svc_min": svc_min,
      "svc_max": svc_max, "svc_std": svc_std, "svc_std_business": svc_std_business, "count5": count_score(5), "count1": count_score(1)}
FE.update({"ctx_ctc": ctx_dev("ctx_ctc", ["Class", "Type of Travel", "Customer Type"]),
           "ctx_bct": ctx_dev("ctx_bct", ["Online boarding", "Class", "Type of Travel"]),
           "ctx_wct": ctx_dev("ctx_wct", ["Inflight wifi service", "Class", "Type of Travel"])})
FE.update({"eq_time_gate": eq_flag("eq_time_gate", [("Departure/Arrival time convenient", "Gate location")]),
           "eq_wifi_time": eq_flag("eq_wifi_time", [("Inflight wifi service", "Departure/Arrival time convenient")]),
           "eq_wifi_gate": eq_flag("eq_wifi_gate", [("Inflight wifi service", "Gate location")]),
           "eq_count4": eq_flag("eq_count4", [("Inflight wifi service", "Departure/Arrival time convenient"), ("Inflight wifi service", "Ease of Online booking"), ("Inflight wifi service", "Gate location"), ("Departure/Arrival time convenient", "Ease of Online booking"), ("Departure/Arrival time convenient", "Gate location"), ("Ease of Online booking", "Gate location")]),
           "rs_business": route_share("rs_business", "Class", "Business"),
           "rs_travel": route_share("rs_travel", "Type of Travel", "Business travel"),
           "rs_loyal": route_share("rs_loyal", "Customer Type", "Loyal Customer"),
           "rsd_legroom": route_std("rsd_legroom", "Leg room service"),
           "rsd_boarding": route_std("rsd_boarding", "Online boarding"),
           "rsd_wifi": route_std("rsd_wifi", "Inflight wifi service"),
           "te_age_class": te_key("te_age_class"), "te_age_travel": te_key("te_age_travel")})
RSD_REST = {"rsd_booking": "Ease of Online booking", "rsd_time": "Departure/Arrival time convenient", "rsd_gate": "Gate location",
            "rsd_food": "Food and drink", "rsd_seat": "Seat comfort", "rsd_ent": "Inflight entertainment", "rsd_onboard": "On-board service",
            "rsd_baggage": "Baggage handling", "rsd_checkin": "Checkin service", "rsd_clean": "Cleanliness"}  # 残り10項目の路線標準偏差(全て不採用、予約は保留)
FE.update({k: route_std(k, c) for k, c in RSD_REST.items()})
FE.update({"rsd_age": route_std("rsd_age", "Age"), "rsd_dep": route_std("rsd_dep", "Departure Delay in Minutes"),
           "rsd_arr": route_std("rsd_arr", "Arrival Delay in Minutes"),  # 年齢・遅延の路線標準偏差(不採用)。客層内偏差の拡張・組み合わせTE・年齢÷距離も不採用
           "ctx_ctc_wifi": ctx_dev("ctx_ctc_wifi", ["Class", "Type of Travel", "Customer Type"], {"v": "Inflight wifi service"}),
           "ctx_ctc_boarding": ctx_dev("ctx_ctc_boarding", ["Class", "Type of Travel", "Customer Type"], {"v": "Online boarding"}),
           "ctx_ctc_z": ctx_dev("ctx_ctc_z", ["Class", "Type of Travel", "Customer Type"], z=True),
           "te_ctc": te_key("te_ctc"), "te_class_wifi": te_key("te_class_wifi"), "te_wifi_boarding": te_key("te_wifi_boarding"),
           "age_div_dist": age_div_dist})


def route_class_std(name, col):
    # 路線×Classごとの評価の標準偏差(追加候補B、不採用)。同じ路線でもBusinessとEcoで客層が違うため細かく分ける。人数が減る分ぶれやすい
    def f(d):
        k = ["Flight Distance", "Class"]
        both = pd.concat([pd.read_csv(fn, usecols=k + [col]) for fn in ("data/train.csv", "data/test.csv")])
        st = both.groupby(k)[col].std().rename("s")
        return pd.DataFrame({name: d[k].merge(st, left_on=k, right_index=True, how="left")["s"].values}, index=d.index)
    return f


FE.update({"rs_wifi5": route_share("rs_wifi5", "Inflight wifi service", 5),  # 追加候補A: 路線のWi-Fi 5点の割合(rsd_wifiと順位相関0.60)
           "rcsd_wifi": route_class_std("rcsd_wifi", "Inflight wifi service"),  # 追加候補B
           # 追加候補C: 路線平均との差(年齢・遅延)。足元の評価の差は不採用、客層グループ内の差(ctx_ctc)は採用
           "rdev_age": ctx_dev("rdev_age", ["Flight Distance"], {"v": "Age"}),
           "rdev_dep": ctx_dev("rdev_dep", ["Flight Distance"], {"v": "Departure Delay in Minutes"}),
           "rdev_arr": ctx_dev("rdev_arr", ["Flight Distance"], {"v": "Arrival Delay in Minutes"})})
# 追加候補D: 飛行距離を10刻み・100刻みに丸めた値のTE(公開ノートブック、不採用)。人数の少ない路線を近い距離とまとめ、満足率のぶれを抑える。
# 丸めたキーはFE関数で作り、TE辞書に名前を登録して学習ループ(te_fold)で満足率に置き換える
TE.update({"te_dist10": (None, 3), "te_dist100": (None, 3)})
FE.update({"te_dist10": lambda d: pd.DataFrame({"te_dist10": d["Flight Distance"] // 10}, index=d.index),
           "te_dist100": lambda d: pd.DataFrame({"te_dist100": d["Flight Distance"] // 100}, index=d.index)})
DROP = {"drop_gate": ["Gate location"], "drop_dep": ["Departure Delay in Minutes"], "drop_arr": ["Arrival Delay in Minutes"]}  # 足すのではなく、生の列を外して効果を見る候補(公開ノートブック、ゲート・遅延2列とも不採用)
FE.update({k: none for k in DROP})
ADOPTED = ["comfort_gate_business", "gate_onboard", "te_distance_m3", "te_dist_class", "freq_dist_travel", "comfort_gate_loyal_business", "rp_legroom", "ev_boarding", "ev_legroom", "ev_checkin", "ev_time", "ctx_ctc", "rsd_wifi"]  # 採用済みのFE(3seed再確認後。先生役orig_teacher_pubはOOFで改善するがPublic LBで3構成とも悪化したため外した)
