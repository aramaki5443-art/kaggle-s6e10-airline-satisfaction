"""スタックのメンバー(公開OOF+自分の5本)を名前から系統(TFM・TabPFN・NN・CatBoost・LightGBM・XGBoost・その他)に分ける。

目的: 系統ごとに外したときの寄与や、系統ごとの正則化(family-wise L2)で使う「どのメンバーがどの系統か」の表を作る。
ひとことで言うと: 264人の専門家を「出身の流派」ごとに名簿に分ける。
やり方: ファイル名のキーワード(tabpfn・tabfm・realmlp・cat・lgb・xgb等)で判定する。dariushafshar氏のgl_a〜nは
  manifest.csv(公開データセットの説明)の手法名で割り当てる。sadamt_e14_seed*はRealMLPのseed違い。
  結果をdata/pub_oof/families.jsonに保存する(系統名→メンバー名の一覧)。
使い方: python src/9_public_oof_stacking/group_public_oof_by_model.py
"""
import glob
import json
import os
import re

names = [os.path.basename(f)[4:-4] for folder in ["data/pub_oof/cdeotte124", "data/pub_oof/extra"]
         for f in glob.glob(f"{folder}/oof_*.npy")]
names = [n for n in names if n != "nj_01"] + ["rmlp", "lgb", "cat", "xgb", "tabpfn"]  # 末尾5本は自分のモデル
# dariushafshar氏のgl_a〜nの系統(manifest.csvの手法名から)
GOLEM_FAMILY = {"a": "lgb", "b": "xgb", "c": "cat", "d": "other", "e": "mlp", "f": "other", "g": "other", "h": "lgb", "i": "xgb",
                "j": "lgb", "k": "lgb", "l": "cat", "m": "lgb", "n": "mlp"}


def model_family(name):
    golem = re.fullmatch(r"gl_([a-n])", name)
    if golem:
        return GOLEM_FAMILY[golem.group(1)]
    lower = name.lower()
    if "tabpfn" in lower or lower.startswith("pfn_"):
        return "tabpfn"
    if any(k in lower for k in ["tabfm", "limix", "mitra", "kumo", "exaone", "causilo", "tabicl"]):
        return "tfm"
    if any(k in lower for k in ["realmlp", "rmlp", "mlp"]) or lower.startswith("sadamt_e14_seed"):
        return "mlp"
    if "tabm" in lower or "deepfm" in lower:
        return "other"
    if re.search(r"(^|_)(cat|cb|catboost)", lower):
        return "cat"
    if "lgb" in lower or "ghost" in lower:
        return "lgb"
    if "xgb" in lower or "xgboost" in lower:
        return "xgb"
    return "other"


members_by_family = {}
for name in names:
    members_by_family.setdefault(model_family(name), []).append(name)
json.dump(members_by_family, open("data/pub_oof/families.json", "w"), indent=1)
print({family: len(members) for family, members in members_by_family.items()})
