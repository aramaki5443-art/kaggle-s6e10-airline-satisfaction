"""スタックのメンバー(公開OOF+自分の5本)を名前から系統(TFM・TabPFN・NN・CatBoost・LightGBM・XGBoost・その他)に分ける。

目的: 系統ごとに外したときの寄与や、系統ごとの正則化(family-wise L2)で使う「どのメンバーがどの系統か」の表を作る。
ひとことで言うと: 264人の専門家を「出身の流派」ごとに名簿に分ける。
やり方: ファイル名のキーワード(tabpfn・tabfm・realmlp・cat・lgb・xgb等)で判定する。dariushafshar氏のgl_a〜nは
  manifest.csv(公開データセットの説明)の手法名で割り当てる。sadamt_e14_seed*はRealMLPのseed違い。
  結果をdata/pub_oof/families.jsonに保存する(系統名→メンバー名の一覧)。
使い方: python src/9_public_oof/pub_oof_families.py
"""
import glob
import json
import os
import re

names = [os.path.basename(f)[4:-4] for d in ["data/pub_oof/cdeotte124", "data/pub_oof/extra"] for f in glob.glob(f"{d}/oof_*.npy")]
names = [n for n in names if n != "nj_01"] + ["rmlp", "lgb", "cat", "xgb", "tabpfn"]  # 末尾5本は自分のモデル
GL = {"a": "lgb", "b": "xgb", "c": "cat", "d": "other", "e": "mlp", "f": "other", "g": "other", "h": "lgb", "i": "xgb",
      "j": "lgb", "k": "lgb", "l": "cat", "m": "lgb", "n": "mlp"}


def fam(n):
    m = re.fullmatch(r"gl_([a-n])", n)
    if m:
        return GL[m.group(1)]
    l = n.lower()
    if "tabpfn" in l or l.startswith("pfn_"):
        return "tabpfn"
    if any(k in l for k in ["tabfm", "limix", "mitra", "kumo", "exaone", "causilo", "tabicl"]):
        return "tfm"
    if any(k in l for k in ["realmlp", "rmlp", "mlp"]) or l.startswith("sadamt_e14_seed"):
        return "mlp"
    if "tabm" in l or "deepfm" in l:
        return "other"
    if re.search(r"(^|_)(cat|cb|catboost)", l):
        return "cat"
    if "lgb" in l or "ghost" in l:
        return "lgb"
    if "xgb" in l or "xgboost" in l:
        return "xgb"
    return "other"


F = {}
for n in names:
    F.setdefault(fam(n), []).append(n)
json.dump(F, open("data/pub_oof/families.json", "w"), indent=1)
print({k: len(v) for k, v in F.items()})
