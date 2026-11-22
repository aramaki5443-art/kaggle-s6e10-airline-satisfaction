# Kaggle Playground S6E10 — 航空会社の乗客満足度予測

乗客アンケートから「満足」と答えたかを当てるコンペで、元の特徴量だけのベースライン(Private LB 0.95910)から、特徴量エンジニアリング・5モデルのスタッキング・公開OOFの導入により **Private LB 0.96239(52位 / 3,748チーム、上位1.4%)** まで伸ばした分析の記録。

採用した施策だけでなく、**不採用にした施策とその判断の根拠**も各ノートブックに残している。

## プロジェクト概要

| 項目 | 内容 |
|---|---|
| コンペ | [Kaggle Playground Series S6E10「Predicting Airline Satisfaction」](https://www.kaggle.com/competitions/playground-series-s6e10)(2026年10月) |
| 課題 | 年齢・座席クラス・旅行目的・飛行距離・遅延・サービス評価13項目から、乗客が満足したかを当てる二値分類 |
| データ | 人工データ。train 699,635行・test 299,844行。満足の割合は約44% |
| 評価指標 | AUC(満足の客を不満の客より上位に並べられたか。1に近いほど良い) |
| 最終スコア | OOF AUC 0.962266 / Public LB 0.96182 / **Private LB 0.96239(52位 / 3,748チーム)** |
| 最終手法 | 自分の4モデル+点検に合格した公開OOF 222本を、2段目(ロジスティック回帰8割+浅いLightGBM2割)で混ぜたスタッキング |

## アプローチ

### 分析の進め方

EDAで客層ごとの偏りを調べたあと、LightGBMで特徴量を1つずつ足して判定した。特徴量の伸びが鈍った時点で、仕組みの違うモデル(ニューラルネット・TabPFN)を足し、最後に他の参加者が公開したOOFを点検してから加えた。ノートブックの番号は工程の順で、実際の順番(下のスコア推移の表)とは一部違う。

主な判断は次の4つ。

- **ブレを測ってから採否の基準を決めた:** fold分割の乱数(seed)を変えるだけで、特徴量の「あり−なし」の差が約0.00005揺れた。そこで3つのseedで判定する基準を設け、のちのハイパラ・公開OOFの採否にも同じ考え方を使った([03](notebooks/03_feature_engineering.ipynb))。
- **単体モデルの強化より、仕組みの違うモデルに時間を使った:** LightGBMのハイパラ調整で単体は+0.00041伸びたが、RealMLPと混ぜると+0.000046に縮んだ。5モデルのスタックで重く使われたのは、木と仕組みの違うTabPFN(係数0.42)とRealMLP(0.35)だった([05](notebooks/05_hyperparameter_tuning.ipynb)・[06](notebooks/06_ensemble.ipynb))。
- **公開OOFを点検してから導入した:** 予測値の平均・単体AUCなどで点検し、124本中1本を除外した。加えたときOOF(+0.00038)とPublic LB(+0.00043)が同じくらい伸び、甘いOOFによる崩れはなかった([07](notebooks/07_public_oof_stacking.ipynb))。
- **最終提出をCVで選んだ:** Public LB(約6万行)は0.0003未満の差を見分けられない。Public LBでのチームの順位は513位だったが、CVで選んだ提出でPrivate LBは52位になった。

### 各ノートブックの概要

| No. | ノートブック | 内容 | 主な成果 |
|---|---|---|---|
| 00 | [Baseline](notebooks/00_baseline.ipynb) | 元の特徴量21個を単変量AUCの高い順に1つずつ足し、LightGBMの土台を決める | 性別を除く20個。OOF 0.95846 |
| 01 | [EDA](notebooks/01_EDA.ipynb) | 全体像から生まれた2つの仮説(客層の違い・0点の意味)をEDAで確かめ、特徴量を作る | 「私用客は評価に関係なく満足率約10%」→ ゲート特徴量の着想 |
| 02 | [Preprocessing](notebooks/02_preprocessing.ipynb) | 欠損値の扱い、カテゴリ変数の渡し方、エンコーディングの比較 | 欠損(0.04%)は補完しない。カテゴリはcategory型のまま |
| 03 | [Feature Engineering](notebooks/03_feature_engineering.ipynb) | 採用した13種類の仮説・確認・コード、採否の基準、不採用169個の一覧、SHAP | OOF 0.95846 → 0.96046 |
| 04 | [Modeling](notebooks/04_modeling.ipynb) | XGBoost・CatBoost・RealMLP・TabPFNを作り、単体の強さと予測の似かたを比べる | 5モデルとも0.9604〜0.9612、木どうしだけが似ている |
| 05 | [Hyperparameter Tuning](notebooks/05_hyperparameter_tuning.ipynb) | Optunaで木の3モデルを調整し、混ぜたときの伸びを確かめる | 単体の伸びは、混ぜると約1/9に縮む |
| 06 | [Ensemble](notebooks/06_ensemble.ipynb) | 5モデルを2段目のロジスティック回帰で混ぜ、1モデルずつ抜いて役割を確かめる | OOF 0.961685 |
| 07 | [Public OOF Stacking](notebooks/07_public_oof_stacking.ipynb) | 公開OOFの点検・形の統一・メンバー選び・混ぜ方の調整、最終提出の選び方 | 226本。OOF 0.962266 |

### 特徴量エンジニアリングとアンサンブルの要点

- **特徴量:** 採用した16個は、①条件で絞る(出張客の快適さなど)、②似た乗客で数える・平均する(飛行距離の値ごとの満足率など)、③モデルで予想する(サービス評価の期待値)の3種類。最大の改善は飛行距離の値ごとの満足率(+0.00136)。不採用169個も含めた一覧は[03](notebooks/03_feature_engineering.ipynb)。
- **アンサンブル:** 5モデルの単体のOOF AUCは0.9608〜0.9612とほぼ同じだった。木どうしの順位相関は0.98〜0.99と高く、TabPFNはほかの4モデルと0.96台と低い。2段目の係数はTabPFN 0.42・RealMLP 0.35で、木と違う予測をするモデルが重く使われた([06](notebooks/06_ensemble.ipynb))。

## スコア推移

| 段階 | 施策 | OOF AUC | Private LB(上位%) |
|---|---|---:|---:|
| ベースライン | 元の特徴量20個のLightGBM | 0.95846 | 0.95910(76%) |
| 特徴量エンジニアリング | 16個を追加(36個)、3seedの採否基準 | 0.96046 | 0.96102(47%) |
| RealMLPの導入 | LightGBMと混ぜる | +0.000888(混ぜた伸び) | 0.96192(31%) |
| LightGBMのハイパラ調整 | Optuna 50試行 | +0.000046(混ぜた伸び) | 0.96189(32%) |
| 5モデルのスタック | XGBoost・CatBoost・TabPFNを追加 | 0.961685 | 0.96208(29%) |
| 公開OOFの導入 | 点検に合格した222本を追加、混ぜ方の調整 | 0.962266 | **0.96239(1.4%、52位)** |

## プロジェクト構成

```
kaggle-s6e10-airline-satisfaction/
├── notebooks/                          # 00〜07の8本(上の表)。1ノートブック1工程
├── src/
│   ├── features.py                     # データ読み込み・採用した特徴量の定義
│   ├── modeling.py                     # 交差検証・採用したハイパラ・Optuna・スタッキング
│   ├── public_oof.py                   # 公開OOFの点検・形の統一・系統分け
│   ├── kaggle_gpu/                     # KaggleのGPUで動かすCatBoostの学習・調整
│   └── *.csv                           # 公開OOFの取得先・最終メンバーの一覧
├── data/                               # コンペのデータ・予測(リポジトリには含めない)
├── pyproject.toml・uv.lock・.python-version
└── README.md
```

ノートブックは分析の過程と判断の根拠を書く場所、`src/`は複数のノートブックで使う処理を置く場所として分けている。

## 使用技術

Python 3.13(uv)、LightGBM・XGBoost・CatBoost(自分で実装)、RealMLP(pytabkit)・TabPFN(公開ノートブックの再現。コードは含めない)、Optuna、pandas・NumPy・scikit-learn・matplotlib・seaborn。CatBoost・RealMLP・TabPFNはKaggleのGPUで学習した。開発にはClaude Code(AIコーディングアシスタント)を使い、どの実験をするか・どの結果を採用するかは自分で判断した。

## 実行方法

1. クローンして環境を作る

```bash
git clone https://github.com/aramaki5443-art/kaggle-s6e10-airline-satisfaction.git
cd kaggle-s6e10-airline-satisfaction
uv sync
```

2. Kaggleのコンペページでルールに同意し、APIトークンを設定してから、データを `data/` に置く

```bash
uv run kaggle competitions download -c playground-series-s6e10 -p data
uv run python -m zipfile -e data/playground-series-s6e10.zip data/
```

3. ノートブックを番号順(00 → 07)に実行する。手元のCPUで、00は約70分、そのほかは各数分〜30分ほどかかる。

4. KaggleのGPUで作るモデル(CatBoost・RealMLP・TabPFN)のOOF・testの予測を `data/` に置く(04・06・07が読み込む)。CatBoostは、04が書き出す `data/features36_*.csv` を入力に `src/kaggle_gpu/train_catboost.py --run allcat_t21_cv42` で作る。RealMLP・TabPFNは公開ノートブック(04の4・5節に名前と変えた点)をもとに学習する。

5. 公開OOFは、`src/public_oof_sources.csv`・`src/late_public_oof_extraction.csv` の一覧から `data/pub_oof/` に取得し、06を `RUN_COLLECTION = True` で実行する(約3.5GB)。

GPUや並列計算では足し算の順番が実行ごとに変わるため、再実行するとOOF AUCが小数第5〜6位で少しずれることがある。

## 外部情報の扱い

公開ノートブック・Discussionのアイデアは、自分のコードで実装し、自分のCVで確かめてから採否を決めた(出典は各ノートブックの該当する節)。RealMLPとTabPFNは公開ノートブックをもとにしたため、コードは含めない。公開OOFは他の参加者の予測なので、ファイルは含めない。
