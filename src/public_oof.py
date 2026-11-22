"""公開OOF(他の参加者が公開した、OOFとtestの予測のセット)の点検・形の統一・系統分け。notebooks/07_public_oof_stacking から使う。

公開OOFのファイル自体は他の参加者の成果物なので、このリポジトリには含めない。取得先は public_oof_sources.csv・late_public_oof_extraction.csv。
そろえた後の形は <フォルダ>/oof_<名前>.npy・test_<名前>.npy(行の順は train.csv・test.csv と同じ)。
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from modeling import to_logit


def inspect_public_oof(folder, y, n_test, own_oof, suspicious_auc=0.9617):
    """フォルダ内の公開OOFを1本ずつ点検した表を返す: 行数・欠損・値の範囲・単体AUC・自分のモデルとの順位相関・testの平均。
    単体AUCが自分の5モデルのスタック(suspicious_auc)より高いものは、それ自体が答えを使って混ぜた結果の可能性があるので「要確認」にする。"""
    sample_rows = np.random.default_rng(0).choice(len(y), 100000, replace=False)  # 相関は10万行で計算(速さのため)
    rows = []
    for oof_file in sorted(glob.glob(f"{folder}/oof_*.npy")):
        name = os.path.basename(oof_file)[4:-4]
        oof = np.load(oof_file).astype(float).ravel()
        test_file = f"{folder}/test_{name}.npy"
        test = np.load(test_file).astype(float).ravel() if os.path.exists(test_file) else None
        row = {"name": name, "n": len(oof), "nan": int(np.isnan(oof).sum()), "min": oof.min(), "max": oof.max(),
               "oof_mean": float(np.nanmean(oof)), "test_n": None if test is None else len(test),
               "test_mean": None if test is None else float(np.nanmean(test))}
        if len(oof) == len(y) and row["nan"] == 0:
            row["auc"] = roc_auc_score(y, oof)
            row["correlation_with_own_model"] = spearmanr(oof[sample_rows], own_oof[sample_rows])[0]
        rows.append(row)
    result = pd.DataFrame(rows)
    shape_ok = (result["n"] == len(y)) & (result["nan"] == 0) & (result["test_n"] == n_test)
    result["judgement"] = np.where(~shape_ok, "形が合わない", np.where(result["auc"] > suspicious_auc, "要確認(高すぎ)", "OK"))
    return result


def sort_by_id(df, first_id):
    """id列があれば並べ替え、train・testと同じ行の順になっているかを確かめる。"""
    if "id" in df.columns:
        df = df.sort_values("id")
        assert (df["id"].values == np.arange(first_id, first_id + len(df))).all()
    return df


def read_raw_public_oof(raw_dir, n_train, named_pairs=()):
    """取得したままの公開OOF(データセットごとにnpy・npz・parquet・csvと形が違う)を読み、(名前, OOF, test, 出典)の一覧にする。
    名前の衝突を防ぐため、投稿者名の先頭6文字を接頭辞にする。named_pairs は「名前=OOFのファイル=testのファイル」で個別に指定する組。"""
    candidates = []
    for folder in sorted(glob.glob(f"{raw_dir}/*/")):
        dataset_name = os.path.basename(folder.rstrip("/\\"))
        prefix = dataset_name.removeprefix("nb_").split("_")[0][:6]
        for oof_file in sorted(glob.glob(f"{folder}oof_*.npy")):
            name = os.path.basename(oof_file)[4:-4]
            if os.path.exists(f"{folder}test_{name}.npy"):
                candidates.append((f"{prefix}_{name}", np.load(oof_file), np.load(f"{folder}test_{name}.npy"), dataset_name))
        for npz_file in sorted(glob.glob(f"{folder}*.npz")):
            npz = np.load(npz_file)
            base_name = os.path.basename(npz_file)[:-4].removeprefix("preds_")
            for oof_key, test_key in (("oof", "test"), ("train", "test")):
                if {oof_key, test_key} <= set(npz.files):
                    candidates.append((f"{prefix}_{base_name}", npz[oof_key], npz[test_key], dataset_name))
                    break
            else:  # 1つのnpzに oof_X・test_X が並ぶ形
                candidates += [(f"{prefix}_{k[4:]}", npz[k], npz["test_" + k[4:]], dataset_name)
                               for k in npz.files if k.startswith("oof_") and "test_" + k[4:] in npz.files]
        for oof_file, test_file, read_table in ((f"{folder}oof_predictions.parquet", f"{folder}test_predictions.parquet", pd.read_parquet),
                                                (f"{folder}oof_members.csv", f"{folder}test_members.csv", pd.read_csv)):
            if os.path.exists(oof_file):  # 列ごとに1モデルの表
                oof_table, test_table = sort_by_id(read_table(oof_file), 0), sort_by_id(read_table(test_file), n_train)
                candidates += [(f"{prefix}_{c}", oof_table[c].values, test_table[c].values, dataset_name)
                               for c in oof_table.columns if c in test_table.columns and c != "id"]
        csv_pairs = [(f, f"{folder}test_{os.path.basename(f)[4:]}", os.path.basename(f)[4:-4]) for f in glob.glob(f"{folder}oof_*.csv")]
        csv_pairs += [(f, f[:-8] + "_test.csv", os.path.basename(f)[:-8]) for f in glob.glob(f"{folder}*_oof.csv")]
        for oof_file, test_file, name in sorted(csv_pairs):
            if os.path.exists(test_file):
                oof_table, test_table = sort_by_id(pd.read_csv(oof_file), 0), sort_by_id(pd.read_csv(test_file), n_train)
                candidates.append((f"{prefix}_{name}", oof_table.iloc[:, -1].values, test_table.iloc[:, -1].values, dataset_name))
    for pair in named_pairs:
        name, oof_file, test_file = pair.split("=")
        oof, test = [np.load(f) if f.endswith(".npy") else sort_by_id(pd.read_csv(f), first_id).iloc[:, -1].values
                     for f, first_id in ((oof_file, 0), (test_file, n_train))]
        candidates.append((name, oof, test, os.path.basename(os.path.dirname(oof_file))))
    return candidates


def save_unique_public_oof(candidates, reference_dirs, out_dir, n_train, n_test, duplicate_correlation=0.9999):
    """候補を同じ形に書き直して out_dir に保存し、採否の表を返す。
    外すもの: 行数・範囲が合わない / 名前に stack・blend を含む(それ自体が2段目の結果) /
    既存(reference_dirs)か先に採用したものとlogitの相関が duplicate_correlation 以上(同じ予測の写し)。"""
    sample_rows = np.random.default_rng(0).choice(n_train, 100000, replace=False)
    kept_names, kept = [], []
    for folder in reference_dirs:
        for oof_file in sorted(glob.glob(f"{folder}/oof_*.npy")):
            logit = to_logit(np.load(oof_file).astype(float).ravel()[sample_rows])
            kept_names.append(os.path.basename(oof_file)[4:-4]); kept.append((logit - logit.mean()) / logit.std())
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, oof, test, dataset_name in candidates:
        oof, test = oof.astype(float).ravel(), test.astype(float).ravel()
        row = {"name": name, "dataset": dataset_name, "n": len(oof), "test_n": len(test)}
        if (oof.min() < 0 or oof.max() > 1) and (test.min() < 0 or test.max() > 1):  # logitで保存されたものは確率に戻す
            oof, test = 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-test))
        if len(oof) != n_train or len(test) != n_test or np.isnan(oof).any() or np.isnan(test).any() or oof.min() < 0 or oof.max() > 1:
            rows.append({**row, "judgement": "形が合わない"}); continue
        if any(word in name.lower() for word in ("stack", "blend")):
            rows.append({**row, "judgement": "除外(2段目の結果)"}); continue
        logit = to_logit(oof[sample_rows]); logit = (logit - logit.mean()) / logit.std()
        if kept:
            correlations = np.array(kept) @ logit / len(logit)
            row["max_correlation"], row["nearest"] = float(correlations.max()), kept_names[int(correlations.argmax())]
        if row.get("max_correlation", 0) >= duplicate_correlation:
            rows.append({**row, "judgement": "重複"}); continue
        np.save(f"{out_dir}/oof_{name}.npy", oof.astype(np.float32)); np.save(f"{out_dir}/test_{name}.npy", test.astype(np.float32))
        kept_names.append(name); kept.append(logit)
        rows.append({**row, "judgement": "採用"})
    return pd.DataFrame(rows)


def extract_late_public_oof(extraction_table, data_dir):
    """後から足した公開OOF: ノートブックの出力(data/pub_oof/raw_late/<所有者>_<ノートブック名>/)から、
    extraction_table(late_public_oof_extraction.csv)に書いたファイル・列を読み、同じ形で保存する。"""
    def read_prediction(path, column):
        if path.endswith(".npy"):
            return np.load(path).ravel()
        if path.endswith(".npz"):
            return np.load(path)[column].ravel()
        table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        table.columns = [str(c) for c in table.columns]
        return table[column].values

    for _, entry in extraction_table.iterrows():
        notebook_dir = f"{data_dir}/pub_oof/raw_late/{entry['kaggle_notebook'].replace('/', '_')}"
        out_dir = f"{data_dir}/{entry['data_subfolder']}"
        os.makedirs(out_dir, exist_ok=True)
        for prediction_type in ("oof", "test"):
            values = read_prediction(f"{notebook_dir}/{entry[f'{prediction_type}_file']}", entry[f"{prediction_type}_column"])
            np.save(f"{out_dir}/{prediction_type}_{entry['member_name']}.npy", values.astype(np.float32))


# dariushafshar氏のgl_a〜nの系統(データセットのmanifest.csvの手法名から)
GOLEM_FAMILY = {"a": "lgb", "b": "xgb", "c": "cat", "d": "other", "e": "mlp", "f": "other", "g": "other", "h": "lgb", "i": "xgb",
                "j": "lgb", "k": "lgb", "l": "cat", "m": "lgb", "n": "mlp"}


def model_family(name):
    """メンバーの名前から系統を決める: tabpfn / tfm(表の基盤モデル) / mlp(ニューラルネット) / cat / lgb / xgb / other。"""
    golem = re.fullmatch(r"gl_([a-n])", name)
    if golem:
        return GOLEM_FAMILY[golem.group(1)]
    lower = name.lower()
    if "tabpfn" in lower or lower.startswith("pfn_"):
        return "tabpfn"
    if any(word in lower for word in ["tabfm", "limix", "mitra", "kumo", "exaone", "causilo", "tabicl"]):
        return "tfm"
    if any(word in lower for word in ["realmlp", "rmlp", "mlp"]) or lower.startswith("sadamt_e14_seed"):
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
