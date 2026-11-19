"""公開OOFライブラリを1つのフォルダにそろえ、すでに持っている予測と重複するものを外す。

目的: 参加者ごとにファイルの形(npy/csv、float32/64、名前の付け方)が違う公開OOFを、make_final_submission.pyで読める形
  (oof_<名前>.npy・test_<名前>.npy、train.csv・test.csvと同じ行の順)にそろえる。
ひとことで言うと: 形式の違う公開OOFを同じ形式のファイルに書き直し、同じ予測の写しが二重に入らないようにする。
やり方:
  - 入力: data/pub_oof/raw/<データセット名>/ の中の oof_*.npy と同名の test_*.npy の組、または --named-pair で指定した
    「名前=OOF=test」の組(npyまたはcsv)。
  - 名前に stack・blend を含むもの(それ自体が2段目の結果)は入れない。2段目に2段目を重ねると、混ぜ方を学んだ行と
    採点する行が重なり、OOFが甘くなるため。
  - 既存のフォルダ(--reference-dir、例: cdeotte124)と、この中で先に採用したものの両方に対し、logitの相関を10万行で計算する。
    相関0.9999以上は「同じモデルの写し」とみなして外す(同じファイルが別のデータセットに再掲されていることが多い)。
  - 出力: data/pub_oof/extra/ と、採否を書いた _gather.csv。
なぜ: 同じ予測が2本入ると、2段目のL2正則化がその系統を実質2倍の重みで扱ってしまう。
使い方: python src/9_public_oof_stacking/collect_public_oof.py --reference-dir data/pub_oof/cdeotte124 --out-dir data/pub_oof/extra
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--raw-dir", default="data/pub_oof/raw")
ap.add_argument("--reference-dir", nargs="*", default=[])
ap.add_argument("--out-dir", default="data/pub_oof/extra")
ap.add_argument("--named-pair", nargs="*", default=[], help="名前=OOFのファイル=testのファイル(npyまたはcsv、csvは最後の列を予測値とみなす)")
ap.add_argument("--duplicate-corr", type=float, default=0.9999)
args = ap.parse_args()

n_train = len(pd.read_csv("data/train.csv", usecols=["id"]))
n_test = len(pd.read_csv("data/test.csv", usecols=["id"]))
sample_rows = np.random.default_rng(0).choice(n_train, 100000, replace=False)  # 相関は10万行で計算(速さのため)


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sort_by_id(df, first_id):
    """id列があれば並べ替え、train/testと同じ行の順になっているかを確かめる。"""
    if "id" in df.columns:
        df = df.sort_values("id")
        assert (df["id"].values == np.arange(first_id, first_id + len(df))).all()
    return df


candidates = []  # (名前, OOF, test, 出典)
for folder in sorted(glob.glob(f"{args.raw_dir}/*/")):
    dataset_name = os.path.basename(folder.rstrip("/\\"))
    prefix = dataset_name.removeprefix("nb_").split("_")[0][:6]  # 投稿者名の先頭6文字を接頭辞にして名前の衝突を防ぐ
    # 形式ごとに読み方が違う: npyの組 / npz(oof・testの2配列) / 列ごとに1モデルのparquet / id・predのcsvの組
    for oof_file in sorted(glob.glob(f"{folder}oof_*.npy")):
        name = os.path.basename(oof_file)[4:-4]
        test_file = f"{folder}test_{name}.npy"
        if os.path.exists(test_file):
            candidates.append((f"{prefix}_{name}", np.load(oof_file, allow_pickle=False), np.load(test_file, allow_pickle=False), dataset_name))
    for npz_file in sorted(glob.glob(f"{folder}*.npz")):
        npz = np.load(npz_file, allow_pickle=False)
        base_name = os.path.basename(npz_file)[:-4].removeprefix("preds_")
        for oof_key, test_key in (("oof", "test"), ("train", "test")):
            if {oof_key, test_key} <= set(npz.files):
                candidates.append((f"{prefix}_{base_name}", npz[oof_key], npz[test_key], dataset_name)); break
        else:  # 1つのnpzに oof_X・test_X が並ぶ形
            candidates += [(f"{prefix}_{k[4:]}", npz[k], npz["test_" + k[4:]], dataset_name)
                           for k in npz.files if k.startswith("oof_") and "test_" + k[4:] in npz.files]
    for oof_file, test_file, read_table in ((f"{folder}oof_predictions.parquet", f"{folder}test_predictions.parquet", pd.read_parquet),
                                            (f"{folder}oof_members.csv", f"{folder}test_members.csv", pd.read_csv)):
        if not os.path.exists(oof_file):
            continue
        oof_table, test_table = sort_by_id(read_table(oof_file), 0), sort_by_id(read_table(test_file), n_train)
        for col in [c for c in oof_table.columns if c in test_table.columns and c != "id"]:
            candidates.append((f"{prefix}_{col}", oof_table[col].values, test_table[col].values, dataset_name))
    csv_pairs = [(f, f"{folder}test_{os.path.basename(f)[4:]}", os.path.basename(f)[4:-4]) for f in glob.glob(f"{folder}oof_*.csv")]
    csv_pairs += [(f, f[:-8] + "_test.csv", os.path.basename(f)[:-8]) for f in glob.glob(f"{folder}*_oof.csv")]
    for oof_file, test_file, name in sorted(csv_pairs):
        if os.path.exists(test_file):
            oof_table, test_table = sort_by_id(pd.read_csv(oof_file), 0), sort_by_id(pd.read_csv(test_file), n_train)
            candidates.append((f"{prefix}_{name}", oof_table.iloc[:, -1].values, test_table.iloc[:, -1].values, dataset_name))


for pair in args.named_pair:
    name, oof_file, test_file = pair.split("=")
    oof, test = [np.load(f, allow_pickle=False) if f.endswith(".npy") else sort_by_id(pd.read_csv(f), first_id).iloc[:, -1].values
                 for f, first_id in ((oof_file, 0), (test_file, n_train))]
    candidates.append((name, oof, test, os.path.basename(os.path.dirname(oof_file))))

# 比べる相手(--reference-dir のフォルダ)のlogitを、平均0・標準偏差1にそろえて持つ(内積/行数 = 相関になる)
ref_logits = {}
for folder in args.reference_dir:
    for oof_file in sorted(glob.glob(f"{folder}/oof_*.npy")):
        ref_logits[os.path.basename(oof_file)[4:-4]] = to_logit(np.load(oof_file).astype(float).ravel()[sample_rows])
kept_names = list(ref_logits)
kept = np.array(list(ref_logits.values())) if ref_logits else np.zeros((0, len(sample_rows)))
kept_std = (kept - kept.mean(1, keepdims=True)) / kept.std(1, keepdims=True) if len(kept) else kept

os.makedirs(args.out_dir, exist_ok=True)
rows = []
for name, oof, test, dataset_name in candidates:
    oof, test = oof.astype(float).ravel(), test.astype(float).ravel()
    row = {"name": name, "dataset": dataset_name, "n": len(oof), "test_n": len(test)}
    if (oof.min() < 0 or oof.max() > 1) and (test.min() < 0 or test.max() > 1):  # logitで保存されたもの(arhancanli12等)は確率に戻す
        oof, test = 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-test)); row["logit"] = True
    if len(oof) != n_train or len(test) != n_test or np.isnan(oof).any() or np.isnan(test).any() or oof.min() < 0 or oof.max() > 1:
        row["判定"] = "形が合わない"; rows.append(row); continue
    if any(k in name.lower() for k in ("stack", "blend")):
        row["判定"] = "除外(2段目の結果)"; rows.append(row); continue
    logit = to_logit(oof[sample_rows]); logit = (logit - logit.mean()) / logit.std()
    if len(kept_std):
        corrs = kept_std @ logit / len(logit); j = int(corrs.argmax())
        row["max_corr"], row["nearest"] = float(corrs[j]), kept_names[j]
    if row.get("max_corr", 0) >= args.duplicate_corr:
        row["判定"] = "重複"
    else:
        row["判定"] = "採用"
        np.save(f"{args.out_dir}/oof_{name}.npy", oof.astype(np.float32)); np.save(f"{args.out_dir}/test_{name}.npy", test.astype(np.float32))
        kept_names.append(name); kept_std = np.vstack([kept_std, logit]) if len(kept_std) else logit[None]
    rows.append(row)
result = pd.DataFrame(rows)
result.to_csv(f"{args.out_dir}/_gather.csv", index=False)
print(result["判定"].value_counts().to_string())
print(result.to_string(index=False, max_colwidth=40))
