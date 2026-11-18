"""後から足した公開OOF 11本を、ノートブックの出力から取り出してmake_final_submission.pyで読める形にする。

目的: 最初に集めたデータセットとは別に、コンペの後半に公開されたノートブックの出力から11本をスタックに足した。
  ファイルの形(csvの列・npzのキー・parquetの列)がノートブックごとに違うので、late_public_oof_sources.csvに取り出し方を書いておき、
  data/<folder>/oof_<name>.npy・test_<name>.npy(train.csv・test.csvと同じ行の順)に保存する。
ひとことで言うと: よそのノートブックの出力から、使った予測の列だけを同じ書式で書き写す。
やり方:
  - ノートブックの出力は `kaggle kernels output <notebook> -p data/pub_oof/raw_late/<owner>_<slug>` で取得しておく。
  - npyはそのまま、npzはキー、csv・parquetは列(keyの列名。parquetの列名は数字の文字列)を読む。
使い方: python src/9_public_oof_stacking/extract_late_public_oof.py
"""
import os

import numpy as np
import pandas as pd

rows = pd.read_csv("src/9_public_oof_stacking/late_public_oof_sources.csv", dtype=str, keep_default_na=False)


def read(path, key):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False).ravel()
    if path.endswith(".npz"):
        return np.load(path, allow_pickle=False)[key].ravel()
    d = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    d.columns = [str(c) for c in d.columns]
    return d[key].values


for _, r in rows.iterrows():
    src = f"data/pub_oof/raw_late/{r['notebook'].replace('/', '_')}"
    out = f"data/{r['folder']}"
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/oof_{r['name']}.npy", read(f"{src}/{r['oof_file']}", r["oof_key"]).astype(np.float32))
    np.save(f"{out}/test_{r['name']}.npy", read(f"{src}/{r['test_file']}", r["test_key"]).astype(np.float32))
    print("保存", out, r["name"])
