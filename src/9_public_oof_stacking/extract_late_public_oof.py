"""後から足した公開OOF 11本を、ノートブックの出力から取り出してmake_final_submission.pyで読める形にする。

目的: 最初に集めたデータセットとは別に、コンペの後半に公開されたノートブックの出力から11本をスタックに足した。
  ファイルの形(csvの列・npzのキー・parquetの列)がノートブックごとに違うので、late_public_oof_extraction.csvに取り出し方を書いておき、
  data/<data_subfolder>/oof_<member_name>.npy・test_<member_name>.npy(train.csv・test.csvと同じ行の順)に保存する。
ひとことで言うと: よそのノートブックの出力から、使った予測の列だけを同じ書式で書き写す。
やり方:
  - ノートブックの出力は `kaggle kernels output <kaggle_notebook> -p data/pub_oof/raw_late/<所有者>_<ノートブック名>` で取得しておく。
  - npyはそのまま、npzは配列名、csv・parquetは列名(oof_column・test_columnに書いた名前。parquetの列名は数字の文字列)を読む。
使い方: python src/9_public_oof_stacking/extract_late_public_oof.py
"""
import os

import numpy as np
import pandas as pd

extraction = pd.read_csv("src/9_public_oof_stacking/late_public_oof_extraction.csv", dtype=str, keep_default_na=False)


def read_prediction(path, column):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False).ravel()
    if path.endswith(".npz"):
        return np.load(path, allow_pickle=False)[column].ravel()
    table = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    table.columns = [str(c) for c in table.columns]
    return table[column].values


for _, entry in extraction.iterrows():
    notebook_dir = f"data/pub_oof/raw_late/{entry['kaggle_notebook'].replace('/', '_')}"
    out_dir = f"data/{entry['data_subfolder']}"
    os.makedirs(out_dir, exist_ok=True)
    oof = read_prediction(f"{notebook_dir}/{entry['oof_file']}", entry["oof_column"])
    test = read_prediction(f"{notebook_dir}/{entry['test_file']}", entry["test_column"])
    np.save(f"{out_dir}/oof_{entry['member_name']}.npy", oof.astype(np.float32))
    np.save(f"{out_dir}/test_{entry['member_name']}.npy", test.astype(np.float32))
    print("保存", out_dir, entry["member_name"])
