"""公開OOFライブラリを1つのフォルダにそろえ、すでに持っている予測と重複するものを外す。

目的: 参加者ごとにファイルの形(npy/csv、float32/64、名前の付け方)が違う公開OOFを、stack_final.pyで読める形
  (oof_<名前>.npy・test_<名前>.npy、train.csv・test.csvと同じ行の順)にそろえる。
ひとことで言うと: よそから集めた「味見の記録」を同じ書式の帳面に写し、同じ記録の写しが二重に入らないようにする。
やり方:
  - 入力: data/pub_oof/raw/<データセット名>/ の中の oof_*.npy と同名の test_*.npy の組、または --csv で指定した
    「名前=OOF=test」の組(npyまたはcsv)。
  - 名前に stack・blend を含むもの(それ自体が2段目の結果)は入れない。2段目に2段目を重ねると、混ぜ方を学んだ行と
    採点する行が重なり、OOFが甘くなるため。
  - 既存のフォルダ(--ref、例: cdeotte124)と、この中で先に採用したものの両方に対し、logitの相関を10万行で計算する。
    相関0.9999以上は「同じモデルの写し」とみなして外す(同じファイルが別のデータセットに再掲されていることが多い)。
  - 出力: data/pub_oof/extra/ と、採否を書いた _gather.csv。
なぜ: 同じ予測が2本入ると、2段目のL2正則化がその系統を実質2倍の重みで扱ってしまう。
使い方: python src/9_public_oof/pub_oof_gather.py --ref data/pub_oof/cdeotte124 --out data/pub_oof/extra
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/pub_oof/raw")
ap.add_argument("--ref", nargs="*", default=[])
ap.add_argument("--out", default="data/pub_oof/extra")
ap.add_argument("--pair", nargs="*", default=[], help="名前=OOFのファイル=testのファイル(npyまたはcsv、csvは最後の列を予測値とみなす)")
ap.add_argument("--thr", type=float, default=0.9999)
args = ap.parse_args()

n_tr = len(pd.read_csv("data/train.csv", usecols=["id"]))
n_te = len(pd.read_csv("data/test.csv", usecols=["id"]))
idx = np.random.default_rng(0).choice(n_tr, 100000, replace=False)
lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))

def by_id(df, start):
    """id列があれば並べ替え、train/testと同じ行の順になっているかを確かめる。"""
    if "id" in df.columns:
        df = df.sort_values("id")
        assert (df["id"].values == np.arange(start, start + len(df))).all()
    return df


cands = []  # (名前, OOF, test, 出典)
for d in sorted(glob.glob(f"{args.raw}/*/")):
    src = os.path.basename(d.rstrip("/\\"))
    pre = src.removeprefix("nb_").split("_")[0][:6]  # 投稿者名の先頭6文字を接頭辞にして名前の衝突を防ぐ
    # 形式ごとに読み方が違う: npyの組 / npz(oof・testの2配列) / 列ごとに1モデルのparquet / id・predのcsvの組
    for f in sorted(glob.glob(f"{d}oof_*.npy")):
        n = os.path.basename(f)[4:-4]
        t = f"{d}test_{n}.npy"
        if os.path.exists(t):
            cands.append((f"{pre}_{n}", np.load(f, allow_pickle=False), np.load(t, allow_pickle=False), src))
    for f in sorted(glob.glob(f"{d}*.npz")):
        z = np.load(f, allow_pickle=False)
        b = os.path.basename(f)[:-4].removeprefix("preds_")
        for ko, kt in (("oof", "test"), ("train", "test")):
            if {ko, kt} <= set(z.files):
                cands.append((f"{pre}_{b}", z[ko], z[kt], src)); break
        else:  # 1つのnpzに oof_X・test_X が並ぶ形
            cands += [(f"{pre}_{k[4:]}", z[k], z["test_" + k[4:]], src) for k in z.files if k.startswith("oof_") and "test_" + k[4:] in z.files]
    for fo, ft, rd in ((f"{d}oof_predictions.parquet", f"{d}test_predictions.parquet", pd.read_parquet),
                       (f"{d}oof_members.csv", f"{d}test_members.csv", pd.read_csv)):
        if not os.path.exists(fo):
            continue
        o, t = by_id(rd(fo), 0), by_id(rd(ft), n_tr)
        for c in [c for c in o.columns if c in t.columns and c != "id"]:
            cands.append((f"{pre}_{c}", o[c].values, t[c].values, src))
    pairs = [(f, f"{d}test_{os.path.basename(f)[4:]}", os.path.basename(f)[4:-4]) for f in glob.glob(f"{d}oof_*.csv")]
    pairs += [(f, f[:-8] + "_test.csv", os.path.basename(f)[:-8]) for f in glob.glob(f"{d}*_oof.csv")]
    for f, t, n in sorted(pairs):
        if os.path.exists(t):
            o, tt = by_id(pd.read_csv(f), 0), by_id(pd.read_csv(t), n_tr)
            cands.append((f"{pre}_{n}", o.iloc[:, -1].values, tt.iloc[:, -1].values, src))
for c in args.pair:
    n, o, t = c.split("=")
    rd = lambda f, s: np.load(f, allow_pickle=False) if f.endswith(".npy") else by_id(pd.read_csv(f), s).iloc[:, -1].values
    cands.append((n, rd(o, 0), rd(t, n_tr), os.path.basename(os.path.dirname(o))))

ref = {}
for d in args.ref:
    for f in sorted(glob.glob(f"{d}/oof_*.npy")):
        ref[os.path.basename(f)[4:-4]] = lg(np.load(f).astype(float).ravel()[idx])
names, R = list(ref), np.array(list(ref.values())) if ref else np.zeros((0, len(idx)))
Rz = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True) if len(R) else R

os.makedirs(args.out, exist_ok=True)
rows = []
for n, o, t, src in cands:
    o, t = o.astype(float).ravel(), t.astype(float).ravel()
    r = {"name": n, "src": src, "n": len(o), "test_n": len(t)}
    if (o.min() < 0 or o.max() > 1) and (t.min() < 0 or t.max() > 1):  # logitで保存されたもの(arhancanli12等)は確率に戻す
        o, t = 1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t)); r["logit"] = True
    if len(o) != n_tr or len(t) != n_te or np.isnan(o).any() or np.isnan(t).any() or o.min() < 0 or o.max() > 1:
        r["判定"] = "形が合わない"; rows.append(r); continue
    if any(k in n.lower() for k in ("stack", "blend")):
        r["判定"] = "除外(2段目の結果)"; rows.append(r); continue
    z = lg(o[idx]); z = (z - z.mean()) / z.std()
    if len(Rz):
        c = Rz @ z / len(z); j = int(c.argmax()); r["max_corr"], r["nearest"] = float(c[j]), names[j]
    if r.get("max_corr", 0) >= args.thr:
        r["判定"] = "重複"
    else:
        r["判定"] = "採用"
        np.save(f"{args.out}/oof_{n}.npy", o.astype(np.float32)); np.save(f"{args.out}/test_{n}.npy", t.astype(np.float32))
        names.append(n); Rz = np.vstack([Rz, z]) if len(Rz) else z[None]
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f"{args.out}/_gather.csv", index=False)
print(df["判定"].value_counts().to_string())
print(df.to_string(index=False, max_colwidth=40))
