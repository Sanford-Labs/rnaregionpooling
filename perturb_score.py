"""perturb_score.py MODEL... — score the perturbation sets with probes fitted on the original tables.

Scores of the models named replace theirs in results/perturb/scores.json; the other models' scores stay.

Probes: mRNABench's ridge on the raw features (LOO alpha) fitted on the table's default train split (parquet split ==
"train"): `mbmean` as mRNABench runs it (their alpha grid, A), `bases` and `regions` ours (the grid widened, B). Delta = prediction(variant) - prediction
(original row), in label sd. For `regions`, Delta is split into the three blocks' contributions w_r . dh_r.
References: uAUG -> measured MRL effects by kind / Kozak in mrl-egfp (results/perturb/uaug_truth.json);
codons -> optimal up, non-optimal down; let-7 -> down, 8mer < 7mer-m8 < 7mer-A1 < 6mer (more negative first);
ARE -> down, more copies more negative; controls -> about zero.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from paths import CODE, WORK

spec = importlib.util.spec_from_file_location("cp", CODE / "cpu_probe.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)
R = WORK / "results/external"
P = WORK / "results/perturb"
TARGET = {"te-human": "target", "sugimoto": "target", "mrl-egfp": "target_mrl_egfp_unmod", "rnahl-human": "target"}
torch.set_num_threads(8)
DEV = torch.device("cuda:0")  # ridge on the GPU in float64; pick the card with CUDA_VISIBLE_DEVICES


def load(fd, name):
    x = np.load(fd / f"{name}.npy")
    return torch.from_numpy(x.reshape(x.shape[0], -1)).double()


def fit_ridge(x, tr, y, grid):
    fit = cp.Fitter(x, tr, [y[tr]], ("raw",))
    w, alpha, ym = fit.fits["raw"][1][0][grid]
    return fit.mu, w, ym, alpha


def main():
    truth = json.loads((P / "uaug_truth.json").read_text())["target_mrl_egfp_unmod"]
    dest = P / "scores.json"
    out = json.loads(dest.read_text()) if dest.exists() else {}
    for model in sys.argv[1:]:
        out[model] = {}
        for table, ycol in TARGET.items():
            fd, pd_ = R / model / table / "features", R / f"{model}-perturb" / table / "features"
            if not (fd / "meta.json").exists() or not (pd_ / "meta.json").exists():
                continue
            meta = json.loads((fd / "meta.json").read_text())
            sets = np.array(meta["sets"])
            y = torch.tensor(meta["targets"][ycol], dtype=torch.float64, device=DEV)
            tr = torch.from_numpy(sets == "train").to(DEV) & torch.isfinite(y)
            ysd = float(y[tr].std())
            pert = pd.read_parquet(P / f"{table}.parquet")
            res = {}
            for feat in ("bases", "regions", "mbmean"):
                X, Xp = load(fd, feat).to(DEV), load(pd_, feat).to(DEV)
                mu, w, ym, alpha = fit_ridge(X, tr, y, "A" if feat == "mbmean" else "B")
                zs = Xp - mu
                pred = (zs @ w + ym).cpu().numpy()
                orig = {r: pred[i] for i, (r, k) in enumerate(zip(pert["row"], pert["kind"])) if k == "original"}
                delta = np.array([(pred[i] - orig[r]) / ysd for i, r in enumerate(pert["row"])])
                blocks = None
                if feat == "regions":
                    d3 = zs.shape[1] // 3
                    base = {r: i for i, (r, k) in enumerate(zip(pert["row"], pert["kind"])) if k == "original"}
                    contrib = torch.stack([zs[:, j * d3:(j + 1) * d3] @ w[j * d3:(j + 1) * d3] for j in range(3)], 1).cpu().numpy()
                    blocks = np.array([(contrib[i] - contrib[base[r]]) / ysd for i, r in enumerate(pert["row"])])
                g = pert.assign(delta=delta)
                if blocks is not None:
                    g = g.assign(b5=blocks[:, 0], bc=blocks[:, 1], b3=blocks[:, 2])
                summ = g[g["kind"] != "original"].groupby(["kind", "variant", "dose"]).agg(
                    n=("delta", "size"), mean=("delta", "mean"), neg=("delta", lambda v: float((v < 0).mean())),
                    **({"b5": ("b5", "mean"), "bc": ("bc", "mean"), "b3": ("b3", "mean")} if blocks is not None else {}))
                res[feat] = {"alpha": alpha, "summary": summ.reset_index().to_dict(orient="records")}
                if table in ("te-human", "sugimoto", "mrl-egfp"):
                    u = summ.loc["uAUG"]["mean"] if "uAUG" in summ.index.get_level_values(0) else None
                    if u is not None:
                        cats = [c for c in ("extension/strong", "extension/weak", "overlap/strong", "overlap/weak", "uORF/strong", "uORF/weak")
                                if (c, 1) in u.index]
                        ref = [truth[f"kind/kozak:{c}"]["delta_sd"] for c in cats]
                        got = [float(u.loc[(c, 1)]) for c in cats]
                        res[feat]["uaug_spearman"] = float(spearmanr(ref, got).statistic) if len(cats) > 2 else None
                        res[feat]["uaug_cats"] = dict(zip(cats, got))
            out[model][table] = res
            print(f"== {model} {table}", flush=True)
            for feat, r in res.items():
                line = f"  {feat:8} alpha {r['alpha']:g}"
                if "uaug_spearman" in r:
                    line += f"  uAUG Spearman vs MPRA {r['uaug_spearman']:+.2f}"
                print(line)
                for rec in r["summary"]:
                    extra = f"  blocks 5'/CDS/3' {rec['b5']:+.3f}/{rec['bc']:+.3f}/{rec['b3']:+.3f}" if "b5" in rec else ""
                    print(f"    {rec['kind']:12} {str(rec['variant']):18} dose {rec['dose']:<5} n {rec['n']:<5} "
                          f"mean {rec['mean']:+.3f} sd  neg {rec['neg']:.2f}{extra}")
        tmp = dest.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=1, default=float))
        tmp.replace(dest)


if __name__ == "__main__":
    main()
