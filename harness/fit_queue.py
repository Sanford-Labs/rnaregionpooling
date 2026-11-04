"""fit_queue.py --card N [--cls-only] [--shared] — fit every results/external/<model>/<table>/features that has no fit yet, oldest first.

Regression tables go to fit_ext.py (fit.json; it takes a card through the fit slots), classification tables to
fit_cls.py on card N (fit_cls.json). A worker claims a fit with a lock directory under results/.fit-locks and frees
it if the fit fails, recording the failure under results/.fit-failed (no worker takes that fit again until the
file is removed); it rescans every two minutes and runs until stopped. Perturbation and token outputs are
left to their own scripts. A classification fit without fit_cls.py's
current solver tag counts as not done.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK

P = WORK
R = P / "results/external"
LOCKS = P / "results/.fit-locks"
FAILED = P / "results/.fit-failed"  # one file per failed fit; retried once its file is removed
CLS = {"vep-traitgym-complex", "vep-traitgym-mendelian", "rna-loc-fazal", "rna-lifecycle-ietswaart", "prot-loc",
       "go-bp", "go-cc", "go-mf", "eclip-binding-hepg2", "eclip-binding-k562"}


def todo():
    for meta in sorted(R.glob("*/*/features/meta.json"), key=os.path.getmtime):
        d = meta.parent.parent
        group = d.parent.name
        if group.endswith(("-perturb", "-tokens")):
            continue
        cls = d.name in CLS
        out = d / ("fit_cls.json" if cls else "fit.json")
        if not out.exists() or (cls and '"solver": "newton-centred-f64fallback' not in out.read_text()):
            yield d, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--shared", action="store_true", help="the card runs other work too: cap the memory share")
    ap.add_argument("--cls-only", action="store_true", help="classification fits only (ridge fits take a slot on card 0 or 1)")
    a = ap.parse_args()
    LOCKS.mkdir(parents=True, exist_ok=True)
    FAILED.mkdir(parents=True, exist_ok=True)
    while True:
        pick = None
        for d, cls in todo():
            name = f"{d.parent.name}_{d.name}_{'cls' if cls else 'ridge'}"
            if (FAILED / name).exists() or (a.cls_only and not cls) \
                    or (a.card in (4, 5, 6, 7) and d.parent.name.startswith("evo")):  # Evo 1 / 2 are 4,096-12,288 wide: past 18% of a card
                continue
            try:
                os.mkdir(LOCKS / name)
            except FileExistsError:
                continue
            pick = d, cls, name
            break
        if pick is None:
            time.sleep(120)
            continue
        d, cls, name = pick
        script = "fit_cls.py" if cls else "fit_ext.py"
        cmd = ["uv", "run", "--project", str(CODE), "--no-sync", "python", str(CODE / "harness" / script),
               *(["--device", "cuda:0"] if cls else []), str(d / "features")]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(a.card), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                   OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        if a.shared:  # a card shared with other work: at most 18% of its memory, as the embedding jobs there
            env["EMBED_MEM_FRACTION"] = "0.18"
        started = time.monotonic()
        print(f"=== {time.strftime('%m-%d %H:%M')} card {a.card}: {d.parent.name}/{d.name} {'cls' if cls else 'ridge'}", flush=True)
        log = d / ("fit_cls.log" if cls else "fit.log")
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, cwd=CODE, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
        lines = [l for l in log.read_text(errors="replace").splitlines() if "warn" not in l.lower()]
        ok = proc.returncode == 0 and (d / ("fit_cls.json" if cls else "fit.json")).exists()
        print(f"    {'ok' if ok else 'FAILED rc=' + str(proc.returncode)} {(time.monotonic() - started) / 60:.1f} min: "
              + (lines[-1] if lines else ""), flush=True)
        if not ok:
            last = log.read_text(errors="replace").splitlines()[-12:]
            (FAILED / name).write_text(f"{time.strftime('%m-%d %H:%M')} card {a.card} rc={proc.returncode}\n" + "\n".join(last) + "\n")
            (LOCKS / name).rmdir()


if __name__ == "__main__":
    main()
