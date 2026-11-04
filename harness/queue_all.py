"""queue_all.py --card N [--big] [--only GROUP] — one worker per card over a single ordered task list (results/external/...).

A task is done once OUT/features/meta.json exists; a worker claims a task by creating its lock directory under
results/.locks (mkdir is atomic), runs it, and frees the lock only if the task failed; a failed task is then recorded
under results/.failed (the log's last lines) and no worker claims it until that file is removed. `--big` marks a card
the work has to itself: RiNALMo giga, AIDO.RNA 1.6B, Evo 1 and Evo 2 run only there. On the other cards the
multimolecule and transformers jobs use half the token budget and at most EMBED_MEM_FRACTION of the card's memory.
Each pass starts from the top, so earlier tasks go first; on --big cards the big tasks go before the rest.
"""
import argparse
import os
import re
import subprocess
import time
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, MODEL_DIR, WORK

P = WORK
R = P / "results/external"
PERT = P / "results/perturb"
LOCKS = P / "results/.locks"
FAILED = P / "results/.failed"  # one file per failed task; a task is retried once its file is removed
MB = str(CODE / "envs/multimolecule")
EVO = str(CODE / "envs/evo2")
EVO1 = str(CODE / "envs/evo1")
# mRNABench's Evo 1 row uses the 1.1_fix revision of togethercomputer/evo-1-8k-base
EVO1_WEIGHTS = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub/models--togethercomputer--evo-1-8k-base/refs/1.1_fix"
REG = ["sugimoto", "rnahl-human", "rnahl-mouse", "te-human", "te-mouse", "mrl-hl"]
CLS = ["vep-traitgym-complex", "vep-traitgym-mendelian", "rna-loc-fazal", "rna-lifecycle-ietswaart", "prot-loc",
       "go-bp", "go-cc", "go-mf", "eclip-binding-hepg2", "eclip-binding-k562"]
MPRA = ["mrl-designed", "mrl-varying", "mrl-mcherry", "mrl-egfp"]
ALL = REG + CLS + MPRA
# Evo 1: the longest tables first (Evo 2's minutes per table)
EVO1_ORDER = MPRA[::-1] + ["eclip-binding-k562", "eclip-binding-hepg2", "rnahl-human", "rnahl-mouse", "sugimoto", "te-mouse",
                           "te-human", "go-mf", "prot-loc", "rna-lifecycle-ietswaart", "go-cc", "rna-loc-fazal", "go-bp",
                           "vep-traitgym-complex", "vep-traitgym-mendelian", "mrl-hl"]
assert sorted(EVO1_ORDER) == sorted(ALL)


def task(kind, model, table, out, *extra, big=False):
    return {"kind": kind, "model": model, "table": table, "out": Path(out), "extra": list(extra), "big": big}


def tasks():
    t = []
    t += [task("mm", "rinalmo-giga", x, R / "rinalmo-giga" / x, big=True) for x in ALL]
    if (MODEL_DIR / "helix-mrna-hf/model.safetensors").exists():  # gated: listed once the weights are down
        t += [task("helix", "helix-mrna", x, R / "helix-mrna" / x) for x in ALL]
        t += [task("helix", "helix-mrna", x, R / "helix-mrna-noE" / x, "--no-e") for x in ALL]  # the same weights without the codon marks
    t += [task("mm", "ernierna", x, R / "ernierna" / x) for x in ALL]
    t += [task("mm", "ernierna", x, R / "ernierna-iso" / x, "--isolate") for x in REG]
    t += [task("mm", "ernierna", x, R / "ernierna-perturb" / x, "--parquet", str(PERT / f"{x}.parquet"))
          for x in ("te-human", "sugimoto", "mrl-egfp", "rnahl-human")]
    t += [task("mm", "rinalmo-giga", x, R / "rinalmo-giga-iso" / x, "--isolate", big=True) for x in REG]
    t += [task("mm", "rinalmo-giga", x, R / "rinalmo-giga-perturb" / x, "--parquet", str(PERT / f"{x}.parquet"), big=True)
          for x in ("te-human", "sugimoto", "mrl-egfp", "rnahl-human")]
    for m in ("rnaernie", "rnamsm", "utrlm-te_el", "rnafm", "splicebert"):
        t += [task("mm", m, x, R / m / x) for x in ALL]
    for m in ("rnabert", "utrbert-6mer", "dnabert2", "dnaberts"):
        t += [task("mm2", m, x, R / m / x) for x in ALL]
    for m in ("hyenadna-medium-450k", "nt-2.5b-multi"):
        t += [task("hf", m, x, R / m / x) for x in ALL]
    t += [task("mm2", "rinalmo-giga", x, R / "rinalmo-giga-w1022" / x, "--window", "1022", "--stride", "511", big=True) for x in REG]
    t += [task("mm2", "rinalmo-giga", "te-human", R / "rinalmo-giga-tokens" / "te-human", "--tokens", big=True)]
    # genbio's layout through embed_mm.py / aido_model.py
    t += [task("mm", "aido-rna-1.6b", x, R / "aido-rna-1.6b" / x, big=True) for x in ALL]
    # the perturbation sets for the other models, then their per-region encoding
    PT = ("te-human", "sugimoto", "mrl-egfp", "rnahl-human")
    pq = lambda x: ("--parquet", str(PERT / f"{x}.parquet"))  # noqa: E731
    for m in ("rnaernie", "rnamsm", "utrlm-te_el", "rnafm", "splicebert"):
        t += [task("mm", m, x, R / f"{m}-perturb" / x, *pq(x)) for x in PT]
    for m in ("rnabert", "utrbert-6mer", "dnabert2", "dnaberts"):
        t += [task("mm2", m, x, R / f"{m}-perturb" / x, *pq(x)) for x in PT]
    t += [task("hf", "hyenadna-medium-450k", x, R / "hyenadna-medium-450k-perturb" / x, *pq(x)) for x in PT]
    if (MODEL_DIR / "helix-mrna-hf/model.safetensors").exists():
        t += [task("helix", "helix-mrna", x, R / "helix-mrna-perturb" / x, *pq(x)) for x in PT]  # embed_helix.py has no per-region mode
    t += [task("hf", "nt-2.5b-multi", x, R / "nt-2.5b-multi-perturb" / x, *pq(x)) for x in PT]
    t += [task("mm", "aido-rna-1.6b", x, R / "aido-rna-1.6b-perturb" / x, *pq(x), big=True) for x in PT]
    for m in ("rnaernie", "rnamsm", "utrlm-te_el", "rnafm", "splicebert"):
        t += [task("mm", m, x, R / f"{m}-iso" / x, "--isolate") for x in REG]
    for m in ("rnabert", "utrbert-6mer", "dnabert2", "dnaberts"):
        t += [task("mm2", m, x, R / f"{m}-iso" / x, "--isolate") for x in REG]
    t += [task("hf", "hyenadna-medium-450k", x, R / "hyenadna-medium-450k-iso" / x, "--isolate") for x in REG]
    # middle layer (hidden_states[L // 2]) of five multimolecule models on the regression tables
    for m, k in (("ernierna", 6), ("rnaernie", 6), ("rnafm", 6), ("splicebert", 3)):
        t += [task("mm", m, x, R / f"{m}-mid" / x, "--layer", str(k)) for x in REG]
    t += [task("mm", "rinalmo-giga", x, R / "rinalmo-giga-mid" / x, "--layer", "16", big=True) for x in REG]
    t += [task("evo2", "evo2_7b_base", x, R / "evo2-7b-base-perturb" / x, *pq(x), big=True) for x in PT]
    if EVO1_WEIGHTS.exists():
        t += [task("evo1", "evo-1-8k-base", x, R / "evo1-8k-base-perturb" / x, *pq(x), big=True) for x in PT]
    t += [task("evo2", "evo2_7b_base", x, R / "evo2-7b-base" / x, big=True) for x in REG + CLS + MPRA]
    if EVO1_WEIGHTS.exists():  # mRNABench's Evo 1 row (evo-model, the 1.1_fix weights); listed once they are downloaded
        t += [task("evo1", "evo-1-8k-base", x, R / "evo1-8k-base" / x, big=True) for x in EVO1_ORDER]
    return t


def lockname(t):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(t["out"].relative_to(R)))


def command(t, card, shared):
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    budget = ["--budget", "32768"] if shared else []
    if shared:  # a job takes at most this share of a card it shares
        env["EMBED_MEM_FRACTION"] = "0.18"
    if t["kind"] == "evo2":
        env |= {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(card)}
        return (["uv", "run", "--no-sync", "python", str(CODE / "harness/embed_evo2.py"), t["model"], t["table"], str(t["out"]),
                 "--device", "cuda:0", *t["extra"]], EVO, env)
    if t["kind"] == "evo1":
        env |= {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(card)}
        return (["uv", "run", "--no-sync", "python", str(CODE / "harness/embed_evo1.py"), t["model"], t["table"], str(t["out"]),
                 "--device", "cuda:0", *t["extra"]], EVO1, env)
    if t["kind"] == "helix":
        env |= {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(card)}
        return (["uv", "run", "--no-sync", "python", str(CODE / "harness/embed_helix.py"), t["table"], str(t["out"]),
                 "--device", "cuda:0", *budget, *t["extra"]], MB, env)
    if t["kind"] in ("mm", "mm2", "hf"):
        env |= {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(card)}
        script = {"mm": "embed_mm.py", "mm2": "embed_mm2.py", "hf": "embed_hf.py"}[t["kind"]]
        dtype = [] if t["kind"] == "hf" else ["--dtype", "bf16"]
        cmd = ["uv", "run", "--no-sync", "python", str(CODE / "harness" / script), t["model"], t["table"], str(t["out"]),
               "--device", "cuda:0", *dtype, *budget, *t["extra"]]
        return cmd, MB, env
    raise ValueError(f"no command for task kind {t['kind']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--big", action="store_true")
    ap.add_argument("--only", default="", help="one output group (results/external/<group>); exits when it is claimed")
    a = ap.parse_args()
    LOCKS.mkdir(parents=True, exist_ok=True)
    FAILED.mkdir(parents=True, exist_ok=True)
    while True:
        pick = None
        for t in sorted(tasks(), key=lambda t: not t["big"]) if a.big else tasks():  # whole cards: big work first
            if (t["out"] / "features" / "meta.json").exists() or (t["big"] and not a.big) or (FAILED / lockname(t)).exists() \
                    or (a.only and t["out"].parent.name != a.only):
                continue
            try:
                os.mkdir(LOCKS / lockname(t))
            except FileExistsError:
                continue
            pick = t
            break
        if pick is None:
            print(f"=== {time.strftime('%m-%d %H:%M')} card {a.card}: nothing left to claim", flush=True)
            return
        cmd, cwd, env = command(pick, a.card, shared=not a.big)
        print(f"=== {time.strftime('%m-%d %H:%M')} card {a.card}: {pick['out'].relative_to(R)}", flush=True)
        pick["out"].mkdir(parents=True, exist_ok=True)
        log = pick["out"] / "embed.log"  # a file, not a pipe: the job outlives a worker that is stopped
        with open(log, "w") as fh:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
        tail = [l for l in log.read_text(errors="replace").splitlines() if re.search(r"^wrote|Error|Traceback|error:", l)]
        ok = proc.returncode == 0 and (pick["out"] / "features" / "meta.json").exists()
        print(f"    {'ok' if ok else 'FAILED rc=' + str(proc.returncode)}: " + " | ".join(tail[-3:]), flush=True)
        if not ok:
            last = log.read_text(errors="replace").splitlines()[-12:]
            (FAILED / lockname(pick)).write_text(f"{time.strftime('%m-%d %H:%M')} card {a.card} rc={proc.returncode}\n" + "\n".join(last) + "\n")
            (LOCKS / lockname(pick)).rmdir()


if __name__ == "__main__":
    main()
