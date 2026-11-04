"""reap_locks.py [--loop SECONDS] — free the scheduler locks of tasks that are neither done nor running.

A lock is stale when OUT/features/meta.json does not exist and no python embed process names OUT in its arguments
(a worker that dies takes its piped child with it, and the lock stays). Freed tasks are claimed again by
queue_all.py's next pass. A fit lock (results/.fit-locks, fit_queue.py) is stale when its fit output is missing (or, for
a classification fit, lacks the current solver tag) and no python fit process names that features directory.
"""
import re
import subprocess
import sys
import time
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK

R = WORK / "results/external"
LOCKS = WORK / "results/.locks"
FIT_LOCKS = WORK / "results/.fit-locks"


def name(out):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(Path(out).relative_to(R)))


def once():
    rows = subprocess.run(["ps", "-u", "sna", "-o", "comm=,args="], capture_output=True, text=True).stdout.splitlines()
    live, fitting = set(), set()
    for line in rows:
        comm, _, args = line.partition(" ")
        if comm.startswith("python") and re.search(r"harness/embed_\w+\.py", args):
            for tok in args.split():
                if tok.startswith(str(R) + "/"):
                    live.add(name(tok))
        if comm.startswith("python") and re.search(r"harness/fit_(cls|ext)\.py", args):
            kind = "cls" if "fit_cls.py" in args else "ridge"
            for tok in args.split():
                if tok.startswith(str(R) + "/"):
                    d = Path(tok).parent
                    fitting.add(f"{d.parent.name}_{d.name}_{kind}")
    outs = {}
    for meta in R.glob("*/*/features/meta.json"):
        outs[name(meta.parent.parent)] = True
    freed = []
    for lock in LOCKS.iterdir():
        if lock.name in live or outs.get(lock.name) or time.time() - lock.stat().st_mtime < 300:  # a fresh claim may not have started yet
            continue
        lock.rmdir()
        freed.append(lock.name)
    dirs = {f"{d.parent.name}_{d.name}_{k}": d / ("fit_cls.json" if k == "cls" else "fit.json")
            for d in R.glob("*/*") if d.is_dir() for k in ("cls", "ridge")}  # group names hold "_" (utrlm-te_el)
    for lock in FIT_LOCKS.iterdir() if FIT_LOCKS.exists() else ():
        out = dirs.get(lock.name)
        done = out is not None and out.exists() and (out.name != "fit_cls.json" or '"solver": "newton-centred-f64fallback' in out.read_text())
        if done or lock.name in fitting or time.time() - lock.stat().st_mtime < 300:
            continue
        lock.rmdir()
        freed.append("fit:" + lock.name)
    print(time.strftime("%m-%d %H:%M"), "freed:", ", ".join(freed) if freed else "none", flush=True)


if __name__ == "__main__":
    if "--loop" in sys.argv:
        every = int(sys.argv[sys.argv.index("--loop") + 1])
        while True:
            once()
            time.sleep(every)
    once()
