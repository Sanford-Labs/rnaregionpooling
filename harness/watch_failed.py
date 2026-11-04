"""watch_failed.py [LOG ...] — wait until something in the embedding or fit queues needs a hand, print it, exit.

Exits on the first of: a new file under results/.failed or results/.fit-failed (queue_all.py / fit_queue.py write
one per failed task); a new FAILED line in one of the LOG files (worker output from before the failure files); a
claimed embedding whose embed.log and feature files have not changed for 45 minutes. Polls once a minute.
"""
import sys
import time
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK

sys.path.insert(0, CODE / "harness")
import queue_all as q  # noqa: E402

DIRS = [q.P / "results/.failed", q.P / "results/.fit-failed"]
STALL = 45 * 60


def markers():
    return {p for d in DIRS if d.exists() for p in d.iterdir()}


def lines(f):
    try:
        return Path(f).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []


def failed_at(f):
    ls = lines(f)
    return [(ls[i - 1] if i else "", ls[i]) for i, l in enumerate(ls) if "FAILED" in l]


def main():
    logs = sys.argv[1:]
    seen = markers()
    counts = {f: len(failed_at(f)) for f in logs}
    while True:
        out = [f"== {m.parent.name}/{m.name}\n{m.read_text(errors='replace')}" for m in sorted(markers() - seen)]
        for f in logs:
            new = failed_at(f)[counts[f]:]
            out += [f"== {Path(f).name}: {a.strip()}\n{b.strip()}" for a, b in new]
        now = time.time()
        for t in q.tasks():
            if not (q.LOCKS / q.lockname(t)).exists() or (t["out"] / "features" / "meta.json").exists():
                continue
            files = [t["out"] / "embed.log", *(t["out"] / "features").glob("*.npy")]
            last = max((p.stat().st_mtime for p in files if p.exists()), default=None)
            if last is not None and now - last > STALL:
                out.append(f"== stalled: {t['out'].relative_to(q.R)}: nothing written for {(now - last) / 60:.0f} min")
        if out:
            print(time.strftime("%m-%d %H:%M"), flush=True)
            print("\n".join(out), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
