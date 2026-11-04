"""qstatus.py — done / running / pending per output group of queue_all's task list."""
import collections
import sys
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from paths import CODE, WORK

sys.path.insert(0, CODE / "harness")
import queue_all as q

groups = collections.OrderedDict()
for t in q.tasks():
    g = t["out"].parent.name
    done = (t["out"] / "features" / "meta.json").exists()
    locked = (q.LOCKS / q.lockname(t)).exists()
    s = "done" if done else "run" if locked else "wait"
    groups.setdefault(g, collections.Counter())[s] += 1
    if s == "run":
        groups[g].setdefault("_running", [])
for g, c in groups.items():
    running = [t["table"] for t in q.tasks() if t["out"].parent.name == g
               and not (t["out"] / "features" / "meta.json").exists() and (q.LOCKS / q.lockname(t)).exists()]
    print(f"{g:24s} done {c['done']:3d}  run {c['run']:2d}  wait {c['wait']:3d}  {' '.join(running)}")
tot = collections.Counter()
for c in groups.values():
    tot.update({k: v for k, v in c.items() if k in ("done", "run", "wait")})
print(f"{'total':24s} done {tot['done']:3d}  run {tot['run']:2d}  wait {tot['wait']:3d}")
