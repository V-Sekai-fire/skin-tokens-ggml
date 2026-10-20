#!/usr/bin/env python3
"""gstat_diff.py A.log B.log [tol]: line up GGML_RD_GRAPH_STATS lines of two runs and name the first
graph whose last node differs by more than tol (relative, on mean/absmean/absmax and the samples)."""
import re, sys

PAT = re.compile(r"GSTAT (\d+) '([^']*)' (\S+) (\S+)(.*)")


def load(path):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        rest = m.group(5)
        vals = dict(re.findall(r"(\w+)=([^ ]+)", rest))
        rows.append((int(m.group(1)), m.group(2), m.group(3), vals, rest.strip()))
    return rows


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-6)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
    print(f"graphs: {len(a)} vs {len(b)}")
    worst = 0.0
    for ra, rb in zip(a, b):
        if ra[1] != rb[1] or ra[2] != rb[2]:
            print(f"STRUCTURE differs at graph {ra[0]}: {ra[1]} {ra[2]} vs {rb[1]} {rb[2]}")
            return 1
        va, vb = ra[3], rb[3]
        if "mean" not in va or "mean" not in vb:
            continue
        d = max(rel(float(va[k]), float(vb[k])) for k in ("mean", "absmean", "absmax"))
        sa = [float(x) for x in va["v"].split(",")]
        sb = [float(x) for x in vb["v"].split(",")]
        d = max(d, max(rel(x, y) for x, y in zip(sa, sb)))
        if va.get("nonfinite") != vb.get("nonfinite"):
            d = max(d, 1.0)
        worst = max(worst, d)
        if d > tol:
            print(f"FIRST DIFF graph {ra[0]} '{ra[1]}' {ra[2]}: rel {d:.3g}\n  A: {ra[4]}\n  B: {rb[4]}")
            return 1
    print(f"no graph differs over {tol} (worst {worst:.3g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
