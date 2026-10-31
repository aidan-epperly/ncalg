"""Performance canary: build and run examples/canary.rs, compare with this
machine's baseline (canary/baseline-<host>.json, a local file that is not
committed: timings only compare on the same machine).

    python scripts/canary.py              compare; exit 1 on a regression
    python scripts/canary.py --update     write the baseline (first run on a machine, or an intended change)
    python scripts/canary.py --repeat 3   runs per instance (default 2; the fastest counts)

Counters (iterations, operator products, factorization retries, kernel
searches) are deterministic for a given build, so an increase beyond a small
allowance is a regression, and sizes and statuses must match exactly.
Values must agree to 1e-6 (relative).  Times are flagged above 1.5 times the
baseline and 0.2 s.  Run it on an otherwise idle machine.  See
CONTRIBUTING.md (Performance).  Standard library only.
"""
import datetime
import json
import os
import socket
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMES = ["build", "solve", "post", "presolve", "elimination", "dualize", "setup",
         "schur_assembly", "schur_factor", "newton_solves", "iterate"]
# counter: (relative, absolute) allowance for an increase
COUNTERS = {"iterations": (0.0, 2), "operator_products": (0.3, 4), "factor_retries": (0.0, 1),
            "kernel_searched": (0.0, 0), "kernel_rows": (0.2, 100), "solves": (0.0, 0)}
EXACT = ["num_vars", "blocks", "status"]
TIME_RATIO, TIME_FLOOR, VALUE_TOL = 1.5, 0.2, 1e-6


def baseline_path():
    return os.path.join(ROOT, "canary", "baseline-%s.json" % socket.gethostname().lower())


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def run(repeat):
    out = os.path.join(ROOT, "target", "canary-out.json")
    cmd = ["cargo", "run", "--release", "--quiet", "--example", "canary", "--", out, "--repeat", str(repeat)]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit("canary: the example failed to build or run")
    with open(out) as f:
        return json.load(f)


def compare(base, new):
    """(regressions, notes), one line each."""
    bad, notes = [], []
    old = {r["name"]: r for r in base["instances"]}
    for r in new:
        b = old.get(r["name"])
        if b is None:
            notes.append("%s: new instance (not in the baseline)" % r["name"])
            continue
        name = r["name"]
        for k in EXACT:
            if r[k] != b[k]:
                bad.append("%s: %s %s, baseline %s" % (name, k, r[k], b[k]))
        if r["value"] is not None and b["value"] is not None:
            rel = abs(r["value"] - b["value"]) / max(1.0, abs(b["value"]))
            if rel > VALUE_TOL:
                bad.append("%s: value %.10g, baseline %.10g (relative change %.1e)" % (name, r["value"], b["value"], rel))
        # the certificate must certify the solve's value
        if r.get("certified") is not None and r["value"] is not None:
            gap = abs(r["certified"] - r["value"]) / max(1.0, abs(r["value"]))
            if gap > VALUE_TOL:
                bad.append("%s: certified %.10g, value %.10g (relative gap %.1e)" % (name, r["certified"], r["value"], gap))
        if r["warnings"] > b["warnings"]:
            bad.append("%s: %d solver warning(s)" % (name, r["warnings"]))
        for k, (rel, ab) in COUNTERS.items():
            if r[k] > b[k] * (1 + rel) + ab:
                bad.append("%s: %s %d, baseline %d" % (name, k, r[k], b[k]))
            elif r[k] != b[k]:
                notes.append("%s: %s %d, baseline %d" % (name, k, r[k], b[k]))
        for k in TIMES:
            t, t0 = r.get(k) or 0.0, b.get(k) or 0.0
            if t > TIME_RATIO * t0 and t - t0 > TIME_FLOOR:
                bad.append("%s: %s %.3f s, baseline %.3f s" % (name, k, t, t0))
    return bad, notes


def table(new, base):
    old = {r["name"]: r for r in base["instances"]} if base else {}
    print("%-16s %18s %18s %18s %10s %8s" % ("instance", "build s (base)", "solve s (base)", "post s (base)", "iters", "warn"))
    for r in new:
        b = old.get(r["name"], {})

        def cell(k):
            return "%8.3f (%7s)" % (r[k], "%.3f" % b[k] if k in b else "-")
        print("%-16s %18s %18s %18s %10d %8d" % (r["name"], cell("build"), cell("solve"), cell("post"),
                                                 r["iterations"], r["warnings"]))


def main():
    update = "--update" in sys.argv
    repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 2
    path = baseline_path()
    base = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            base = json.load(f)
    new = run(repeat)
    table(new, base)
    total = sum(r["solve"] + r["build"] + r["post"] for r in new)
    if update or base is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        meta = {"host": socket.gethostname(), "commit": git("rev-parse", "--short", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "date": datetime.date.today().isoformat(), "repeat": repeat, "instances": new}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        print("canary: baseline %s written (%d instances, %.1f s)" % (os.path.relpath(path, ROOT), len(new), total))
        return
    bad, notes = compare(base, new)
    for n in notes:
        print("note: " + n)
    for b in bad:
        print("REGRESSION: " + b)
    base_total = sum(r["solve"] + r["build"] + r["post"] for r in base["instances"])
    print("canary: %d instances, %d regression(s), %.1f s against %.1f s for baseline %s (%s)" % (
        len(new), len(bad), total, base_total, base["commit"], base["date"]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
