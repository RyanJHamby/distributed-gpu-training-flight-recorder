"""Score one run directory (rank*.jsonl + truth.json) against ground truth.

  python bench/score_real.py runs/power_cap-1 [--gfr bin/gfr]   # appends to results.jsonl
"""
import argparse, glob, json, os, subprocess, sys, tempfile


def merge(run_dir: str) -> str:
    out = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for p in sorted(glob.glob(os.path.join(run_dir, "rank*.jsonl"))):
        out.write(open(p).read())
    out.close()
    return out.name


def clock_skew_warning(run_dir: str):
    """Attribution joins NVML samples to collectives on one host clock. Verify."""
    for p in glob.glob(os.path.join(run_dir, "rank*.jsonl")):
        first = {}
        for line in open(p):
            e = json.loads(line)
            first.setdefault(e["type"], e["event"]["timestamp_ns"])
        if "nccl_collective" in first and "gpu_metric" in first:
            d = abs(first["nccl_collective"] - first["gpu_metric"]) / 1e9
            if d > 10:
                return f"WARNING: collective and NVML timestamps differ by {d:.0f}s in {p}: clock domains mismatch"
    return None


def score(run_dir: str, gfr: str) -> dict:
    truth = json.load(open(os.path.join(run_dir, "truth.json"))) if os.path.exists(
        os.path.join(run_dir, "truth.json")) else {"mode": "clean", "expected_causes": []}
    merged = merge(run_dir)
    try:
        rep = json.loads(subprocess.run([gfr, "replay", merged, "--json"], capture_output=True,
                                        text=True, check=True).stdout)
    finally:
        os.unlink(merged)
    attrs = rep["attributions"]
    res = {"run": os.path.basename(run_dir.rstrip("/")), "mode": truth["mode"], "flagged": [a["anomaly"]["straggler_rank"] for a in attrs],
           "warning": clock_skew_warning(run_dir)}
    if truth["mode"] == "clean":
        res["detected"] = None
        res["false_positive"] = len(attrs) > 0
        return res
    mine = [a for a in attrs if a["anomaly"]["straggler_rank"] == truth["rank"]]
    res["detected"] = bool(mine)
    res["false_positive"] = len(attrs) > len(mine)
    top = mine[0]["causal_chain"][0]["cause"] if mine else None
    res["top_cause"] = top
    res["attributed"] = top in truth["expected_causes"]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--gfr", default="bin/gfr")
    ap.add_argument("--results", default="results.jsonl")
    a = ap.parse_args()
    r = score(a.run_dir, a.gfr)
    print(json.dumps(r))
    with open(a.results, "a") as f:
        f.write(json.dumps(r) + "\n")
