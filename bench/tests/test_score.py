"""score_real against the simulator's traces (no GPU): checks the scoring
logic and the merge -> `gfr replay --json` -> compare path."""
import json, os, shutil, subprocess, sys
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "bench"))
import score_real  # noqa: E402

GFR = os.path.join(ROOT, "bin", "gfr")
pytestmark = pytest.mark.skipif(not os.path.exists(GFR), reason="build bin/gfr first (make build)")


def split_by_rank(src, dst):
    os.makedirs(dst)
    files = {}
    for line in open(src):
        r = json.loads(line)["event"]["rank"]
        if r not in files:  # setdefault(r, open(...)) would reopen and truncate on every line
            files[r] = open(os.path.join(dst, f"rank{r}.jsonl"), "w")
        files[r].write(line)
    for f in files.values():
        f.close()


def write_truth(d, obj):
    with open(d + "/truth.json", "w") as f:
        json.dump(obj, f)


def test_detects_and_attributes(tmp_path):
    d = str(tmp_path / "run")
    split_by_rank(os.path.join(ROOT, "testdata", "thermal.jsonl"), d)
    write_truth(d, {"mode": "power_cap", "rank": 2, "expected_causes": ["thermal_throttle"]})
    r = score_real.score(d, GFR)
    assert r["detected"] and r["attributed"] and not r["false_positive"] and r["top_cause"] == "thermal_throttle"


def test_wrong_expectation_is_reported_not_hidden(tmp_path):
    d = str(tmp_path / "run")
    split_by_rank(os.path.join(ROOT, "testdata", "thermal.jsonl"), d)
    write_truth(d, {"mode": "x", "rank": 5, "expected_causes": ["ecc_errors"]})
    r = score_real.score(d, GFR)
    assert not r["detected"] and r["false_positive"]  # flagged rank 2, truth said rank 5


def test_clean_run(tmp_path):
    d = str(tmp_path / "run")
    split_by_rank(os.path.join(ROOT, "testdata", "clean.jsonl"), d)
    r = score_real.score(d, GFR)
    assert r["mode"] == "clean" and r["false_positive"] is False
