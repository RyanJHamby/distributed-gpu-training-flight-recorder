"""Spec: specs/gfr-shim-schema-probe.md.

Expected to fail until the spec is implemented. Delete the xfail marker when you
make it pass (strict=True: a passing test with the marker still present fails CI).
"""
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.xfail(strict=True, reason="spec gfr-shim-schema-probe: not implemented yet")

PROBE = os.path.join(os.path.dirname(__file__), "..", "probe_flight_recorder.py")


def entry(seq, **over):
    e = {"collective_seq_id": seq, "pg_id": 0, "profiling_name": "nccl:all_reduce",
         "state": "completed", "duration_ms": 10.0, "time_created_ns": 1_000 + seq,
         "input_sizes": [[8]], "input_dtypes": ["Float"]}
    e.update(over)
    return {k: v for k, v in e.items() if v is not ...}


def run(tmp_path, trace):
    p = tmp_path / "dump.json"
    p.write_text(json.dumps(trace))
    r = subprocess.run([sys.executable, PROBE, "--from-json", str(p)],
                       capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout + r.stderr


def test_good_dump_passes(tmp_path):
    rc, out = run(tmp_path, {"entries": [entry(1), entry(2)]})
    assert rc == 0, out


def test_renamed_sequence_field_is_named(tmp_path):
    bad = [entry(i, collective_seq_id=...) for i in (1, 2)]
    rc, out = run(tmp_path, {"entries": bad})
    assert rc == 1 and "collective_seq_id" in out, out


def test_missing_durations_points_at_timing_flag(tmp_path):
    rc, out = run(tmp_path, {"entries": [entry(i, duration_ms=None) for i in (1, 2)]})
    assert rc == 1 and "TORCH_NCCL_ENABLE_TIMING" in out, out


def test_no_entries_fails(tmp_path):
    rc, out = run(tmp_path, {"entries": []})
    assert rc == 1 and "entries" in out, out


def test_unreadable_input_is_a_usage_error(tmp_path):
    p = tmp_path / "nope.json"
    r = subprocess.run([sys.executable, PROBE, "--from-json", str(p)],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "nope.json" in r.stderr and "Traceback" not in r.stderr
