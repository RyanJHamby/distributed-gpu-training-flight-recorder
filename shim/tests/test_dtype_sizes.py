"""Spec: specs/gfr-shim-dtype-sizes.md.

Expected to fail until the spec is implemented. Delete the XFAIL markers
when you make it pass; strict=True means a passing test with the marker still
present fails CI, so a marker cannot be left behind.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gfr_torch as g

XFAIL = pytest.mark.xfail(strict=True, reason="spec gfr-shim-dtype-sizes: not implemented yet")

# PyTorch's flight recorder records c10 dtype names ("BFloat16"); torch-style
# spellings ("torch.bfloat16") are accepted too. Bytes per element.
KNOWN = {
    "Float": 4, "Double": 8, "Half": 2, "BFloat16": 2, "Long": 8, "Int": 4,
    "Short": 2, "Char": 1, "Byte": 1, "Bool": 1,
    "ComplexFloat": 8, "ComplexDouble": 16,
    "Float8_e4m3fn": 1, "Float8_e5m2": 1,
    "torch.float32": 4, "torch.float64": 8, "torch.float16": 2, "torch.bfloat16": 2,
    "torch.int64": 8, "torch.int32": 4, "torch.int16": 2, "torch.int8": 1,
    "torch.uint8": 1, "torch.bool": 1,
}


# Spellings the current code already sizes correctly stay as plain regression
# tests (no marker). The rest are expected failures until the spec is done; when
# you fix them, remove XFAIL from the parametrize list and from the tests below.
WORKS_TODAY = {"Float", "Double", "Half", "Long", "Int", "Short", "Char", "Byte", "Bool",
               "torch.float32", "torch.int32", "torch.bool"}
CASES = [pytest.param(d, w, marks=[] if d in WORKS_TODAY else XFAIL, id=d) for d, w in sorted(KNOWN.items())]


@pytest.mark.parametrize("dtype,width", CASES)
def test_bytes_per_element(dtype, width):
    assert g._nbytes([[10]], [dtype]) == 10 * width


@XFAIL
def test_multiple_tensors_sum():
    assert g._nbytes([[2, 3], [4]], ["BFloat16", "Float"]) == 2 * 3 * 2 + 4 * 4


@XFAIL
def test_unknown_dtype_contributes_zero_not_a_guess():
    assert g._nbytes([[10], [5]], ["NotADtype", "Half"]) == 5 * 2


@XFAIL
def test_entries_to_events_reports_bf16_size():
    entry = {"collective_seq_id": 1, "pg_id": 0, "profiling_name": "nccl:all_reduce",
             "state": "completed", "duration_ms": 1.0, "time_created_ns": 1,
             "input_sizes": [[1024, 1024]], "input_dtypes": ["BFloat16"]}
    (ev,) = g.entries_to_events({"entries": [entry]}, rank=0, node_id="n")
    assert ev["event"]["data_size"] == 1024 * 1024 * 2
