---
id: gfr-shim-dtype-sizes
goal: The Python shim computes `data_size` correctly for every dtype name PyTorch's flight recorder can emit (BFloat16 and Half-class types currently report 4 bytes per element instead of 2), and no longer guesses 4 bytes for dtypes it does not know.
acceptance:
  - python3 -m pytest -q --runxfail shim/tests/test_dtype_sizes.py
  - python3 -c "import sys; sys.exit(int('xfail' in open('shim/tests/test_dtype_sizes.py').read().lower()))"
  - python3 -m pytest -q shim/tests bench/tests
scope:
  - shim/gfr_torch.py
  - shim/tests/test_dtype_sizes.py
non_goals:
  - do not touch the Go code, testdata/, or the JSONL field names (the Go contract test must keep passing)
  - do not change any other function in shim/gfr_torch.py (op_name, NVMLSampler, Recorder)
  - do not import torch or any new dependency
  - do not edit existing tests in shim/tests/test_shim.py
  - do not claim the dtype names are verified against real PyTorch; they are not (see body)
risk_tier: 1
max_runs: 2
autonomy: inherit
---

Priority suggestion: p1. This is a real, GPU-free bug, but `data_size` is carried on the wire and never used by detection or attribution today, so no report changes. It matters when anyone starts reading it.

## The bug (reproduced on origin/main b68af62)

`shim/gfr_torch.py::_nbytes` looks up `str(dt).replace("torch.", "").capitalize()` in `_DTYPE_BYTES` and defaults to 4. `"BFloat16".capitalize()` is `"Bfloat16"`, which is not a key, so it silently becomes 4 bytes. Observed: `_nbytes([[10]], ["BFloat16"]) == 40` (should be 20); same for `torch.bfloat16`, `torch.float16`, `torch.int64`, `Float8_*` and complex types. bf16 is the common training dtype, so the shim reports 2x the real payload.

## What is true when done

- Bytes per element: Float 4, Double 8, Half 2, BFloat16 2, Long 8, Int 4, Short 2, Char 1, Byte 1, Bool 1, ComplexFloat 8, ComplexDouble 16, Float8_e4m3fn 1, Float8_e5m2 1, plus the `torch.<lowercase>` spellings listed in `KNOWN` in the test file (case-insensitive lookup is the simplest approach).
- An unknown dtype contributes 0 bytes for that tensor (data_size is then a lower bound), instead of a guess of 4. This is a decision made in this spec; the test pins it.
- `tests/test_dtype_sizes.py` already exists, with `XFAIL` markers on the currently failing cases. Remove every `XFAIL` marker (the second acceptance line enforces that). Do not weaken or delete cases.

## Design notes and honesty

The dtype spellings are my recollection of how the flight recorder prints c10 dtype names; they are pinned by this test, not confirmed against a real PyTorch dump. `docs`/README already say the shim is unverified against real PyTorch; keep it that way. Do not edit README.

## Why the tests are shaped this way

CI runs plain `pytest`, so the new cases that fail today are marked `xfail(strict=True)` and CI stays green. The acceptance command uses `--runxfail`, which ignores the markers: it exits non-zero now and zero once the code is right. Cases that already pass are unmarked regression tests.

## Log
- 2026-10-31 draft
