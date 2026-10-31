---
id: gfr-shim-schema-probe
goal: A small script, `shim/probe_flight_recorder.py --from-json DUMP`, checks a saved flight-recorder dump against exactly the fields the shim reads and prints which are missing or unusable, so the real-GPU smoke session starts with a one-command schema check instead of guesswork.
acceptance:
  - python3 -m pytest -q --runxfail shim/tests/test_probe_flight_recorder.py
  - python3 -c "import sys; sys.exit(int('xfail' in open('shim/tests/test_probe_flight_recorder.py').read().lower()))"
  - python3 -m pytest -q shim/tests bench/tests
scope:
  - shim/probe_flight_recorder.py
  - shim/tests/test_probe_flight_recorder.py
non_goals:
  - no live mode, no `import torch`, no NVML: the script only reads a JSON file
  - do not modify shim/gfr_torch.py; import its helpers if useful
  - do not edit README, bench/README.md or docs/ (the owner updates the runbook after a real run)
  - no new dependencies; stdlib only
  - do not claim any field name is verified; this tool exists to find out
risk_tier: 1
max_runs: 3
autonomy: inherit
---

Priority suggestion: p0 among these three. README's own Status table says the entry field names are "unverified" and `bench/README.md` smoke check 1 is "fix the shim if field names are wrong". This turns that into a repeatable check, built and tested without a GPU.

## Interface (the tests in `shim/tests/test_probe_flight_recorder.py` pin it)

`python3 shim/probe_flight_recorder.py --from-json PATH` where PATH holds the JSON of a flight-recorder dump: `{"entries": [ {...}, ... ]}` (what `pickle.loads(_dump_nccl_trace(...))` returns, saved as JSON by the owner).

Fields the shim reads (from `entries_to_events` in `shim/gfr_torch.py`): `collective_seq_id`, `pg_id`, `profiling_name`, `state`, `duration_ms`, `time_created_ns`, `input_sizes`, `input_dtypes`.

Exit codes and output (stdout or stderr, tests read both):
- 0: at least one entry; every field above is present in at least one entry; at least one entry has `state == "completed"` and a non-null `duration_ms`; and `gfr_torch.entries_to_events` yields at least one event from the dump. Print a short OK summary with the entry count.
- 1: any of the above fails. One line per problem. A field present in no entry: `MISSING <field>`. No entries: a line containing `entries`. No completed entry with a duration: a line containing `TORCH_NCCL_ENABLE_TIMING` (the env var the shim docstring requires).
- 2: PATH unreadable or not valid JSON. One line on stderr naming PATH, no traceback.

## Out of scope, owner task

Running it against a real dump needs a GPU box with PyTorch; that is the owner's smoke session (`bench/README.md`, Session 1), not agent work. Saving a dump to JSON for the script is a one-liner the owner can add to the runbook later.

## Why the tests are shaped this way

CI runs plain `pytest`, so the tests are `xfail(strict=True)` and CI stays green. The acceptance command uses `--runxfail`: non-zero now (the script does not exist), zero when done. Remove the module-level `pytestmark` xfail (the second acceptance line enforces it).

## Log
- 2026-10-31 draft
