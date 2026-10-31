---
id: gfr-repo-hygiene-pyc
goal: Compiled Python bytecode is no longer tracked in git and `.gitignore` keeps it out, so running the test suite no longer dirties the working tree.
acceptance:
  - python3 -m pytest -q --runxfail shim/tests/test_repo_hygiene.py
  - python3 -c "import sys; sys.exit(int('xfail' in open('shim/tests/test_repo_hygiene.py').read().lower()))"
  - python3 -m pytest -q shim/tests bench/tests
scope:
  - .gitignore
  - shim/tests/test_repo_hygiene.py
  - shim/__pycache__/**
  - shim/tests/__pycache__/**
  - bench/__pycache__/**
  - bench/tests/__pycache__/**
non_goals:
  - do not touch any .py source, Go code, CI config or other ignore rules
  - do not rewrite history; a normal `git rm -r --cached` commit is enough
  - do not delete anything except the tracked bytecode listed below
risk_tier: 1
max_runs: 2
autonomy: inherit
status: ready
issue: 9
---

Priority suggestion: p1 (lowest of the three; housekeeping, but it reproduces).

## Reproduced on origin/main b68af62

`git ls-files | grep -c pyc` prints 7: `bench/__pycache__/{inject_fault,score_real,train_ddp}.cpython-314.pyc`, `bench/tests/__pycache__/test_score.cpython-314-pytest-9.0.2.pyc`, `shim/__pycache__/gfr_torch.cpython-314.pyc`, `shim/tests/__pycache__/test_shim.cpython-314{,-pytest-9.0.2}.pyc`. `.gitignore` has no Python entries, so every `pytest` run shows these as modified (observed) and new test files add untracked bytecode.

## What is true when done

- `git ls-files` lists no `.pyc` and nothing under a `__pycache__/` directory.
- `.gitignore` contains the lines `__pycache__/` and `*.pyc`.
- Do it with `git rm -r --cached` on the four `__pycache__` directories above (the files stay on disk, ignored), plus the `.gitignore` edit. Remove the module-level xfail from the new test file (second acceptance line enforces it).

## Why the tests are shaped this way

CI runs plain `pytest`; both tests fail today, so they are `xfail(strict=True)` and CI stays green. Acceptance uses `--runxfail`: non-zero now, zero once done.

## Log
- 2026-10-31 draft
- 2026-10-31 ready, issue #9 (handshake was provisional; repo not yet onboarded with ap init; waiting for the owner's agent-ready label)
