"""Spec: specs/gfr-repo-hygiene-pyc.md.

Expected to fail until the spec is implemented. Delete the xfail marker when you
make it pass (strict=True: a passing test with the marker still present fails CI).
"""
import os
import subprocess

import pytest

pytestmark = pytest.mark.xfail(strict=True, reason="spec gfr-repo-hygiene-pyc: not implemented yet")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_no_bytecode_is_tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    bad = [p for p in out.splitlines() if p.endswith(".pyc") or "/__pycache__/" in p or p.startswith("__pycache__/")]
    assert not bad, bad


def test_gitignore_covers_bytecode():
    lines = {l.strip() for l in open(os.path.join(ROOT, ".gitignore"))}
    assert "__pycache__/" in lines and "*.pyc" in lines
