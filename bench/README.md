# Real-GPU validation runbook (budget: $10-20)

Status: **drafts, never run on a GPU.** Expect to fix things in the smoke session. Every number in the main README is synthetic until this has run.

## Budget

Cost = hourly price x hours. Check the live Vast.ai price before renting; I have not quoted one.

| Session | Hours | Purpose |
|---|---|---|
| 1. Smoke | ~1 | confirm flight-recorder field names, NVML fields, which injections the host permits |
| 2. Matrix | ~1 | 5 modes x 3 repeats x ~2.5 min = ~40 min |
| 3. Overhead | ~0.5 | step time with vs without the recorder |
| Reserve | ~2 | reruns after bugs |

About 4.5 h total. At $1.50/h that is ~$7; at $3/h ~$14. If the rate x 4.5 exceeds your cap, pick a cheaper box rather than cutting the reserve. Destroy the instance after every session; the storage clock runs while it is stopped.

## Instance

2 GPUs on one host (any generation with NVML; PCIe-only is fine), on-demand rather than interruptible (a preemption mid-run wastes the run), a PyTorch CUDA image, 30 GB disk. Ask, or test in the smoke session, whether the container allows `nvidia-smi -pl` and `-lgc`; if not, those modes are skipped automatically and the results table says so.

## Session 1: smoke (do not run the matrix yet)

```bash
# laptop
make build-linux && scp bin/gfr-linux <box>:gfr/bin/gfr && rsync -a bench shim <box>:gfr/
# box
cd gfr && pip install nvidia-ml-py pytest && python -m pytest -q shim/tests
python bench/inject_fault.py probe                       # which modes are allowed
MODES=clean REPEATS=1 SECONDS_PER_RUN=60 bench/run_matrix.sh
head -c 600 runs/clean-1/rank0.jsonl                     # collectives present? durations non-zero?
```

Check, and fix the shim if any fails:
1. `nccl_collective` lines exist with `duration_ns > 0` (else the flight-recorder field names in `shim/gfr_torch.py` are wrong; dump `_dump_nccl_trace` once and compare).
2. `gpu_metric` has a plausible temperature, power and `sm_clock_mhz`.
3. `score_real` prints no clock-domain WARNING.
4. The clean run reports zero anomalies. A false positive on a clean real run is the most important finding of the whole project: record it, do not tune it away silently.

## Session 2: matrix, then rsync results back and score/read them locally

```bash
NGPU=2 REPEATS=3 SECONDS_PER_RUN=120 MAX_MINUTES=60 bench/run_matrix.sh
rsync -a <box>:gfr/runs ./runs   # then destroy the instance
```

`MAX_MINUTES` is the spend cap: the runner will not start a run that would pass it.

## Session 3: overhead

Run `train_ddp.py` 3x with and 3x with `--no-recorder`; compare median step time from `steps_rank*.json`. Report the number, whatever it is.

## Honest expectations

- `contention` should come back `unknown` (no per-GPU signal): a correct result, reported as "located, not explained".
- `power_cap` may be reported as `power_throttle` or `clock_reduced`; both are accepted.
- Report failures in the README table. A result that does not match the synthetic scorecard is the point of this exercise.
