---
id: gfr-ringbuf
goal: The per-rank ring buffer actually stores events - Push, Len and Range work, old events are overwritten in order when it wraps, capacity is a power of two, and a producer plus a Len() poller are clean under the race detector.
acceptance:
  - go test -race -tags spec_gfr_ringbuf ./internal/buffer/... -count=1
  - sh -c 'test -z "$(grep -l spec_gfr_ringbuf internal/buffer/ringbuffer_wrap_test.go)"'
  - go test -race ./internal/buffer/... -count=1
  - go vet ./internal/buffer/...
scope:
  - internal/buffer/ringbuffer.go
  - internal/buffer/ringbuffer_wrap_test.go
non_goals:
  - do not change any assertion in internal/buffer/ringbuffer_wrap_test.go; the only permitted edit is deleting its first line (`//go:build spec_gfr_ringbuf`) and the blank line after it, which makes the third command run the test in normal CI
  - do not edit internal/buffer/ringbuffer_test.go (its TODO comments may stay)
  - no mutexes: keep the atomic-index design; the buffer stays single-producer
  - no support for Range running concurrently with Push (documented as unsupported in a comment on Range); no multi-producer support
  - no changes to internal/correlator, internal/attribution, internal/agent, internal/types, no new dependencies
  - do not quote any benchmark or throughput number; no benchmarks in this spec
risk_tier: 2
max_runs: 3
autonomy: inherit
status: draft
---

## Why
`internal/buffer/ringbuffer.go` is the foundation of the recorder (README: "records system-level events in a ring buffer")
but `Push` and `Range` are TODO stubs and `Push` does nothing today. The existing `TestPushAndLen` has its assertion
commented out, so CI is green on a buffer that stores nothing. This is the largest real test gap in the repo that is
independent of the detector logic, and it can be fixed and verified with no GPU.

## Where
`internal/buffer/ringbuffer.go` (package `buffer`, 44 lines). The new test `internal/buffer/ringbuffer_wrap_test.go` is
already written (build tag `spec_gfr_ringbuf`); read it first, it is the contract. It reuses `makeEvent` from the existing
`ringbuffer_test.go` (same package).

## Required behaviour
- `NewRingBuffer(capacity)`: round capacity UP to the next power of two (1000 -> 1024, 3 -> 4); capacity <= 0 gives 1
  (today a non-power-of-two size makes `mask` wrong and a negative size panics in `make`). Careful: a naive
  `for size < uint64(capacity)` loops forever for negative capacity.
- `Push(event)`: store at `events[w & mask]`, then publish `writeIdx = w+1`. If the buffer is now over capacity, advance
  `readIdx` to `w+1-size` so the oldest event is dropped. Single producer only.
- `Len()`: always within `[0, capacity]`. Load `readIdx` BEFORE `writeIdx` and clamp, because the two atomic loads are not
  one snapshot and the producer can move both between them (today's `int(w - r)` can underflow or exceed capacity;
  the concurrency test catches this).
- `Range(fn)`: snapshot `writeIdx`/`readIdx`, clamp `r` to `w-size`, call `fn` oldest to newest, stop when `fn` returns
  false. It does not consume events (`Len()` unchanged afterwards). Add a comment: Range is only safe when no Push runs
  concurrently (slots are plain interface values); concurrent `Len()` with Push is safe.
- Do not add a lock. `go test -race` must be clean.

## Verify
Run the four acceptance commands in order. The second fails while the build-tag line is present, so delete the tag line
(plus the blank line after it) only after the first passes. `gofmt -l internal/buffer` must print nothing.

## Log
- 2026-10-30 draft
