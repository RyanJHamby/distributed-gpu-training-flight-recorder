"""Per-rank telemetry shim: PyTorch NCCL flight-recorder + NVML -> JSONL.

Run inside each training process. It writes one JSONL file per rank in the
format `gfr replay` / the agent's `--tail` collector read:

    {"type": "nccl_collective" | "gpu_metric" | "pcie_bandwidth", "event": {...}}

Requirements: launch the job with TORCH_NCCL_TRACE_BUFFER_SIZE=<N> and
TORCH_NCCL_ENABLE_TIMING=1 (durations are only recorded with timing on).
NVML is optional and every metric degrades independently: consumer GPUs
report ECC as unsupported, for example.

UNVERIFIED against real hardware: the flight-recorder entry field names below
(`collective_seq_id`, `pg_id`, `profiling_name`, `duration_ms`, `state`,
`time_created_ns`, `input_sizes`, `input_dtypes`) follow PyTorch 2.x source as
remembered; they are pinned by tests/fixtures and must be confirmed in the GPU
smoke session before results are trusted.
"""
from __future__ import annotations

import json
import os
import pickle
import socket
import threading
import time
from typing import Any, Callable, Iterable

_DTYPE_BYTES = {
    "Float": 4, "Double": 8, "Half": 2, "BFloat16": 2, "Long": 8, "Int": 4,
    "Short": 2, "Char": 1, "Byte": 1, "Bool": 1,
}


def op_name(profiling_name: str) -> str:
    """'nccl:all_reduce' -> 'AllReduce'."""
    base = profiling_name.split(":", 1)[-1]
    return "".join(p.capitalize() for p in base.split("_")) or "Unknown"


def _nbytes(sizes: Iterable[Iterable[int]], dtypes: Iterable[str]) -> int:
    total = 0
    for shape, dt in zip(sizes or [], dtypes or []):
        n = 1
        for d in shape:
            n *= d
        total += n * _DTYPE_BYTES.get(str(dt).replace("torch.", "").capitalize(), 4)
    return total


def entries_to_events(trace: dict, rank: int, node_id: str, gpu_uuid: str = "",
                      seen: set | None = None) -> list[dict]:
    """Convert a flight-recorder dump into completed-collective events.

    Only entries that finished with a recorded duration are emitted (in-flight
    or untimed entries carry no lag information). `seen` de-duplicates across
    successive dumps of the same ring buffer (keyed on pg_id, seq).
    """
    seen = seen if seen is not None else set()
    out = []
    for e in trace.get("entries", []):
        if e.get("state") != "completed" or e.get("duration_ms") is None:
            continue
        if e.get("collective_seq_id") is None:
            continue
        key = (str(e.get("pg_id")), int(e["collective_seq_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": "nccl_collective", "event": {
            "timestamp_ns": int(e["time_created_ns"]), "rank": rank, "node_id": node_id,
            "gpu_uuid": gpu_uuid, "pg_id": key[0], "seq_id": key[1],
            "op_type": op_name(e.get("profiling_name", "")),
            "data_size": _nbytes(e.get("input_sizes"), e.get("input_dtypes")),
            "duration_ns": int(float(e["duration_ms"]) * 1e6),
        }})
    return out


class NVMLSampler:
    """Reads one GPU. `nvml` is injectable for tests (defaults to pynvml)."""

    def __init__(self, index: int, nvml: Any = None):
        if nvml is None:
            import pynvml as nvml  # nvidia-ml-py
        self.n = nvml
        self.n.nvmlInit()
        self.h = self.n.nvmlDeviceGetHandleByIndex(index)
        u = self.n.nvmlDeviceGetUUID(self.h)
        self.uuid = u.decode() if isinstance(u, bytes) else u

    def _try(self, fn: Callable[[], Any], default: Any = 0) -> Any:
        try:
            return fn()
        except Exception:  # NVMLError: not supported / no permission
            return default

    def sample(self, rank: int, node_id: str) -> list[dict]:
        n, h = self.n, self.h
        ts = time.time_ns()
        base = {"timestamp_ns": ts, "rank": rank, "node_id": node_id, "gpu_uuid": self.uuid}
        reasons_fn = getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            getattr(n, "nvmlDeviceGetCurrentClocksThrottleReasons", None)
        util = self._try(lambda: n.nvmlDeviceGetUtilizationRates(h), None)
        metric = dict(base,
            temperature_c=float(self._try(lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))),
            power_watts=self._try(lambda: n.nvmlDeviceGetPowerUsage(h)) / 1000.0,
            utilization_pct=float(util.gpu) if util else 0.0,
            mem_bandwidth_pct=float(util.memory) if util else 0.0,
            ecc_errors_sbe=int(self._try(lambda: n.nvmlDeviceGetTotalEccErrors(
                h, n.NVML_MEMORY_ERROR_TYPE_CORRECTED, n.NVML_VOLATILE_ECC))),
            ecc_errors_dbe=int(self._try(lambda: n.nvmlDeviceGetTotalEccErrors(
                h, n.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, n.NVML_VOLATILE_ECC))),
            sm_clock_mhz=float(self._try(lambda: n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM))),
            throttle_reasons=int(self._try(lambda: reasons_fn(h)) if reasons_fn else 0),
            # 0 = unknown (unsupported, or a container hides other PIDs), never "no processes".
            compute_procs=len(self._try(lambda: n.nvmlDeviceGetComputeRunningProcesses(h), [])))
        # NVML reports PCIe throughput in KB/s.
        rx = self._try(lambda: n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_RX_BYTES), None)
        tx = self._try(lambda: n.nvmlDeviceGetPcieThroughput(h, n.NVML_PCIE_UTIL_TX_BYTES), None)
        evs = [{"type": "gpu_metric", "event": metric}]
        if rx is not None and tx is not None:
            evs.append({"type": "pcie_bandwidth", "event": dict(base,
                read_bw_mbs=rx / 1000.0, write_bw_mbs=tx / 1000.0)})
        return evs


class Recorder:
    """Background thread: periodically dump the flight recorder + sample NVML.

    Usage (after dist.init_process_group):
        rec = gfr_torch.Recorder(out_dir="/tmp/gfr"); rec.start() ... rec.stop()
    """

    def __init__(self, out_dir: str, rank: int | None = None, local_rank: int | None = None,
                 interval_s: float = 0.5, dump_fn: Callable[[], dict] | None = None,
                 sampler: NVMLSampler | None = None, node_id: str | None = None):
        self.rank = int(os.environ.get("RANK", 0)) if rank is None else rank
        lr = int(os.environ.get("LOCAL_RANK", 0)) if local_rank is None else local_rank
        self.node_id = node_id or socket.gethostname()
        self.interval_s = interval_s
        self.dump_fn = dump_fn or _torch_dump
        self.sampler = sampler
        if self.sampler is None:
            try:
                self.sampler = NVMLSampler(lr)
            except Exception:
                self.sampler = None  # collectives-only mode
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"rank{self.rank}.jsonl")
        self._seen: set = set()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def poll_once(self) -> int:
        evs = []
        uuid = self.sampler.uuid if self.sampler else ""
        try:
            evs += entries_to_events(self.dump_fn(), self.rank, self.node_id, uuid, self._seen)
        except Exception:
            pass  # a bad dump must never take down training
        if self.sampler:
            evs += self.sampler.sample(self.rank, self.node_id)
        if evs:
            with open(self.path, "a") as f:
                for e in evs:
                    f.write(json.dumps(e) + "\n")
        return len(evs)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            self.poll_once()

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=5)
        self.poll_once()  # final flush


def _torch_dump() -> dict:
    import torch  # noqa: F401  (imported lazily so the shim's parsing is testable without torch)
    from torch._C._distributed_c10d import _dump_nccl_trace
    return pickle.loads(_dump_nccl_trace(includeCollectives=True, includeStackTraces=False, onlyActive=False))
