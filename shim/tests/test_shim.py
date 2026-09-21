import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import gfr_torch as g


def entry(seq, dur=10.0, state="completed", pg=0, name="nccl:all_reduce"):
    return {"collective_seq_id": seq, "pg_id": pg, "profiling_name": name, "state": state,
            "duration_ms": dur, "time_created_ns": 1_000 + seq,
            "input_sizes": [[1024, 1024]], "input_dtypes": ["Float"]}


def test_op_name():
    assert g.op_name("nccl:all_reduce") == "AllReduce"
    assert g.op_name("nccl:reduce_scatter_tensor_coalesced".replace("_coalesced", "")) == "ReduceScatterTensor"


def test_only_completed_timed_entries_and_units():
    tr = {"entries": [entry(1), entry(2, state="started"), entry(3, dur=None), entry(4, dur=2.5)]}
    evs = g.entries_to_events(tr, rank=3, node_id="n", gpu_uuid="GPU-1")
    assert [e["event"]["seq_id"] for e in evs] == [1, 4]
    e = evs[1]["event"]
    assert e["duration_ns"] == 2_500_000 and e["rank"] == 3 and e["op_type"] == "AllReduce"
    assert e["data_size"] == 1024 * 1024 * 4 and e["pg_id"] == "0"


def test_dedupe_across_dumps():
    seen = set()
    a = g.entries_to_events({"entries": [entry(1), entry(2)]}, 0, "n", seen=seen)
    b = g.entries_to_events({"entries": [entry(2), entry(3)]}, 0, "n", seen=seen)
    assert len(a) == 2 and [e["event"]["seq_id"] for e in b] == [3]


class FakeNVML:
    NVML_TEMPERATURE_GPU = 0; NVML_CLOCK_SM = 1; NVML_PCIE_UTIL_TX_BYTES = 0; NVML_PCIE_UTIL_RX_BYTES = 1
    NVML_MEMORY_ERROR_TYPE_CORRECTED = 0; NVML_MEMORY_ERROR_TYPE_UNCORRECTED = 1; NVML_VOLATILE_ECC = 0
    class U: gpu = 91; memory = 40
    def nvmlInit(self): pass
    def nvmlDeviceGetHandleByIndex(self, i): return i
    def nvmlDeviceGetUUID(self, h): return b"GPU-abc"
    def nvmlDeviceGetTemperature(self, h, t): return 77
    def nvmlDeviceGetPowerUsage(self, h): return 250_000
    def nvmlDeviceGetUtilizationRates(self, h): return self.U()
    def nvmlDeviceGetClockInfo(self, h, c): return 1500
    def nvmlDeviceGetCurrentClocksThrottleReasons(self, h): return 0x40
    def nvmlDeviceGetPcieThroughput(self, h, k): return 2_000_000
    def nvmlDeviceGetTotalEccErrors(self, h, t, v): raise RuntimeError("not supported")  # consumer GPU


def test_sampler_units_and_independent_degradation():
    s = g.NVMLSampler(0, nvml=FakeNVML())
    m, p = s.sample(rank=1, node_id="n")
    e = m["event"]
    assert m["type"] == "gpu_metric" and e["gpu_uuid"] == "GPU-abc"
    assert e["power_watts"] == 250.0 and e["throttle_reasons"] == 0x40 and e["sm_clock_mhz"] == 1500.0
    assert e["ecc_errors_sbe"] == 0 and e["ecc_errors_dbe"] == 0  # unsupported -> 0, others still read
    assert p["event"]["read_bw_mbs"] == 2000.0  # KB/s -> MB/s


def test_recorder_writes_jsonl_and_survives_bad_dump(tmp_path):
    calls = {"n": 0}
    def dump():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return {"entries": [entry(calls["n"])]}
    r = g.Recorder(str(tmp_path), rank=2, sampler=g.NVMLSampler(0, nvml=FakeNVML()), dump_fn=dump)
    assert r.poll_once() == 3 and r.poll_once() == 2  # second dump fails; NVML still recorded
    lines = [json.loads(l) for l in open(r.path)]
    assert {l["type"] for l in lines} == {"nccl_collective", "gpu_metric", "pcie_bandwidth"}
    assert all(l["event"]["rank"] == 2 for l in lines)
