"""Inject a fault into one GPU/rank for a fixed window and log ground truth.

  python bench/inject_fault.py probe
  python bench/inject_fault.py run --mode power_cap --gpu 2 --start-after 60 \
      --duration 60 --truth runs/x/truth.json --control runs/x/fault.json

DRAFT: written without GPU access. Whatever a mode changes is ALWAYS reverted
(finally + signal handlers), because a leaked power cap or clock lock would
poison later runs and the rented box. Assumes single node, rank == GPU index.
"""
import argparse, json, os, signal, subprocess, sys, time

EXPECTED = {  # acceptable top causes per mode; contention has no hardware signal by design
    "power_cap": ["power_throttle", "clock_reduced"],
    "clock_lock": ["clock_reduced"],
    "host_stall": ["host_stall"],
    "contention": ["unknown"],
    "clean": [],
}


def smi(*args) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout.strip()


def query(gpu: int, field: str) -> str:
    return smi("-i", str(gpu), f"--query-gpu={field}", "--format=csv,noheader,nounits")


class Fault:
    def __init__(self, mode, gpu, control, clock_mhz):
        self.mode, self.gpu, self.control, self.clock_mhz, self.proc = mode, gpu, control, clock_mhz, None

    def apply(self):
        g = str(self.gpu)
        if self.mode == "power_cap":
            self.orig = query(self.gpu, "power.limit")
            smi("-i", g, "-pl", str(int(float(query(self.gpu, "power.min_limit")) + 1)))
        elif self.mode == "clock_lock":
            smi("-i", g, "-lgc", f"{self.clock_mhz},{self.clock_mhz}")
        elif self.mode == "contention":
            burner = ("import torch;a=torch.randn(8192,8192,device='cuda');\n"
                      "while True: a=(a@a).clamp_(-1,1)")
            self.proc = subprocess.Popen([sys.executable, "-c", burner],
                                         env={**os.environ, "CUDA_VISIBLE_DEVICES": g})
        elif self.mode == "host_stall":
            with open(self.control, "w") as f:
                json.dump({"host_stall": {"rank": self.gpu, "ms": 40}}, f)

    def revert(self):
        g = str(self.gpu)
        try:
            if self.mode == "power_cap":
                smi("-i", g, "-pl", str(int(float(self.orig))))
            elif self.mode == "clock_lock":
                smi("-i", g, "-rgc")
            elif self.mode == "contention" and self.proc:
                self.proc.kill(); self.proc.wait()
            elif self.mode == "host_stall" and os.path.exists(self.control):
                os.remove(self.control)
        except Exception as e:  # report loudly; do not hide a failed revert
            print(f"REVERT FAILED for {self.mode}: {e}", file=sys.stderr)
            raise


def run(a):
    if a.mode not in EXPECTED or a.mode == "clean":
        sys.exit(f"unknown mode {a.mode}")
    f = Fault(a.mode, a.gpu, a.control, a.clock_mhz)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # SystemExit -> finally runs
    time.sleep(a.start_after)
    start = time.time_ns()
    try:
        f.apply()
        time.sleep(a.duration)
    finally:
        f.revert()
    with open(a.truth, "w") as t:
        json.dump({"mode": a.mode, "rank": a.gpu, "gpu": a.gpu, "start_ns": start,
                   "end_ns": time.time_ns(), "expected_causes": EXPECTED[a.mode]}, t)


def probe(_):
    """Which privileged modes does this host allow? Non-destructive: sets the
    current value back onto itself."""
    out = {}
    try:
        cur = int(float(query(0, "power.limit")))
        smi("-i", "0", "-pl", str(cur))
        out["power_cap"] = True
    except Exception as e:
        out["power_cap"] = f"no: {e}"
    try:
        smi("-i", "0", "-lgc", "0,100000")  # allow-all range, then reset
        smi("-i", "0", "-rgc")
        out["clock_lock"] = True
    except Exception as e:
        out["clock_lock"] = f"no: {e}"
    out["contention"] = out["host_stall"] = True  # unprivileged
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("probe").set_defaults(fn=probe)
    r = sp.add_parser("run")
    r.add_argument("--mode", required=True)
    r.add_argument("--gpu", type=int, required=True)
    r.add_argument("--start-after", type=float, default=60)
    r.add_argument("--duration", type=float, default=60)
    r.add_argument("--clock-mhz", type=int, default=600)
    r.add_argument("--truth", required=True)
    r.add_argument("--control", default="fault.json")
    r.set_defaults(fn=run)
    a = p.parse_args()
    a.fn(a)
