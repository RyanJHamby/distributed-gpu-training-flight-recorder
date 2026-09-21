"""Small DDP transformer workload with the flight-recorder shim attached.

  torchrun --nproc_per_node=N bench/train_ddp.py --seconds 120 --out-dir runs/x

DRAFT: written without GPU access; expect to debug in the smoke session.
Needs env TORCH_NCCL_TRACE_BUFFER_SIZE and TORCH_NCCL_ENABLE_TIMING=1
(run_matrix.sh sets them). The host_stall fault is applied here, driven by a
control file, because a dataloader stall has to happen inside the rank.
"""
import argparse, json, os, sys, time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shim"))
import gfr_torch  # noqa: E402


def stall_ms(control: str, rank: int) -> float:
    try:
        with open(control) as f:
            s = json.load(f).get("host_stall")
        return float(s["ms"]) if s and int(s["rank"]) == rank else 0.0
    except (OSError, ValueError, KeyError):
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--control", default="fault.json")
    ap.add_argument("--no-recorder", action="store_true", help="baseline for overhead measurement")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    a = ap.parse_args()

    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(a.hidden, 16, 4 * a.hidden, batch_first=True), a.layers).cuda()
    model = DDP(model, device_ids=[local])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(a.batch, a.seq, a.hidden, device="cuda")

    os.makedirs(a.out_dir, exist_ok=True)
    rec = None
    if not a.no_recorder:
        rec = gfr_torch.Recorder(out_dir=a.out_dir)
        rec.start()

    # All ranks must stop on the same step or a collective is left unmatched:
    # rank 0 decides from its clock and broadcasts.
    t0, step_ms = time.time(), []
    stop = torch.zeros(1, device="cuda")
    step = 0
    while True:
        s = time.perf_counter()
        ms = stall_ms(a.control, rank)
        if ms:
            time.sleep(ms / 1000)
        opt.zero_grad(set_to_none=True)
        model(x).float().pow(2).mean().backward()
        opt.step()
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - s) * 1000)
        step += 1
        if step % 10 == 0:
            stop.fill_(1.0 if time.time() - t0 > a.seconds else 0.0)
            dist.broadcast(stop, src=0)
            if stop.item() > 0:
                break

    if rec:
        rec.stop()
    with open(os.path.join(a.out_dir, f"steps_rank{rank}.json"), "w") as f:
        json.dump(step_ms, f)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
