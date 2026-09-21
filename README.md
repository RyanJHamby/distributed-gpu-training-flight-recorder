# GPU Flight Recorder


A distributed training observability tool that attaches to GPU nodes, records system-level events in a ring buffer, and when anomalies occur, correlates timelines across ranks to produce automatic root-cause attribution.

## Why This Exists

Large-scale distributed GPU training suffers from straggler problems — a single slow rank can bottleneck an entire collective operation across hundreds of nodes. Identifying *why* a rank is slow (thermal throttling? ECC errors? PCIe contention?) requires correlating hardware telemetry with collective timing across all ranks simultaneously. GPU Flight Recorder does this automatically and continuously.

## Architecture

```
GPU Node 0                GPU Node 1                GPU Node N
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│     Agent        │      │     Agent        │      │     Agent        │
│  ┌────────────┐  │      │  ┌────────────┐  │      │  ┌────────────┐  │
│  │ Collectors  │  │      │  │ Collectors  │  │      │  │ Collectors  │  │
│  │ DCGM|NCCL  │  │      │  │ DCGM|NCCL  │  │      │  │ DCGM|NCCL  │  │
│  │ System     │  │      │  │ System     │  │      │  │ System     │  │
│  └─────┬──────┘  │      │  └─────┬──────┘  │      │  └─────┬──────┘  │
│        │         │      │        │         │      │        │         │
│  ┌─────▼──────┐  │      │  ┌─────▼──────┐  │      │  ┌─────▼──────┐  │
│  │ Ring Buffer │  │      │  │ Ring Buffer │  │      │  │ Ring Buffer │  │
│  └─────┬──────┘  │      │  └─────┬──────┘  │      │  └─────┬──────┘  │
└────────┼─────────┘      └────────┼─────────┘      └────────┼─────────┘
         │ gRPC                    │ gRPC                    │ gRPC
         └─────────────┬──────────┘──────────────────────────┘
                       │
              ┌────────▼────────┐
              │   Coordinator    │
              │  ┌────────────┐  │
              │  │ Correlator  │  │
              │  │ Timeline    │  │
              │  │ Alignment   │  │
              │  └──────┬─────┘  │
              │  ┌──────▼─────┐  │
              │  │ Attribution │  │
              │  │ Engine      │  │
              │  └──────┬─────┘  │
              └─────────┼────────┘
                        │
                   ┌────▼────┐
                   │ Report  │
                   │ JSON/CLI│
                   └─────────┘
```

## Quick Start

### Build

```bash
make build
```

### Run the Coordinator

```bash
./bin/gfr coordinator --listen :50051 --threshold-sigma 2.0
```

### Run an Agent

```bash
./bin/gfr agent --coordinator localhost:50051 --node-id gpu-node-0
```

### Run Tests

```bash
make test
```

### Local Multi-Node Simulation

```bash
docker-compose -f deploy/docker-compose.yaml up
```

## Development

```bash
# Set up development environment
./scripts/setup-dev.sh

# Generate protobuf code
make proto

# Run linter
make lint

# Inject a synthetic fault for testing
./scripts/inject-fault.sh
```

## License

MIT
