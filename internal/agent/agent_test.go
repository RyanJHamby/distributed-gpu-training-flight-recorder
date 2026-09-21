package agent

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/collector"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/coordinator"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/sim"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/trace"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/transport"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

func writeTrace(t *testing.T, f sim.Fault) (string, sim.Truth, int) {
	t.Helper()
	evs, truth := sim.Generate(sim.Params{Fault: f, Seed: 2, Steps: 300})
	p := filepath.Join(t.TempDir(), "t.jsonl")
	fh, _ := os.Create(p)
	if err := trace.Write(fh, evs); err != nil {
		t.Fatal(err)
	}
	fh.Close()
	return p, truth, len(evs)
}

func TestAgentRecordsLocallyAndExitsWhenReplayEnds(t *testing.T) {
	p, _, n := writeTrace(t, sim.Clean)
	a, _ := New(types.DefaultAgentConfig())
	a.AddCollector(collector.NewReplayCollector(p, nil))
	if err := a.Run(context.Background()); err != nil {
		t.Fatal(err)
	}
	if a.Buffer().Len() != n && a.Buffer().Len() != a.Buffer().Cap() {
		t.Fatalf("buffered %d of %d", a.Buffer().Len(), n)
	}
}

func TestAgentCancelStopsCleanly(t *testing.T) {
	p, _, _ := writeTrace(t, sim.Clean)
	cfg := types.DefaultAgentConfig()
	a, _ := New(cfg)
	a.AddCollector(collector.NewReplayCollector(p, nil))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := a.Run(ctx); err != nil {
		t.Fatalf("cancelled run should be clean, got %v", err)
	}
}

func TestNoCollectorsBlocksUntilCancel(t *testing.T) {
	a, _ := New(types.DefaultAgentConfig())
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := a.Run(ctx); err != nil {
		t.Fatal(err)
	}
}

func TestBadReplayFileFailsFast(t *testing.T) {
	a, _ := New(types.DefaultAgentConfig())
	a.AddCollector(collector.NewReplayCollector("/nonexistent.jsonl", nil))
	if err := a.Run(context.Background()); err == nil {
		t.Fatal("expected error")
	}
}

// Eight agents, one per rank, stream to one coordinator over gRPC: the
// distributed shape of the real system, with replay standing in for GPUs.
func TestEightAgentsOneCoordinator(t *testing.T) {
	p, truth, _ := writeTrace(t, sim.Thermal)
	live := coordinator.NewLive(coordinator.Options{}, 0)
	srv := transport.NewServer("127.0.0.1:0", live)
	if err := srv.Listen(); err != nil {
		t.Fatal(err)
	}
	go srv.Start()
	defer srv.Stop()

	errs := make(chan error, 8)
	for r := uint32(0); r < 8; r++ {
		go func(r uint32) {
			cl, err := transport.NewClient(context.Background(), srv.Addr())
			if err != nil {
				errs <- err
				return
			}
			defer cl.Close()
			a, _ := New(types.DefaultAgentConfig())
			a.AddCollector(collector.NewReplayCollector(p, []uint32{r}))
			a.SetClient(cl)
			errs <- a.Run(context.Background())
		}(r)
	}
	for i := 0; i < 8; i++ {
		if err := <-errs; err != nil {
			t.Fatal(err)
		}
	}
	rep := live.Report()
	if len(rep) != 1 || int(rep[0].Anomaly.StragglerRank) != truth.Straggler {
		t.Fatalf("want rank %d, got %+v", truth.Straggler, rep)
	}
	if rep[0].CausalChain[0].Cause != "thermal_throttle" {
		t.Fatalf("cause %s", rep[0].CausalChain[0].Cause)
	}
}
