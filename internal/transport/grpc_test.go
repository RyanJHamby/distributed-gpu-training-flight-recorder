package transport

import (
	"context"
	"testing"
	"time"

	pb "github.com/RyanJHamby/distributed-gpu-training-flight-recorder/api/proto"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/attribution"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/coordinator"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/sim"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
	"reflect"
)

func TestProtoRoundTrip(t *testing.T) {
	b := types.BaseEvent{Timestamp: 5, RankID: 2, NodeID: "n0", GPUUUID: "GPU-x"}
	for _, in := range []types.Event{
		types.GPUMetricEvent{BaseEvent: b, Temperature: 70, ECCErrorsDBE: 1, SMClockMHz: 1100, ThrottleReasons: 0x40, ComputeProcs: 2},
		types.NCCLCollectiveEvent{BaseEvent: b, PGID: "pg0", SeqID: 9, Step: 3, OpType: "AllReduce", DurationNs: 42, Algorithm: "ring"},
		types.PCIeBandwidthEvent{BaseEvent: b, ReadBandwidthMBs: 1.5, WriteBandwidthMBs: 2},
		types.NVLinkEvent{BaseEvent: b, LinkID: 3, ThroughputGB: 7},
		types.ThermalEvent{BaseEvent: b, TemperatureC: 80, ThrottleActive: true},
	} {
		m, err := ToProto(in)
		if err != nil {
			t.Fatal(err)
		}
		out, err := FromProto(m)
		if err != nil || !reflect.DeepEqual(in, out) {
			t.Fatalf("%T: %v %v", in, out, err)
		}
	}
	if _, err := FromProto(&pb.EventMessage{}); err == nil {
		t.Fatal("empty payload must error")
	}
}

func startServer(t *testing.T, h Handler) (*Server, *Client) {
	t.Helper()
	srv := NewServer("127.0.0.1:0", h)
	if err := srv.Listen(); err != nil {
		t.Fatal(err)
	}
	go srv.Start()
	t.Cleanup(srv.Stop)
	cl, err := NewClient(context.Background(), srv.Addr())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { cl.Close() })
	return srv, cl
}

// End to end over a real gRPC connection: stream a synthetic throttled-rank
// trace from one client and read the attribution back through GetReport.
func TestStreamThenReportEndToEnd(t *testing.T) {
	live := coordinator.NewLive(coordinator.Options{}, 0)
	srv, cl := startServer(t, live)
	evs, truth := sim.Generate(sim.Params{Fault: sim.Thermal, Seed: 1, Steps: 300})

	ch := make(chan types.Event, 64)
	go func() {
		for _, e := range evs {
			ch <- e
		}
		close(ch)
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	n, err := cl.StreamEvents(ctx, ch)
	if err != nil || n != uint64(len(evs)) || srv.Received() != n {
		t.Fatalf("acked %d of %d, received %d, err %v", n, len(evs), srv.Received(), err)
	}
	rep, err := cl.GetReport(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Anomalies) != 1 || int(rep.Anomalies[0].StragglerRank) != truth.Straggler ||
		rep.Anomalies[0].CausalChain[0].Cause != string(attribution.CauseThermalThrottle) {
		t.Fatalf("wrong report: %v (truth rank %d)", rep.Anomalies, truth.Straggler)
	}
}

func TestMalformedEventDoesNotKillStream(t *testing.T) {
	srv, cl := startServer(t, nil)
	stream, err := cl.events.StreamEvents(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	good, _ := ToProto(types.ThermalEvent{})
	for _, m := range []*pb.EventMessage{good, {Rank: 1}, good} {
		if err := stream.Send(m); err != nil {
			t.Fatal(err)
		}
	}
	ack, err := stream.CloseAndRecv()
	if err != nil || ack.EventsReceived != 2 || srv.Received() != 2 {
		t.Fatalf("ack %v err %v", ack, err)
	}
}

func TestStreamRetriesUntilServerAppears(t *testing.T) {
	cl, err := NewClient(context.Background(), "127.0.0.1:1") // nothing listening
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Millisecond)
	defer cancel()
	ch := make(chan types.Event)
	if _, err := cl.StreamEvents(ctx, ch); err == nil {
		t.Fatal("expected ctx error with no server")
	}
}
