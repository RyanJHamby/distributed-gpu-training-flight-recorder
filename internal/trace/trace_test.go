package trace

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

func TestRoundTripAllTypes(t *testing.T) {
	b := types.BaseEvent{Timestamp: 5, RankID: 2, NodeID: "n0", GPUUUID: "GPU-x"}
	in := []types.Event{
		types.GPUMetricEvent{BaseEvent: b, Temperature: 70, ThrottleReasons: 0x20, ECCErrorsDBE: 1},
		types.NCCLCollectiveEvent{BaseEvent: b, PGID: "pg0", SeqID: 9, OpType: "AllReduce", DurationNs: 42},
		types.PCIeBandwidthEvent{BaseEvent: b, ReadBandwidthMBs: 1.5},
		types.NVLinkEvent{BaseEvent: b, LinkID: 3, ThroughputGB: 7},
		types.ThermalEvent{BaseEvent: b, ThrottleActive: true},
	}
	var buf bytes.Buffer
	if err := Write(&buf, in); err != nil {
		t.Fatal(err)
	}
	out, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Fatalf("round trip differs:\n%v\n%v", in, out)
	}
}

func TestReadErrorsCarryLineNumbers(t *testing.T) {
	_, err := Read(strings.NewReader("\n{\"type\":\"nope\",\"event\":{}}\n"))
	if err == nil || !strings.Contains(err.Error(), "line 2") {
		t.Fatalf("got %v", err)
	}
	if _, err := Read(strings.NewReader("not json\n")); err == nil {
		t.Fatal("expected error")
	}
}
