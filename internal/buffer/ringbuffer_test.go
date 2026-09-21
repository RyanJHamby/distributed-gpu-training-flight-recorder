package buffer

import (
	"testing"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

func makeEvent(ts int64, rank uint32) types.Event {
	return &types.GPUMetricEvent{
		BaseEvent: types.BaseEvent{Timestamp: ts, RankID: rank},
	}
}

func TestNewRingBuffer(t *testing.T) {
	rb := NewRingBuffer(1024)
	if rb == nil {
		t.Fatal("NewRingBuffer returned nil")
	}
	if rb.Len() != 0 {
		t.Errorf("expected empty buffer, got Len() = %d", rb.Len())
	}
}

func TestPushAndLen(t *testing.T) {
	rb := NewRingBuffer(1024)
	for i := 0; i < 10; i++ {
		rb.Push(makeEvent(int64(i), 0))
	}
	// TODO: assert Len() == 10 once Push is implemented
}

// TODO: tests for overwrite/wrap, Range ordering, concurrent push+range (-race)
