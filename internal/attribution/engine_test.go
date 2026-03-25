package attribution

import (
	"testing"

	"github.com/ryanhamby/gpu-flight-recorder/internal/correlator"
)

func TestNewEngine(t *testing.T) {
	e := NewEngine(5_000_000_000)
	if e == nil {
		t.Fatal("NewEngine returned nil")
	}
}

func TestAttributeReturnsAnomaly(t *testing.T) {
	e := NewEngine(5_000_000_000)
	anomaly := correlator.Anomaly{
		StragglerRank: 3,
		OpType:        "AllReduce",
	}
	result := e.Attribute(anomaly, nil)
	if result.Anomaly.StragglerRank != 3 {
		t.Errorf("expected straggler rank 3, got %d", result.Anomaly.StragglerRank)
	}
}

// TODO: test with thermal events, ECC errors, multiple causes, unknown cause
