package agent

import (
	"testing"

	"github.com/ryanhamby/gpu-flight-recorder/internal/types"
)

func TestNewAgent(t *testing.T) {
	cfg := types.DefaultAgentConfig()
	cfg.NodeID = "test-node"
	a, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}
	if a == nil {
		t.Fatal("New() returned nil agent")
	}
}

// TODO: test run + shutdown lifecycle, event forwarding with mock collectors
