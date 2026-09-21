package agent

import (
	"context"
	"fmt"
	"log"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/buffer"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/collector"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// Agent is the per-node daemon that manages collectors, ring buffers,
// and the gRPC stream to the coordinator.
type Agent struct {
	config     types.AgentConfig
	collectors []collector.Collector
	buffers    map[string]*buffer.RingBuffer
}

func New(cfg types.AgentConfig) (*Agent, error) {
	a := &Agent{
		config:  cfg,
		buffers: make(map[string]*buffer.RingBuffer),
	}
	// TODO: create collectors (dcgm, nccl, system), ring buffer per collector,
	// gRPC client to coordinator
	return a, nil
}

// Run starts all collectors and forwards events to the coordinator. Blocks until ctx is done.
func (a *Agent) Run(ctx context.Context) error {
	log.Printf("agent starting on node %s", a.config.NodeID)

	// TODO: start collectors, fan-in events from all collector channels,
	// push to ring buffer + forward via gRPC stream

	<-ctx.Done()
	return a.shutdown()
}

func (a *Agent) shutdown() error {
	log.Println("agent shutting down")
	var firstErr error
	for _, c := range a.collectors {
		if err := c.Stop(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("stopping collector %s: %w", c.Name(), err)
		}
	}
	return firstErr
}
