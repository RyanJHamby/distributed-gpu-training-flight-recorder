package collector

import (
	"context"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// NCCLCollector captures NCCL collective timing by parsing NCCL_DEBUG=TRACE output.
type NCCLCollector struct {
	config types.AgentConfig
	events chan types.Event
}

func NewNCCLCollector(cfg types.AgentConfig) *NCCLCollector {
	return &NCCLCollector{
		config: cfg,
		events: make(chan types.Event, 1024),
	}
}

func (c *NCCLCollector) Name() string { return "nccl" }

func (c *NCCLCollector) Start(ctx context.Context) error {
	// TODO: read NCCL_DEBUG=TRACE output from named pipe or log file,
	// parse collective ops (AllReduce etc), extract timing + rank + algo,
	// emit NCCLCollectiveEvents. Could also do LD_PRELOAD interposition
	// later for more accurate timing.
	return nil
}

func (c *NCCLCollector) Stop() error {
	close(c.events)
	return nil
}

func (c *NCCLCollector) Events() <-chan types.Event { return c.events }
