package collector

import (
	"context"

	"github.com/ryanhamby/gpu-flight-recorder/internal/types"
)

// DCGMCollector polls GPU metrics via NVIDIA DCGM/NVML.
type DCGMCollector struct {
	config types.AgentConfig
	events chan types.Event
}

func NewDCGMCollector(cfg types.AgentConfig) *DCGMCollector {
	return &DCGMCollector{
		config: cfg,
		events: make(chan types.Event, 1024),
	}
}

func (c *DCGMCollector) Name() string { return "dcgm" }

func (c *DCGMCollector) Start(ctx context.Context) error {
	// TODO: init go-dcgm, create field group (temp, power, util, mem_bw, ecc),
	// poll on ticker at cfg.PollInterval, emit GPUMetricEvents
	return nil
}

func (c *DCGMCollector) Stop() error {
	close(c.events)
	return nil
}

func (c *DCGMCollector) Events() <-chan types.Event { return c.events }
