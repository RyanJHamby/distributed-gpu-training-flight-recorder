package collector

import (
	"context"

	"github.com/ryanhamby/gpu-flight-recorder/internal/types"
)

// SystemCollector collects PCIe bandwidth, NVLink throughput, and thermal state.
type SystemCollector struct {
	config types.AgentConfig
	events chan types.Event
}

func NewSystemCollector(cfg types.AgentConfig) *SystemCollector {
	return &SystemCollector{
		config: cfg,
		events: make(chan types.Event, 1024),
	}
}

func (c *SystemCollector) Name() string { return "system" }

func (c *SystemCollector) Start(ctx context.Context) error {
	// TODO: poll PCIe bw from sysfs or NVML, NVLink counters per link,
	// thermal + throttle reasons. Emit PCIeBandwidthEvent, NVLinkEvent,
	// ThermalEvent on cfg.PollInterval ticker.
	return nil
}

func (c *SystemCollector) Stop() error {
	close(c.events)
	return nil
}

func (c *SystemCollector) Events() <-chan types.Event { return c.events }
