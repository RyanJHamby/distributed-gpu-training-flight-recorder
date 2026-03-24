package types

import "time"

// AgentConfig holds configuration for the per-node agent daemon.
type AgentConfig struct {
	PollInterval       time.Duration `json:"poll_interval" yaml:"poll_interval"`
	BufferSize         int           `json:"buffer_size" yaml:"buffer_size"`
	CoordinatorAddress string        `json:"coordinator_address" yaml:"coordinator_address"`
	NodeID             string        `json:"node_id" yaml:"node_id"`
	GPUIndices         []int         `json:"gpu_indices" yaml:"gpu_indices"`
}

func DefaultAgentConfig() AgentConfig {
	return AgentConfig{
		PollInterval: 100 * time.Millisecond,
		BufferSize:   1 << 20, // 1M events (~64MB at 64 bytes/event)
	}
}

// CoordinatorConfig holds configuration for the coordinator.
type CoordinatorConfig struct {
	ListenAddress         string        `json:"listen_address" yaml:"listen_address"`
	AnomalyThresholdSigma float64       `json:"anomaly_threshold_sigma" yaml:"anomaly_threshold_sigma"`
	LookbackWindow        time.Duration `json:"lookback_window" yaml:"lookback_window"`
	AttributionWindow     time.Duration `json:"attribution_window" yaml:"attribution_window"`
}

func DefaultCoordinatorConfig() CoordinatorConfig {
	return CoordinatorConfig{
		ListenAddress:         ":50051",
		AnomalyThresholdSigma: 2.0,
		LookbackWindow:        30 * time.Second,
		AttributionWindow:     5 * time.Second,
	}
}
