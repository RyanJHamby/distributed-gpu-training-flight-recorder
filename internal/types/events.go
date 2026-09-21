package types

// Event is the common interface for all recorded events.
type Event interface {
	TimestampNs() int64
	Rank() uint32
}

// BaseEvent provides common fields for all event types.
type BaseEvent struct {
	Timestamp int64  `json:"timestamp_ns"`
	RankID    uint32 `json:"rank"`
	NodeID    string `json:"node_id,omitempty"`
	GPUUUID   string `json:"gpu_uuid,omitempty"`
}

func (b BaseEvent) TimestampNs() int64 { return b.Timestamp }
func (b BaseEvent) Rank() uint32       { return b.RankID }

// GPUMetricEvent is a snapshot of GPU hardware metrics from DCGM/NVML.
type GPUMetricEvent struct {
	BaseEvent
	Temperature  float64 `json:"temperature_c"`
	PowerWatts   float64 `json:"power_watts"`
	Utilization  float64 `json:"utilization_pct"`
	MemBandwidth float64 `json:"mem_bandwidth_pct"`
	ECCErrorsSBE uint64  `json:"ecc_errors_sbe"`
	ECCErrorsDBE uint64  `json:"ecc_errors_dbe"`
	SMClockMHz   float64 `json:"sm_clock_mhz,omitempty"`
	// ThrottleReasons is the NVML clocksThrottleReasons bitmask.
	ThrottleReasons uint64 `json:"throttle_reasons,omitempty"`
	// ComputeProcs is the number of compute processes NVML reports on the GPU.
	// 0 means unknown (unsupported, or PIDs hidden by a container), not "none".
	ComputeProcs int `json:"compute_procs,omitempty"`
}

// NCCLCollectiveEvent records timing for a single NCCL collective operation.
type NCCLCollectiveEvent struct {
	BaseEvent
	PGID       string `json:"pg_id"`  // process group; with SeqID identifies one collective across ranks
	SeqID      uint64 `json:"seq_id"` // per-process-group collective sequence number
	Step       int64  `json:"step,omitempty"`
	OpType     string `json:"op_type"` // AllReduce, AllGather, ReduceScatter, etc.
	DataSize   uint64 `json:"data_size"`
	DurationNs int64  `json:"duration_ns"`
	Algorithm  string `json:"algorithm"` // ring, tree
}

// PCIeBandwidthEvent records PCIe bus bandwidth.
type PCIeBandwidthEvent struct {
	BaseEvent
	ReadBandwidthMBs  float64 `json:"read_bw_mbs"`
	WriteBandwidthMBs float64 `json:"write_bw_mbs"`
}

// NVLinkEvent records per-link NVLink throughput.
type NVLinkEvent struct {
	BaseEvent
	LinkID       uint32  `json:"link_id"`
	ThroughputGB float64 `json:"throughput_gbs"`
}

// ThermalEvent records thermal state and throttling.
type ThermalEvent struct {
	BaseEvent
	TemperatureC   float64 `json:"temperature_c"`
	ThrottleActive bool    `json:"throttle_active"`
}
