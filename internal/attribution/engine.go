package attribution

import (
	"github.com/ryanhamby/gpu-flight-recorder/internal/correlator"
	"github.com/ryanhamby/gpu-flight-recorder/internal/types"
)

type CauseType string

const (
	CauseThermalThrottle CauseType = "thermal_throttle"
	CauseECCErrors       CauseType = "ecc_errors"
	CausePCIeBandwidth   CauseType = "pcie_bandwidth_drop"
	CauseNVLinkDegraded  CauseType = "nvlink_degraded"
	CauseMemoryPressure  CauseType = "memory_pressure"
	CauseUnknown         CauseType = "unknown"
)

type CausalLink struct {
	Cause      CauseType   `json:"cause"`
	Evidence   types.Event `json:"-"`
	Confidence float64     `json:"confidence"`
	Detail     string      `json:"detail"`
}

type Attribution struct {
	Anomaly     correlator.Anomaly `json:"anomaly"`
	CausalChain []CausalLink       `json:"causal_chain"`
	Summary     string             `json:"summary"`
}

type Engine struct {
	attributionWindowNs int64
}

func NewEngine(attributionWindowNs int64) *Engine {
	return &Engine{attributionWindowNs: attributionWindowNs}
}

// Attribute looks back in the anomalous rank's event buffer to find
// hardware-level causes (thermal throttle, ECC errors, bandwidth drops).
func (e *Engine) Attribute(anomaly correlator.Anomaly, rankEvents []types.Event) Attribution {
	// TODO: filter events within attribution window, check for thermal spikes,
	// ECC error increases, PCIe/NVLink bandwidth drops, build causal chain
	// sorted by confidence, generate human-readable summary
	return Attribution{
		Anomaly: anomaly,
		Summary: "attribution not yet implemented",
	}
}
