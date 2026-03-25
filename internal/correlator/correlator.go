package correlator

import (
	"sync"
	"time"

	"github.com/ryanhamby/gpu-flight-recorder/internal/types"
)

// Anomaly represents a detected straggler in a collective operation.
type Anomaly struct {
	DetectedAt         int64   `json:"detected_at_ns"`
	StragglerRank      uint32  `json:"straggler_rank"`
	OpType             string  `json:"op_type"`
	ExpectedDurationNs int64   `json:"expected_duration_ns"`
	ActualDurationNs   int64   `json:"actual_duration_ns"`
	DeviationSigma     float64 `json:"deviation_sigma"`
}

// Correlator aligns event timelines across ranks and detects anomalies.
type Correlator struct {
	mu             sync.RWMutex
	rankEvents     map[uint32][]types.Event
	thresholdSigma float64
}

func NewCorrelator(thresholdSigma float64) *Correlator {
	return &Correlator{
		rankEvents:     make(map[uint32][]types.Event),
		thresholdSigma: thresholdSigma,
	}
}

func (c *Correlator) IngestEvent(rank uint32, event types.Event) {
	// TODO: append to rankEvents[rank], trim events older than lookback window
}

// DetectAnomalies finds stragglers by comparing collective durations across ranks.
func (c *Correlator) DetectAnomalies(window time.Duration) []Anomaly {
	// TODO: group NCCLCollectiveEvents by timestamp proximity (~1ms epsilon),
	// compute median + stddev per collective, flag ranks > thresholdSigma
	return nil
}

// EventsForRank returns events for a rank within the given time window.
func (c *Correlator) EventsForRank(rank uint32, since int64) []types.Event {
	// TODO: binary search + copy
	return nil
}
