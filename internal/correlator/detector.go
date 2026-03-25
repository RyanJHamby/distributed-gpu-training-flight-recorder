package correlator

import "math"

// Stats holds descriptive statistics for a set of observations.
type Stats struct {
	Median float64
	Mean   float64
	StdDev float64
	Min    float64
	Max    float64
	Count  int
}

func ComputeStats(values []float64) Stats {
	// TODO: sort copy, compute median/mean/stddev/min/max
	return Stats{}
}

// IsOutlier returns true if value exceeds median by more than thresholdSigma stddevs.
func IsOutlier(value float64, stats Stats, thresholdSigma float64) bool {
	if stats.StdDev == 0 {
		return false
	}
	deviation := (value - stats.Median) / stats.StdDev
	return deviation > thresholdSigma
}

var _ = math.Sqrt // needed for StdDev in ComputeStats
