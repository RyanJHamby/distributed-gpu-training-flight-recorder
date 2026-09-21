package correlator

import (
	"math"
	"sort"
)

// Stats holds descriptive statistics for a set of observations.
// MAD is the raw median absolute deviation; use RobustScale for a
// sigma-comparable spread.
type Stats struct {
	Median float64
	Mean   float64
	StdDev float64
	MAD    float64
	Min    float64
	Max    float64
	Count  int
}

// madToSigma makes MAD consistent with the standard deviation for normal data.
const madToSigma = 1.4826

// ComputeStats returns zero Stats for empty input. StdDev is the population
// standard deviation.
func ComputeStats(values []float64) Stats {
	n := len(values)
	if n == 0 {
		return Stats{}
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	st := Stats{Count: n, Min: sorted[0], Max: sorted[n-1], Median: median(sorted)}
	var sum float64
	for _, v := range sorted {
		sum += v
	}
	st.Mean = sum / float64(n)
	var sq float64
	for _, v := range sorted {
		sq += (v - st.Mean) * (v - st.Mean)
	}
	st.StdDev = math.Sqrt(sq / float64(n))
	dev := make([]float64, n)
	for i, v := range sorted {
		dev[i] = math.Abs(v - st.Median)
	}
	sort.Float64s(dev)
	st.MAD = median(dev)
	return st
}

// RobustScale is MAD scaled to be comparable with a standard deviation.
func (s Stats) RobustScale() float64 { return s.MAD * madToSigma }

// median expects sorted input.
func median(sorted []float64) float64 {
	n := len(sorted)
	if n%2 == 1 {
		return sorted[n/2]
	}
	return (sorted[n/2-1] + sorted[n/2]) / 2
}

// RobustZ scores value against the median using MAD, with scaleFloor to keep
// a near-zero MAD (very regular data) from turning noise into huge scores.
func RobustZ(value float64, s Stats, scaleFloor float64) float64 {
	scale := math.Max(s.RobustScale(), scaleFloor)
	if scale == 0 {
		return 0
	}
	return (value - s.Median) / scale
}

// IsOutlier returns true if value exceeds the median by more than
// thresholdSigma standard deviations (one-sided, slow outliers only).
func IsOutlier(value float64, stats Stats, thresholdSigma float64) bool {
	if stats.StdDev == 0 {
		return false
	}
	return (value-stats.Median)/stats.StdDev > thresholdSigma
}
