package correlator

import (
	"testing"
)

func TestIsOutlierBasic(t *testing.T) {
	stats := Stats{Median: 100, StdDev: 10, Count: 5}

	if IsOutlier(105, stats, 2.0) {
		t.Error("value within 1σ should not be an outlier")
	}
	if !IsOutlier(125, stats, 2.0) {
		t.Error("value at 2.5σ should be an outlier")
	}
}

func TestIsOutlierZeroStdDev(t *testing.T) {
	stats := Stats{Median: 100, StdDev: 0, Count: 5}
	if IsOutlier(100, stats, 2.0) {
		t.Error("zero stddev should never produce outliers")
	}
}

// TODO: TestComputeStats, TestDetectAnomalies with mock NCCLCollectiveEvents
