package correlator

import (
	"math"
	"testing"
)

func TestComputeStats(t *testing.T) {
	s := ComputeStats([]float64{1, 2, 3, 4, 100})
	if s.Median != 3 || s.Min != 1 || s.Max != 100 || s.Count != 5 || s.Mean != 22 {
		t.Fatalf("%+v", s)
	}
	if s.MAD != 1 { // |x-3| = 2,1,0,1,97 -> median 1
		t.Fatalf("MAD=%v", s.MAD)
	}
	if e := ComputeStats([]float64{1, 3}); e.Median != 2 {
		t.Fatalf("even median %v", e.Median)
	}
	if (ComputeStats(nil) != Stats{}) {
		t.Fatal("empty should be zero")
	}
}

func TestStdDev(t *testing.T) {
	if got := ComputeStats([]float64{2, 4, 4, 4, 5, 5, 7, 9}).StdDev; math.Abs(got-2) > 1e-9 {
		t.Fatalf("stddev=%v want 2", got)
	}
}

func TestRobustZResistsOutlierContamination(t *testing.T) {
	s := ComputeStats([]float64{10, 10, 11, 9, 10, 10, 1000})
	if z := RobustZ(1000, s, 0.1); z < 100 {
		t.Fatalf("outlier z=%v, MAD should not be inflated by it", z)
	}
	if RobustZ(5, Stats{}, 0) != 0 {
		t.Fatal("zero scale must yield 0, not Inf")
	}
}

func TestIsOutlier(t *testing.T) {
	s := ComputeStats([]float64{10, 11, 9, 10, 10})
	if !IsOutlier(100, s, 3) || IsOutlier(10, s, 3) || IsOutlier(1, s, 3) {
		t.Fatal("one-sided outlier check wrong")
	}
	if IsOutlier(5, Stats{}, 3) {
		t.Fatal("zero stddev must not flag")
	}
}
