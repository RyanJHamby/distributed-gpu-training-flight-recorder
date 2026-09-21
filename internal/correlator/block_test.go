package correlator

import (
	"math/rand"
	"testing"
	"time"
)

// gen feeds `n` collectives over `ranks` ranks. Ranks in slow arrive `extra`
// ns late from collective `from` to `to` (peers wait); skew offsets each
// rank's clock, to show detection needs no clock alignment.
func gen(c *Correlator, seed int64, ranks, n, from, to int, slow map[int]bool, extra int64, skew int64) {
	rng := rand.New(rand.NewSource(seed))
	for s := 0; s < n; s++ {
		for r := 0; r < ranks; r++ {
			d := int64(10e6 * (1 + rng.NormFloat64()*0.03))
			if s >= from && s < to && !slow[r] {
				d += extra
			}
			c.IngestEvent(uint32(r), coll(uint32(r), uint64(s), int64(s)*20e6+int64(r)*skew, d))
		}
	}
}

func TestFaultStartingMidRunIsDetected(t *testing.T) {
	c := NewCorrelator(4)
	gen(c, 1, 4, 300, 150, 300, map[int]bool{3: true}, 3e6, 0)
	a := c.DetectAnomalies(time.Hour)
	if len(a) != 1 || a[0].StragglerRank != 3 {
		t.Fatalf("got %+v", a)
	}
	// Onset should be located near collective 150 (within two blocks).
	if lo, hi := int64(150-2*30)*20e6, int64(150+2*30)*20e6; a[0].StartNs < lo || a[0].StartNs > hi {
		t.Fatalf("onset %d outside [%d,%d]", a[0].StartNs, lo, hi)
	}
}

func TestTooShortTraceReturnsNothing(t *testing.T) {
	c := NewCorrelator(4)
	gen(c, 2, 4, 25, 0, 25, map[int]bool{1: true}, 3e6, 0) // < one block
	if a := c.DetectAnomalies(time.Hour); a != nil {
		t.Fatalf("got %+v", a)
	}
}

func TestBriefBurstNotReported(t *testing.T) {
	c := NewCorrelator(4)
	gen(c, 3, 4, 300, 100, 120, map[int]bool{2: true}, 3e6, 0) // 20 collectives: under one block
	if a := c.DetectAnomalies(time.Hour); len(a) != 0 {
		t.Fatalf("brief burst reported: %+v", a)
	}
}

func TestTwoSimultaneousStragglers(t *testing.T) {
	c := NewCorrelator(4)
	gen(c, 4, 6, 300, 100, 300, map[int]bool{1: true, 4: true}, 3e6, 0)
	got := map[uint32]bool{}
	for _, a := range c.DetectAnomalies(time.Hour) {
		got[a.StragglerRank] = true
	}
	if !got[1] || !got[4] || len(got) != 2 {
		t.Fatalf("want ranks 1 and 4, got %v", got)
	}
}

func TestPerRankClockSkewDoesNotBreakDetection(t *testing.T) {
	c := NewCorrelator(4)
	// Each rank's clock is 40ms apart (2 collectives' worth): timestamps disagree,
	// but (pg,seq) grouping and duration-based lag do not care.
	gen(c, 5, 4, 300, 100, 300, map[int]bool{2: true}, 3e6, 40e6)
	a := c.DetectAnomalies(time.Hour)
	if len(a) != 1 || a[0].StragglerRank != 2 {
		t.Fatalf("got %+v", a)
	}
}

func TestWindowLimitsAnalysis(t *testing.T) {
	c := NewCorrelator(4)
	gen(c, 6, 4, 300, 0, 100, map[int]bool{1: true}, 3e6, 0) // fault only in the first third
	// Newest event is at ~6s; a 2s window sees only the healthy tail.
	if a := c.DetectAnomalies(2 * time.Second); len(a) != 0 {
		t.Fatalf("stale fault reported: %+v", a)
	}
	if a := c.DetectAnomalies(time.Hour); len(a) != 1 {
		t.Fatalf("full window should see it: %+v", a)
	}
}
