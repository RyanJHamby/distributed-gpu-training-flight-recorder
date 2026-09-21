package correlator

import (
	"math/rand"
	"testing"
	"time"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

func coll(rank uint32, seq uint64, ts, dur int64) types.NCCLCollectiveEvent {
	return types.NCCLCollectiveEvent{
		BaseEvent: types.BaseEvent{Timestamp: ts, RankID: rank},
		PGID:      "pg0", SeqID: seq, OpType: "AllReduce", DurationNs: dur,
	}
}

// feed simulates n collectives over `ranks` ranks: base 10ms op with +-3%
// jitter; if slow>=0 that rank arrives `extra` late, so peers wait `extra` longer.
func feed(c *Correlator, rng *rand.Rand, ranks, n, slow int, extra int64, startSeq uint64) {
	const base = int64(10e6)
	for s := 0; s < n; s++ {
		ts := int64(s) * 20e6
		for r := 0; r < ranks; r++ {
			jit := int64(rng.NormFloat64() * 0.03 * float64(base))
			d := base + jit
			if slow >= 0 && r != slow {
				d += extra
			}
			c.IngestEvent(uint32(r), coll(uint32(r), startSeq+uint64(s), ts, d))
		}
	}
}

func TestHealthyRunHasNoAnomalies(t *testing.T) {
	for seed := int64(0); seed < 20; seed++ {
		c := NewCorrelator(4)
		feed(c, rand.New(rand.NewSource(seed)), 4, 100, -1, 0, 0)
		if a := c.DetectAnomalies(time.Hour); len(a) != 0 {
			t.Fatalf("seed %d: false positive %+v", seed, a)
		}
	}
}

func TestPersistentStragglerIdentified(t *testing.T) {
	c := NewCorrelator(4)
	feed(c, rand.New(rand.NewSource(1)), 4, 100, 2, 3e6, 0)
	a := c.DetectAnomalies(time.Hour)
	if len(a) != 1 || a[0].StragglerRank != 2 {
		t.Fatalf("want rank 2 only, got %+v", a)
	}
	if a[0].LagNs < 2.5e6 || a[0].LagNs > 3.5e6 {
		t.Fatalf("lag %d not ~3ms", a[0].LagNs)
	}
}

func TestTwoRankStraggler(t *testing.T) {
	c := NewCorrelator(4)
	feed(c, rand.New(rand.NewSource(2)), 2, 100, 1, 3e6, 0)
	a := c.DetectAnomalies(time.Hour)
	if len(a) != 1 || a[0].StragglerRank != 1 {
		t.Fatalf("got %+v", a)
	}
}

func TestSingleBlipNotFlagged(t *testing.T) {
	c := NewCorrelator(4)
	rng := rand.New(rand.NewSource(3))
	feed(c, rng, 4, 100, -1, 0, 0)
	c.IngestEvent(1, coll(1, 500, 1e9, 10e6))
	c.IngestEvent(0, coll(0, 500, 1e9, 20e6))
	c.IngestEvent(2, coll(2, 500, 1e9, 20e6))
	c.IngestEvent(3, coll(3, 500, 1e9, 20e6))
	if a := c.DetectAnomalies(time.Hour); len(a) != 0 {
		t.Fatalf("one-off blip flagged: %+v", a)
	}
}

func TestIncompleteAndDuplicateEvents(t *testing.T) {
	c := NewCorrelatorWithConfig(func() Config { x := DefaultConfig(4); x.MinRanks = 3; return x }())
	c.IngestEvent(0, coll(0, 1, 1, 10e6)) // only 1 of 3 ranks: never scored
	c.IngestEvent(0, coll(0, 1, 1, 10e6)) // duplicate
	if a := c.DetectAnomalies(time.Hour); a != nil {
		t.Fatalf("got %+v", a)
	}
	if got := len(c.groups[groupKey{"pg0", 1}]); got != 1 {
		t.Fatalf("duplicate double-counted: %d", got)
	}
}

func TestOutOfOrderIngestKeepsSorted(t *testing.T) {
	c := NewCorrelator(4)
	for _, ts := range []int64{50, 10, 30, 20, 40} {
		c.IngestEvent(0, types.ThermalEvent{BaseEvent: types.BaseEvent{Timestamp: ts}})
	}
	evs := c.EventsForRank(0, 0)
	for i := 1; i < len(evs); i++ {
		if evs[i-1].TimestampNs() > evs[i].TimestampNs() {
			t.Fatal("not sorted")
		}
	}
	if got := c.EventsForRank(0, 30); len(got) != 3 || got[0].TimestampNs() != 30 {
		t.Fatalf("since filter wrong: %v", got)
	}
}

func TestLookbackTrim(t *testing.T) {
	cfg := DefaultConfig(4)
	cfg.Lookback = 100 * time.Nanosecond
	c := NewCorrelatorWithConfig(cfg)
	for ts := int64(0); ts <= 1000; ts += 10 {
		c.IngestEvent(0, types.ThermalEvent{BaseEvent: types.BaseEvent{Timestamp: ts}})
	}
	if n := len(c.EventsForRank(0, 0)); n > 11 {
		t.Fatalf("retained %d events, want <= 11", n)
	}
}

func FuzzIngestDetect(f *testing.F) {
	f.Add(uint32(1), uint64(1), int64(5), int64(9))
	f.Fuzz(func(t *testing.T, rank uint32, seq uint64, ts, dur int64) {
		c := NewCorrelator(4)
		for i := uint32(0); i < 3; i++ {
			c.IngestEvent(rank+i, coll(rank+i, seq, ts, dur+int64(i)))
		}
		_ = c.DetectAnomalies(time.Second)
	})
}
