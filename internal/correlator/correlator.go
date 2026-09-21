package correlator

import (
	"math"
	"sort"
	"sync"
	"time"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// Anomaly is a persistent straggler: one rank that repeatedly arrives last
// at collectives, making every peer wait.
//
// The signal is skew-free. In a synchronising collective the last rank to
// arrive sees the shortest duration and the earliest arrivals the longest, so
// lag = max(duration in group) - own duration needs no cross-node clock
// alignment. The "duration" fields below hold lags, not raw op durations.
type Anomaly struct {
	DetectedAt         int64   `json:"detected_at_ns"` // timestamp of the latest flagged collective
	StragglerRank      uint32  `json:"straggler_rank"`
	OpType             string  `json:"op_type"`
	ExpectedDurationNs int64   `json:"expected_lag_ns"` // unused (0); kept for report compatibility
	ActualDurationNs   int64   `json:"actual_lag_ns"`   // median excess lag over peers in flagged blocks
	DeviationSigma     float64 `json:"deviation_sigma"` // median z of the flagged blocks
	Hits               int     `json:"hits"`            // collectives inside flagged blocks
	Groups             int     `json:"groups"`          // complete collectives since the fault onset
	StartNs            int64   `json:"start_ns"`        // first flagged collective
	EndNs              int64   `json:"end_ns"`          // last flagged collective
}

// Config tunes detection.
type Config struct {
	ThresholdSigma float64       // z cutoff on a block-median shift
	Lookback       time.Duration // events older than newest-Lookback are dropped
	MinRanks       int           // a collective needs this many ranks to be scored
	BlockSize      int           // collectives per statistical block
	MinHits        int           // persistence: minimum flagged blocks
	MinHitFraction float64       // persistence: flagged blocks / blocks since first flagged
	JitterFloor    float64       // spread floor as a fraction of median op duration
}

func DefaultConfig(thresholdSigma float64) Config {
	return Config{
		ThresholdSigma: thresholdSigma,
		Lookback:       60 * time.Second,
		MinRanks:       2,
		BlockSize:      30,
		MinHits:        2,
		MinHitFraction: 0.5,
		JitterFloor:    0.05,
	}
}

type groupKey struct {
	pg  string
	seq uint64
}

// Correlator aligns collectives across ranks and detects stragglers.
type Correlator struct {
	mu         sync.RWMutex
	cfg        Config
	rankEvents map[uint32][]types.Event
	groups     map[groupKey]map[uint32]types.NCCLCollectiveEvent
	newestNs   int64
}

func NewCorrelator(thresholdSigma float64) *Correlator {
	return NewCorrelatorWithConfig(DefaultConfig(thresholdSigma))
}

func NewCorrelatorWithConfig(cfg Config) *Correlator {
	return &Correlator{
		cfg:        cfg,
		rankEvents: make(map[uint32][]types.Event),
		groups:     make(map[groupKey]map[uint32]types.NCCLCollectiveEvent),
	}
}

// IngestEvent stores an event under rank, tolerating out-of-order arrival,
// and trims anything older than the lookback window.
func (c *Correlator) IngestEvent(rank uint32, event types.Event) {
	if event == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	evs := c.rankEvents[rank]
	ts := event.TimestampNs()
	i := sort.Search(len(evs), func(i int) bool { return evs[i].TimestampNs() > ts })
	evs = append(evs, nil)
	copy(evs[i+1:], evs[i:])
	evs[i] = event
	c.rankEvents[rank] = evs

	if n, ok := event.(types.NCCLCollectiveEvent); ok {
		k := groupKey{n.PGID, n.SeqID}
		g := c.groups[k]
		if g == nil {
			g = make(map[uint32]types.NCCLCollectiveEvent)
			c.groups[k] = g
		}
		g[rank] = n // a duplicate from the same rank replaces, never double-counts
	}
	if ts > c.newestNs {
		c.newestNs = ts
	}
	c.trimLocked()
}

func (c *Correlator) trimLocked() {
	if c.cfg.Lookback <= 0 {
		return
	}
	cutoff := c.newestNs - c.cfg.Lookback.Nanoseconds()
	for r, evs := range c.rankEvents {
		i := sort.Search(len(evs), func(i int) bool { return evs[i].TimestampNs() >= cutoff })
		if i > 0 {
			c.rankEvents[r] = append([]types.Event(nil), evs[i:]...)
		}
	}
	for k, g := range c.groups {
		var newest int64
		for _, e := range g {
			if e.Timestamp > newest {
				newest = e.Timestamp
			}
		}
		if newest < cutoff {
			delete(c.groups, k)
		}
	}
}

type gInfo struct {
	ts   int64
	seq  uint64
	op   string
	lags map[uint32]float64 // rank -> max(duration in group) - own duration
}

// DetectAnomalies looks for a rank whose collective lag is persistently higher
// than its peers'. Collectives seen within window of the newest event are
// ordered by time and cut into blocks of BlockSize. Per block, each rank's
// median lag is compared with the median of the other ranks' medians, scaled
// by the standard error of a median of BlockSize noisy lags (noise estimated
// robustly, excluding the rank under test so a straggler cannot inflate its
// own baseline). Evidence therefore accumulates over a block instead of
// hinging on any single collective. A rank is reported when it is flagged in
// at least MinHits blocks and in MinHitFraction of the blocks since its first
// flagged one. Event time, not wall clock, drives windows, so replays behave
// like live runs.
func (c *Correlator) DetectAnomalies(window time.Duration) []Anomaly {
	c.mu.RLock()
	defer c.mu.RUnlock()

	cutoff := c.newestNs - window.Nanoseconds()
	var gs []gInfo
	var durs []float64
	for k, g := range c.groups {
		if len(g) < c.cfg.MinRanks {
			continue
		}
		var newest, maxDur int64
		var op string
		for _, e := range g {
			if e.Timestamp > newest {
				newest = e.Timestamp
			}
			if e.DurationNs > maxDur {
				maxDur = e.DurationNs
			}
			op = e.OpType
			durs = append(durs, float64(e.DurationNs))
		}
		if newest < cutoff {
			continue
		}
		lags := make(map[uint32]float64, len(g))
		for r, e := range g {
			lags[r] = float64(maxDur - e.DurationNs)
		}
		gs = append(gs, gInfo{newest, k.seq, op, lags})
	}
	bs := c.cfg.BlockSize
	if bs < 2 || len(gs) < bs {
		return nil
	}
	sort.Slice(gs, func(i, j int) bool {
		if gs[i].ts != gs[j].ts {
			return gs[i].ts < gs[j].ts
		}
		return gs[i].seq < gs[j].seq
	})
	floor := c.cfg.JitterFloor * ComputeStats(durs).Median

	type flag struct {
		block int
		z     float64
		lag   float64
		op    string
	}
	nBlocks := len(gs) / bs // trailing partial block is ignored
	flags := map[uint32][]flag{}
	for b := 0; b < nBlocks; b++ {
		blk := gs[b*bs : (b+1)*bs]
		perRank := map[uint32][]float64{}
		for _, g := range blk {
			for r, l := range g.lags {
				perRank[r] = append(perRank[r], l)
			}
		}
		med := map[uint32]float64{}
		for r, ls := range perRank {
			if len(ls) >= bs/2 {
				med[r] = ComputeStats(ls).Median
			}
		}
		for r, m := range med {
			var others, pooled []float64
			for r2, m2 := range med {
				if r2 != r {
					others = append(others, m2)
					pooled = append(pooled, perRank[r2]...)
				}
			}
			if len(others) == 0 {
				continue
			}
			base := ComputeStats(others).Median
			sigma := math.Max(ComputeStats(pooled).RobustScale(), floor)
			se := 1.2533 * sigma / math.Sqrt(float64(len(perRank[r])))
			if z := (m - base) / se; z > c.cfg.ThresholdSigma {
				flags[r] = append(flags[r], flag{b, z, m - base, blk[len(blk)/2].op})
			}
		}
	}

	var out []Anomaly
	for r, fl := range flags {
		first := fl[0].block
		if len(fl) < c.cfg.MinHits || float64(len(fl))/float64(nBlocks-first) < c.cfg.MinHitFraction {
			continue
		}
		lags := make([]float64, len(fl))
		zs := make([]float64, len(fl))
		ops := map[string]int{}
		for i, f := range fl {
			lags[i], zs[i] = f.lag, f.z
			ops[f.op]++
		}
		last := fl[len(fl)-1].block
		out = append(out, Anomaly{
			DetectedAt:         gs[(last+1)*bs-1].ts,
			StragglerRank:      r,
			OpType:             modeKey(ops),
			ExpectedDurationNs: 0,
			ActualDurationNs:   int64(ComputeStats(lags).Median),
			DeviationSigma:     ComputeStats(zs).Median,
			Hits:               len(fl) * bs,
			Groups:             (nBlocks - first) * bs,
			StartNs:            gs[first*bs].ts,
			EndNs:              gs[(last+1)*bs-1].ts,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].DeviationSigma > out[j].DeviationSigma })
	return out
}

func modeKey(m map[string]int) string {
	best, bn := "", -1
	for k, n := range m {
		if n > bn || (n == bn && k < best) {
			best, bn = k, n
		}
	}
	return best
}

// EventsForRank returns a copy of the rank's events with timestamp >= since.
func (c *Correlator) EventsForRank(rank uint32, since int64) []types.Event {
	c.mu.RLock()
	defer c.mu.RUnlock()
	evs := c.rankEvents[rank]
	i := sort.Search(len(evs), func(i int) bool { return evs[i].TimestampNs() >= since })
	return append([]types.Event(nil), evs[i:]...)
}
