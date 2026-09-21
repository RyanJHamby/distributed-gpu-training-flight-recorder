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
	ExpectedDurationNs int64   `json:"expected_lag_ns"` // median lag of non-last ranks in window
	ActualDurationNs   int64   `json:"actual_lag_ns"`   // median lag of the straggler's flagged collectives
	DeviationSigma     float64 `json:"deviation_sigma"` // median robust z-score of flagged collectives
	Hits               int     `json:"hits"`            // collectives where this rank was the outlier
	Groups             int     `json:"groups"`          // complete collectives examined in window
	StartNs            int64   `json:"start_ns"`        // first flagged collective
	EndNs              int64   `json:"end_ns"`          // last flagged collective
}

// Config tunes detection.
type Config struct {
	ThresholdSigma float64       // robust-z cutoff per collective
	Lookback       time.Duration // events older than newest-Lookback are dropped
	MinRanks       int           // a collective needs this many ranks to be scored
	MinHits        int           // persistence: minimum flagged collectives
	MinHitFraction float64       // persistence: flagged / examined
	JitterFloor    float64       // spread floor as a fraction of median op duration
}

func DefaultConfig(thresholdSigma float64) Config {
	return Config{
		ThresholdSigma: thresholdSigma,
		Lookback:       60 * time.Second,
		MinRanks:       2,
		MinHits:        3,
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

type scored struct {
	rank uint32
	ts   int64
	op   string
	lag  float64
	z    float64
}

// DetectAnomalies scores collectives seen within window of the newest event
// (event time, not wall clock, so replays behave like live runs).
func (c *Correlator) DetectAnomalies(window time.Duration) []Anomaly {
	c.mu.RLock()
	defer c.mu.RUnlock()

	cutoff := c.newestNs - window.Nanoseconds()
	type cand struct {
		rank uint32
		ts   int64
		op   string
		lag  float64
	}
	var lags, durs []float64
	var cands []cand
	groups := 0
	for _, g := range c.groups {
		if len(g) < c.cfg.MinRanks {
			continue
		}
		var newest, maxDur int64
		for _, e := range g {
			if e.Timestamp > newest {
				newest = e.Timestamp
			}
			if e.DurationNs > maxDur {
				maxDur = e.DurationNs
			}
		}
		if newest < cutoff {
			continue
		}
		groups++
		var last uint32
		minDur := int64(math.MaxInt64)
		tied := false
		var op string
		for r, e := range g {
			durs = append(durs, float64(e.DurationNs))
			switch {
			case e.DurationNs < minDur:
				minDur, last, tied, op = e.DurationNs, r, false, e.OpType
			case e.DurationNs == minDur:
				tied = true
			}
		}
		// Baseline lags exclude each group's last arriver, otherwise a
		// persistent straggler contaminates its own baseline (fatal at 2 ranks).
		for r, e := range g {
			if r != last {
				lags = append(lags, float64(maxDur-e.DurationNs))
			}
		}
		if !tied {
			cands = append(cands, cand{last, newest, op, float64(maxDur - minDur)})
		}
	}
	if groups == 0 {
		return nil
	}

	lagStats := ComputeStats(lags)
	floor := c.cfg.JitterFloor * ComputeStats(durs).Median

	byRank := map[uint32][]scored{}
	for _, cd := range cands {
		z := RobustZ(cd.lag, lagStats, floor)
		if z > c.cfg.ThresholdSigma {
			byRank[cd.rank] = append(byRank[cd.rank], scored{cd.rank, cd.ts, cd.op, cd.lag, z})
		}
	}

	var out []Anomaly
	for r, hits := range byRank {
		if len(hits) < c.cfg.MinHits || float64(len(hits))/float64(groups) < c.cfg.MinHitFraction {
			continue
		}
		hl := make([]float64, len(hits))
		hz := make([]float64, len(hits))
		ops := map[string]int{}
		start, end := hits[0].ts, hits[0].ts
		for i, h := range hits {
			hl[i], hz[i] = h.lag, h.z
			ops[h.op]++
			if h.ts < start {
				start = h.ts
			}
			if h.ts > end {
				end = h.ts
			}
		}
		out = append(out, Anomaly{
			DetectedAt:         end,
			StragglerRank:      r,
			OpType:             modeKey(ops),
			ExpectedDurationNs: int64(lagStats.Median),
			ActualDurationNs:   int64(ComputeStats(hl).Median),
			DeviationSigma:     ComputeStats(hz).Median,
			Hits:               len(hits),
			Groups:             groups,
			StartNs:            start,
			EndNs:              end,
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
