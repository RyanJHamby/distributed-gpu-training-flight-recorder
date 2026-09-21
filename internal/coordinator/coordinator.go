// Package coordinator wires the correlator to the attribution engine.
package coordinator

import (
	"sort"
	"sync"
	"time"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/attribution"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/correlator"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// Report is the JSON-serialisable output of an analysis.
type Report struct {
	Events       int                       `json:"events"`
	Ranks        []uint32                  `json:"ranks"`
	Attributions []attribution.Attribution `json:"attributions"`
}

// Options configure an analysis. Zero values pick defaults.
type Options struct {
	ThresholdSigma    float64       // default 4
	AttributionWindow time.Duration // default 5s
}

// Analyze runs detection and attribution over a complete event set. Lookback
// trimming is disabled: replays are analysed whole.
func Analyze(events []types.Event, o Options) Report {
	if o.ThresholdSigma == 0 {
		o.ThresholdSigma = 4
	}
	if o.AttributionWindow == 0 {
		o.AttributionWindow = 5 * time.Second
	}
	cfg := correlator.DefaultConfig(o.ThresholdSigma)
	cfg.Lookback = 0
	c := correlator.NewCorrelatorWithConfig(cfg)
	seen := map[uint32]bool{}
	for _, e := range events {
		c.IngestEvent(e.Rank(), e)
		seen[e.Rank()] = true
	}

	rep := Report{Events: len(events), Attributions: []attribution.Attribution{}}
	for r := range seen {
		rep.Ranks = append(rep.Ranks, r)
	}
	sort.Slice(rep.Ranks, func(i, j int) bool { return rep.Ranks[i] < rep.Ranks[j] })

	eng := attribution.NewEngine(o.AttributionWindow.Nanoseconds())
	// Window spanning the whole trace: replay time is event time.
	for _, a := range c.DetectAnomalies(time.Duration(1<<62 - 1)) {
		rep.Attributions = append(rep.Attributions, eng.Attribute(a, c.EventsForRank(a.StragglerRank, 0)))
	}
	return rep
}

// Live accumulates streamed events and analyses them on demand. It satisfies
// transport.Handler. Memory is bounded by MaxEvents (oldest dropped).
type Live struct {
	mu        sync.Mutex
	opts      Options
	maxEvents int
	events    []types.Event
}

func NewLive(o Options, maxEvents int) *Live {
	if maxEvents <= 0 {
		maxEvents = 2_000_000
	}
	return &Live{opts: o, maxEvents: maxEvents}
}

func (l *Live) Ingest(e types.Event) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, e)
	if len(l.events) > l.maxEvents {
		l.events = append([]types.Event(nil), l.events[len(l.events)-l.maxEvents:]...)
	}
}

func (l *Live) Report() []attribution.Attribution {
	l.mu.Lock()
	snap := append([]types.Event(nil), l.events...)
	l.mu.Unlock()
	return Analyze(snap, l.opts).Attributions
}
