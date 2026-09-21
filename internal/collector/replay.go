package collector

import (
	"context"
	"os"
	"sync"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/trace"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// ReplayCollector emits events from a JSONL trace. With a non-empty Ranks set
// it emits only those ranks, so N agent processes can each play one rank of a
// recorded run. The channel is closed when the trace is exhausted.
type ReplayCollector struct {
	path   string
	ranks  map[uint32]bool
	events chan types.Event

	stop     context.CancelFunc
	stopOnce sync.Once
	done     chan struct{}
}

func NewReplayCollector(path string, ranks []uint32) *ReplayCollector {
	rs := map[uint32]bool{}
	for _, r := range ranks {
		rs[r] = true
	}
	return &ReplayCollector{path: path, ranks: rs, events: make(chan types.Event, 1024), done: make(chan struct{})}
}

func (c *ReplayCollector) Name() string { return "replay" }

// Start loads the trace (failing fast on a bad file) and emits in the background.
func (c *ReplayCollector) Start(ctx context.Context) error {
	f, err := os.Open(c.path)
	if err != nil {
		return err
	}
	evs, err := trace.Read(f)
	f.Close()
	if err != nil {
		return err
	}
	ctx, c.stop = context.WithCancel(ctx)
	go func() {
		defer close(c.done)
		defer close(c.events)
		for _, e := range evs {
			if len(c.ranks) > 0 && !c.ranks[e.Rank()] {
				continue
			}
			select {
			case c.events <- e:
			case <-ctx.Done():
				return
			}
		}
	}()
	return nil
}

// Stop is idempotent and safe before Start.
func (c *ReplayCollector) Stop() error {
	c.stopOnce.Do(func() {
		if c.stop != nil {
			c.stop()
			<-c.done
		}
	})
	return nil
}

func (c *ReplayCollector) Events() <-chan types.Event { return c.events }
