package agent

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/buffer"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/collector"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/transport"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// Agent is the per-node daemon: it fans in every collector, keeps the recent
// history in a ring buffer, and forwards events to the coordinator.
type Agent struct {
	config     types.AgentConfig
	collectors []collector.Collector
	buf        *buffer.RingBuffer
	client     *transport.Client // nil: record locally only
}

func New(cfg types.AgentConfig) (*Agent, error) {
	return &Agent{config: cfg, buf: buffer.NewRingBuffer(cfg.BufferSize)}, nil
}

// AddCollector registers a source. Call before Run.
func (a *Agent) AddCollector(c collector.Collector) { a.collectors = append(a.collectors, c) }

// SetClient sets the coordinator client. Call before Run.
func (a *Agent) SetClient(c *transport.Client) { a.client = c }

// Buffer exposes the local ring buffer (recent history for post-mortems).
func (a *Agent) Buffer() *buffer.RingBuffer { return a.buf }

// Run starts all collectors and blocks until ctx is done or every collector
// has finished (replay sources end; live ones do not).
func (a *Agent) Run(ctx context.Context) error {
	log.Printf("agent starting on node %q with %d collectors", a.config.NodeID, len(a.collectors))
	if len(a.collectors) == 0 {
		log.Println("warning: no collectors configured; nothing will be recorded")
		<-ctx.Done()
		return a.shutdown()
	}
	for _, c := range a.collectors {
		if err := c.Start(ctx); err != nil {
			_ = a.shutdown()
			return fmt.Errorf("starting collector %s: %w", c.Name(), err)
		}
	}

	merged := make(chan types.Event, 1024)
	var wg sync.WaitGroup
	for _, c := range a.collectors {
		wg.Add(1)
		go func(ch <-chan types.Event) {
			defer wg.Done()
			for e := range ch {
				select {
				case merged <- e:
				case <-ctx.Done():
					return
				}
			}
		}(c.Events())
	}
	go func() { wg.Wait(); close(merged) }()

	var out chan types.Event
	streamDone := make(chan error, 1)
	if a.client != nil {
		out = make(chan types.Event, 1024)
		go func() {
			n, err := a.client.StreamEvents(ctx, out)
			if err == nil {
				log.Printf("coordinator acknowledged %d events", n)
			}
			streamDone <- err
		}()
	}

	for e := range merged {
		a.buf.Push(e)
		if out != nil {
			select {
			case out <- e:
			case <-ctx.Done():
			}
		}
	}
	var streamErr error
	if out != nil {
		close(out)
		streamErr = <-streamDone
	}
	if err := a.shutdown(); err != nil {
		return err
	}
	if ctx.Err() != nil {
		return nil
	}
	return streamErr
}

func (a *Agent) shutdown() error {
	log.Println("agent shutting down")
	var firstErr error
	for _, c := range a.collectors {
		if err := c.Stop(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("stopping collector %s: %w", c.Name(), err)
		}
	}
	return firstErr
}
