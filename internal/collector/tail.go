package collector

import (
	"bytes"
	"context"
	"io"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/trace"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// TailCollector follows JSONL files matching a glob (the per-rank files the
// Python shim writes) and emits each complete line as an event. Files that
// appear later are picked up on the next poll. A trailing partial line is held
// until its newline arrives, so a writer mid-append never yields a bad event.
type TailCollector struct {
	glob     string
	interval time.Duration
	events   chan types.Event

	stop     context.CancelFunc
	stopOnce sync.Once
	done     chan struct{}
}

func NewTailCollector(glob string, interval time.Duration) *TailCollector {
	if interval <= 0 {
		interval = 100 * time.Millisecond
	}
	return &TailCollector{glob: glob, interval: interval, events: make(chan types.Event, 1024), done: make(chan struct{})}
}

func (c *TailCollector) Name() string { return "tail" }

type tailState struct {
	off     int64
	partial []byte
}

func (c *TailCollector) Start(ctx context.Context) error {
	if _, err := filepath.Match(c.glob, ""); err != nil {
		return err // malformed pattern: fail fast
	}
	ctx, c.stop = context.WithCancel(ctx)
	go func() {
		defer close(c.done)
		defer close(c.events)
		files := map[string]*tailState{}
		t := time.NewTicker(c.interval)
		defer t.Stop()
		for {
			matches, _ := filepath.Glob(c.glob)
			for _, m := range matches {
				st := files[m]
				if st == nil {
					st = &tailState{}
					files[m] = st
				}
				if !c.drain(ctx, m, st) {
					return
				}
			}
			select {
			case <-ctx.Done():
				return
			case <-t.C:
			}
		}
	}()
	return nil
}

// drain reads new bytes from one file; false means ctx was cancelled.
func (c *TailCollector) drain(ctx context.Context, path string, st *tailState) bool {
	f, err := os.Open(path)
	if err != nil {
		return true
	}
	defer f.Close()
	if fi, err := f.Stat(); err == nil && fi.Size() < st.off { // truncated/rotated
		st.off, st.partial = 0, nil
	}
	if _, err := f.Seek(st.off, io.SeekStart); err != nil {
		return true
	}
	data, err := io.ReadAll(f)
	if err != nil || len(data) == 0 {
		return true
	}
	st.off += int64(len(data))
	buf := append(st.partial, data...)
	for {
		i := bytes.IndexByte(buf, '\n')
		if i < 0 {
			break
		}
		line := bytes.TrimSpace(buf[:i])
		buf = buf[i+1:]
		if len(line) == 0 {
			continue
		}
		e, err := trace.DecodeLine(line)
		if err != nil {
			log.Printf("tail %s: skipping bad line: %v", path, err)
			continue
		}
		select {
		case c.events <- e:
		case <-ctx.Done():
			return false
		}
	}
	st.partial = append([]byte(nil), buf...)
	return true
}

func (c *TailCollector) Stop() error {
	c.stopOnce.Do(func() {
		if c.stop != nil {
			c.stop()
			<-c.done
		}
	})
	return nil
}

func (c *TailCollector) Events() <-chan types.Event { return c.events }
