package collector

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

func recv(t *testing.T, ch <-chan types.Event, n int) []types.Event {
	t.Helper()
	var out []types.Event
	timeout := time.After(3 * time.Second)
	for len(out) < n {
		select {
		case e, ok := <-ch:
			if !ok {
				t.Fatalf("channel closed after %d/%d events", len(out), n)
			}
			out = append(out, e)
		case <-timeout:
			t.Fatalf("timed out with %d/%d events", len(out), n)
		}
	}
	return out
}

const line1 = `{"type":"thermal","event":{"timestamp_ns":1,"rank":0,"temperature_c":80,"throttle_active":true}}` + "\n"
const line2 = `{"type":"thermal","event":{"timestamp_ns":2,"rank":1,"temperature_c":70,"throttle_active":false}}` + "\n"

func TestTailHandlesPartialLinesLateFilesAndBadLines(t *testing.T) {
	dir := t.TempDir()
	c := NewTailCollector(filepath.Join(dir, "rank*.jsonl"), 10*time.Millisecond)
	if err := c.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	defer c.Stop()

	f, _ := os.Create(filepath.Join(dir, "rank0.jsonl"))
	f.WriteString(line1[:20]) // partial line: must not be emitted or dropped
	time.Sleep(60 * time.Millisecond)
	f.WriteString(line1[20:] + "garbage\n") // completes line 1, then a bad line
	f.Close()
	if got := recv(t, c.Events(), 1); got[0].TimestampNs() != 1 {
		t.Fatalf("got %v", got)
	}
	// A file that appears later is picked up.
	os.WriteFile(filepath.Join(dir, "rank1.jsonl"), []byte(line2), 0o644)
	if got := recv(t, c.Events(), 1); got[0].Rank() != 1 {
		t.Fatalf("got %v", got)
	}
}

func TestTailStopIsIdempotentAndClosesChannel(t *testing.T) {
	c := NewTailCollector(filepath.Join(t.TempDir(), "*.jsonl"), 10*time.Millisecond)
	c.Start(context.Background())
	c.Stop()
	c.Stop()
	if _, ok := <-c.Events(); ok {
		t.Fatal("channel should be closed")
	}
}

func TestReplayRankFilterAndEnd(t *testing.T) {
	p := filepath.Join(t.TempDir(), "t.jsonl")
	os.WriteFile(p, []byte(line1+line2), 0o644)
	c := NewReplayCollector(p, []uint32{1})
	if err := c.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	got := recv(t, c.Events(), 1)
	if got[0].Rank() != 1 {
		t.Fatal("rank filter ignored")
	}
	if _, ok := <-c.Events(); ok {
		t.Fatal("replay should close when exhausted")
	}
	c.Stop()
	c.Stop()
}

func TestStopBeforeStartIsSafe(t *testing.T) {
	NewReplayCollector("x", nil).Stop()
	NewTailCollector("x", 0).Stop()
}
