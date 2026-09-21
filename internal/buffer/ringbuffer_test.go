package buffer

import (
	"sync"
	"testing"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

func ev(ts int64) types.Event { return types.GPUMetricEvent{BaseEvent: types.BaseEvent{Timestamp: ts}} }

func TestCapacityRoundsUpToPowerOfTwo(t *testing.T) {
	for in, want := range map[int]int{-3: 1, 0: 1, 1: 1, 2: 2, 3: 4, 5: 8, 8: 8, 1000: 1024} {
		if got := NewRingBuffer(in).Cap(); got != want {
			t.Errorf("cap(%d)=%d want %d", in, got, want)
		}
	}
}

func TestPushLenAndOrder(t *testing.T) {
	rb := NewRingBuffer(8)
	for i := 1; i <= 5; i++ {
		rb.Push(ev(int64(i)))
	}
	if rb.Len() != 5 {
		t.Fatalf("len=%d want 5", rb.Len())
	}
	for i, e := range rb.Snapshot() {
		if e.TimestampNs() != int64(i+1) {
			t.Fatalf("order broken at %d: %d", i, e.TimestampNs())
		}
	}
}

func TestOverwriteOldestOnWrap(t *testing.T) {
	rb := NewRingBuffer(4)
	for i := 1; i <= 10; i++ {
		rb.Push(ev(int64(i)))
	}
	snap := rb.Snapshot()
	if len(snap) != 4 || snap[0].TimestampNs() != 7 || snap[3].TimestampNs() != 10 {
		t.Fatalf("wrong retained window: %v", snap)
	}
	if rb.Dropped() != 6 {
		t.Fatalf("dropped=%d want 6", rb.Dropped())
	}
}

func TestRangeEarlyStop(t *testing.T) {
	rb := NewRingBuffer(8)
	for i := 1; i <= 5; i++ {
		rb.Push(ev(int64(i)))
	}
	n := 0
	rb.Range(func(types.Event) bool { n++; return n < 2 })
	if n != 2 {
		t.Fatalf("visited %d want 2", n)
	}
}

func TestConcurrentPushRace(t *testing.T) {
	rb := NewRingBuffer(64)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 1000; i++ {
				rb.Push(ev(int64(i)))
				_ = rb.Len()
				_ = rb.Snapshot()
			}
		}()
	}
	wg.Wait()
	if rb.Len() != 64 || rb.Dropped() != 8000-64 {
		t.Fatalf("len=%d dropped=%d", rb.Len(), rb.Dropped())
	}
}

func BenchmarkPush(b *testing.B) {
	rb := NewRingBuffer(1 << 16)
	e := ev(1)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rb.Push(e)
	}
}
