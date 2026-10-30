//go:build spec_gfr_ringbuf

package buffer

import (
	"sync"
	"testing"

	"github.com/ryanhamby/gpu-flight-recorder/internal/types"
)

func collect(rb *RingBuffer) []types.Event {
	var out []types.Event
	rb.Range(func(e types.Event) bool {
		out = append(out, e)
		return true
	})
	return out
}

func TestCapacityRoundsUpToPowerOfTwo(t *testing.T) {
	cases := []struct{ in, want int }{{1000, 1024}, {1024, 1024}, {3, 4}, {1, 1}, {0, 1}, {-5, 1}}
	for _, c := range cases {
		rb := NewRingBuffer(c.in)
		for i := 0; i < 5000; i++ {
			rb.Push(makeEvent(int64(i), 0))
		}
		if got := rb.Len(); got != c.want {
			t.Errorf("NewRingBuffer(%d): Len after overfilling = %d, want capacity %d", c.in, got, c.want)
		}
	}
}

func TestPushLenAndRangeOrderBeforeWrap(t *testing.T) {
	rb := NewRingBuffer(16)
	if got := len(collect(rb)); got != 0 {
		t.Fatalf("empty buffer yielded %d events", got)
	}
	for i := 0; i < 10; i++ {
		rb.Push(makeEvent(int64(i), uint32(i%3)))
	}
	if rb.Len() != 10 {
		t.Fatalf("Len = %d, want 10", rb.Len())
	}
	got := collect(rb)
	if len(got) != 10 {
		t.Fatalf("Range yielded %d events, want 10", len(got))
	}
	for i, e := range got {
		if e.TimestampNs() != int64(i) || e.Rank() != uint32(i%3) {
			t.Errorf("event %d = (ts %d, rank %d), want (ts %d, rank %d)", i, e.TimestampNs(), e.Rank(), i, i%3)
		}
	}
	if rb.Len() != 10 {
		t.Errorf("Range must not consume events: Len = %d", rb.Len())
	}
}

func TestWraparoundKeepsNewestInOrder(t *testing.T) {
	rb := NewRingBuffer(8)
	for i := 0; i < 20; i++ {
		rb.Push(makeEvent(int64(i), 0))
	}
	if rb.Len() != 8 {
		t.Fatalf("Len = %d, want 8 (full)", rb.Len())
	}
	got := collect(rb)
	if len(got) != 8 {
		t.Fatalf("Range yielded %d events, want 8", len(got))
	}
	for i, e := range got {
		if want := int64(12 + i); e.TimestampNs() != want {
			t.Errorf("position %d: ts %d, want %d (oldest 12 overwritten up to 11)", i, e.TimestampNs(), want)
		}
	}
	// Pushing exactly capacity more must replace every slot again.
	for i := 20; i < 28; i++ {
		rb.Push(makeEvent(int64(i), 0))
	}
	got = collect(rb)
	if len(got) != 8 || got[0].TimestampNs() != 20 || got[7].TimestampNs() != 27 {
		t.Errorf("second lap wrong: %d events, first/last ts unexpected", len(got))
	}
}

func TestRangeStopsWhenCallbackReturnsFalse(t *testing.T) {
	rb := NewRingBuffer(8)
	for i := 0; i < 8; i++ {
		rb.Push(makeEvent(int64(i), 0))
	}
	calls := 0
	rb.Range(func(types.Event) bool {
		calls++
		return calls < 3
	})
	if calls != 3 {
		t.Errorf("callback called %d times, want 3", calls)
	}
}

func TestPushStoresTheSameEventValue(t *testing.T) {
	rb := NewRingBuffer(4)
	e := makeEvent(77, 9)
	rb.Push(e)
	got := collect(rb)
	if len(got) != 1 || got[0] != e {
		t.Fatal("Range must return the exact event that was pushed")
	}
}

// One producer pushes while another goroutine polls Len(). Len must always
// stay within [0, capacity] and the run must be clean under -race. Range
// concurrent with Push is NOT part of the contract and is not exercised.
func TestSingleProducerConcurrentLenIsBoundedAndRaceFree(t *testing.T) {
	const capacity, total = 64, 20000
	rb := NewRingBuffer(capacity)
	done := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-done:
				return
			default:
				if n := rb.Len(); n < 0 || n > capacity {
					t.Errorf("Len() = %d outside [0,%d]", n, capacity)
					return
				}
			}
		}
	}()
	for i := 0; i < total; i++ {
		rb.Push(makeEvent(int64(i), 0))
	}
	close(done)
	wg.Wait()
	got := collect(rb)
	if len(got) != capacity {
		t.Fatalf("after %d pushes Range yielded %d events, want %d", total, len(got), capacity)
	}
	for i, e := range got {
		if want := int64(total - capacity + i); e.TimestampNs() != want {
			t.Fatalf("position %d: ts %d, want %d", i, e.TimestampNs(), want)
		}
	}
}
