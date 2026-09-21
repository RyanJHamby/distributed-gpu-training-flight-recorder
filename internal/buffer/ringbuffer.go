package buffer

import (
	"math/bits"
	"sync"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// RingBuffer is a fixed-capacity, overwrite-oldest buffer for per-rank event
// storage. It is mutex-guarded: overwrite-on-full forces the producer to move
// the read cursor, which a lock-free SPSC design cannot do safely.
// Capacity is rounded up to a power of two so indexing is a mask, not a mod.
type RingBuffer struct {
	mu      sync.Mutex
	events  []types.Event
	mask    uint64
	written uint64 // total events ever pushed
}

// NewRingBuffer returns a buffer holding at least capacity events (minimum 1).
func NewRingBuffer(capacity int) *RingBuffer {
	if capacity < 1 {
		capacity = 1
	}
	size := uint64(1) << bits.Len64(uint64(capacity-1))
	return &RingBuffer{events: make([]types.Event, size), mask: size - 1}
}

// Cap returns the actual (power-of-two) capacity.
func (rb *RingBuffer) Cap() int { return len(rb.events) }

// Push adds an event. If full, the oldest event is overwritten.
func (rb *RingBuffer) Push(event types.Event) {
	rb.mu.Lock()
	rb.events[rb.written&rb.mask] = event
	rb.written++
	rb.mu.Unlock()
}

// Len returns the number of events currently retained.
func (rb *RingBuffer) Len() int {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	return rb.lenLocked()
}

// Dropped returns how many events have been overwritten.
func (rb *RingBuffer) Dropped() uint64 {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	return rb.written - uint64(rb.lenLocked())
}

func (rb *RingBuffer) lenLocked() int {
	if rb.written > uint64(len(rb.events)) {
		return len(rb.events)
	}
	return int(rb.written)
}

// Snapshot returns a copy of retained events, oldest to newest.
func (rb *RingBuffer) Snapshot() []types.Event {
	rb.mu.Lock()
	defer rb.mu.Unlock()
	n := rb.lenLocked()
	out := make([]types.Event, n)
	start := rb.written - uint64(n)
	for i := 0; i < n; i++ {
		out[i] = rb.events[(start+uint64(i))&rb.mask]
	}
	return out
}

// Range iterates a snapshot oldest-to-newest. Return false from fn to stop.
// It iterates a copy, so fn may call back into the buffer without deadlock.
func (rb *RingBuffer) Range(fn func(types.Event) bool) {
	for _, e := range rb.Snapshot() {
		if !fn(e) {
			return
		}
	}
}
