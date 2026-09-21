package buffer

import (
	"sync/atomic"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// RingBuffer is a lock-free SPSC ring buffer for per-rank event storage.
// Uses atomic read/write indices instead of mutexes. Size must be power of 2.
type RingBuffer struct {
	events   []types.Event
	size     uint64
	mask     uint64 // size-1 for fast mod
	writeIdx atomic.Uint64
	readIdx  atomic.Uint64
}

func NewRingBuffer(capacity int) *RingBuffer {
	// TODO: round up to next power of 2
	size := uint64(capacity)
	return &RingBuffer{
		events: make([]types.Event, size),
		size:   size,
		mask:   size - 1,
	}
}

// Push adds an event. If full, the oldest event is silently overwritten.
func (rb *RingBuffer) Push(event types.Event) {
	// TODO: store at events[writeIdx & mask], increment writeIdx,
	// advance readIdx if consumer fell behind (writeIdx - readIdx >= size)
}

func (rb *RingBuffer) Len() int {
	w := rb.writeIdx.Load()
	r := rb.readIdx.Load()
	return int(w - r)
}

// Range iterates events oldest-to-newest. Return false from fn to stop.
func (rb *RingBuffer) Range(fn func(types.Event) bool) {
	// TODO: snapshot read/write idx, iterate [readIdx, writeIdx) with mask
}
