package collector

import (
	"context"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// Collector is the interface all metric sources implement.
type Collector interface {
	Name() string
	Start(ctx context.Context) error
	Stop() error
	Events() <-chan types.Event
}
