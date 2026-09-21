// Package trace reads and writes JSONL event traces: one {"type","event"}
// envelope per line. It is the replay format for the CPU-only path.
package trace

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

const (
	TypeGPUMetric = "gpu_metric"
	TypeNCCL      = "nccl_collective"
	TypePCIe      = "pcie_bandwidth"
	TypeNVLink    = "nvlink"
	TypeThermal   = "thermal"
)

type envelope struct {
	Type  string          `json:"type"`
	Event json.RawMessage `json:"event"`
}

func typeOf(e types.Event) (string, error) {
	switch e.(type) {
	case types.GPUMetricEvent:
		return TypeGPUMetric, nil
	case types.NCCLCollectiveEvent:
		return TypeNCCL, nil
	case types.PCIeBandwidthEvent:
		return TypePCIe, nil
	case types.NVLinkEvent:
		return TypeNVLink, nil
	case types.ThermalEvent:
		return TypeThermal, nil
	}
	return "", fmt.Errorf("unknown event type %T", e)
}

// Write encodes events as JSONL.
func Write(w io.Writer, events []types.Event) error {
	enc := json.NewEncoder(w)
	for _, e := range events {
		t, err := typeOf(e)
		if err != nil {
			return err
		}
		raw, err := json.Marshal(e)
		if err != nil {
			return err
		}
		if err := enc.Encode(envelope{t, raw}); err != nil {
			return err
		}
	}
	return nil
}

// Read decodes JSONL, skipping blank lines. Errors carry the line number.
func Read(r io.Reader) ([]types.Event, error) {
	var out []types.Event
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	for line := 1; sc.Scan(); line++ {
		b := sc.Bytes()
		if len(b) == 0 {
			continue
		}
		var env envelope
		if err := json.Unmarshal(b, &env); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		e, err := decode(env)
		if err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		out = append(out, e)
	}
	return out, sc.Err()
}

func decode(env envelope) (types.Event, error) {
	var (
		err error
		e   types.Event
	)
	switch env.Type {
	case TypeGPUMetric:
		var v types.GPUMetricEvent
		err, e = json.Unmarshal(env.Event, &v), v
	case TypeNCCL:
		var v types.NCCLCollectiveEvent
		err, e = json.Unmarshal(env.Event, &v), v
	case TypePCIe:
		var v types.PCIeBandwidthEvent
		err, e = json.Unmarshal(env.Event, &v), v
	case TypeNVLink:
		var v types.NVLinkEvent
		err, e = json.Unmarshal(env.Event, &v), v
	case TypeThermal:
		var v types.ThermalEvent
		err, e = json.Unmarshal(env.Event, &v), v
	default:
		return nil, fmt.Errorf("unknown event type %q", env.Type)
	}
	return e, err
}
