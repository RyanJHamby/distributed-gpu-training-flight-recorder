package trace

import (
	"os"
	"testing"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// testdata/shim/rank1.jsonl is produced by shim/gfr_torch.py (with a fake NVML).
// This pins the Python -> Go contract: field names and units must decode.
func TestShimOutputDecodes(t *testing.T) {
	f, err := os.Open("../../testdata/shim/rank1.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	evs, err := Read(f)
	if err != nil {
		t.Fatal(err)
	}
	var coll, gpu, pcie int
	for _, e := range evs {
		switch v := e.(type) {
		case types.NCCLCollectiveEvent:
			coll++
			if v.RankID != 1 || v.PGID != "0" || v.OpType != "AllReduce" || v.NodeID != "node-a" || v.DurationNs == 0 {
				t.Fatalf("bad collective %+v", v)
			}
		case types.GPUMetricEvent:
			gpu++
			if v.ThrottleReasons != 0x40 || v.PowerWatts != 250 || v.GPUUUID != "GPU-abc" {
				t.Fatalf("bad gpu metric %+v", v)
			}
		case types.PCIeBandwidthEvent:
			pcie++
			if v.ReadBandwidthMBs != 2000 {
				t.Fatalf("bad pcie %+v", v)
			}
		}
	}
	if coll != 2 || gpu != 1 || pcie != 1 {
		t.Fatalf("counts coll=%d gpu=%d pcie=%d", coll, gpu, pcie)
	}
}
