// gfr-sim generates synthetic traces and prints a detection/attribution
// scorecard. Synthetic results validate the logic, not real-hardware behaviour.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/sim"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/trace"
)

func main() {
	seeds := flag.Int("seeds", 30, "runs per cell")
	writeDir := flag.String("write-traces", "", "write one seed-0 JSONL trace per fault into this dir and exit")
	flag.Parse()

	if *writeDir != "" {
		if err := writeTraces(*writeDir); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	fmt.Println("## Scorecard (8 ranks, 600 steps, fault at midpoint)")
	for _, jit := range []float64{0.03, 0.10} {
		fmt.Printf("\n### op-duration jitter %.0f%%, straggler lag 30%% of op\n\n", jit*100)
		fmt.Println("| fault | detected | attributed (all causes) | top cause correct | false stragglers |")
		fmt.Println("|---|---|---|---|---|")
		for _, f := range sim.AllFaults {
			row(f, sim.Params{Jitter: jit}, *seeds)
		}
	}
	fmt.Println("\n### Sensitivity: thermal fault, jitter 3%, varying straggler lag")
	fmt.Println()
	fmt.Println("| lag (% of op) | detected |")
	fmt.Println("|---|---|")
	for _, sev := range []float64{0.02, 0.05, 0.10, 0.20, 0.30} {
		det := 0
		for s := 0; s < *seeds; s++ {
			if sim.Score(sim.Params{Fault: sim.Thermal, Severity: sev, Seed: int64(s)}).Detected {
				det++
			}
		}
		fmt.Printf("| %.0f%% | %d/%d |\n", sev*100, det, *seeds)
	}
}

func row(f sim.Fault, p sim.Params, n int) {
	p.Fault = f
	var det, attr, top, fp int
	for s := 0; s < n; s++ {
		p.Seed = int64(s)
		o := sim.Score(p)
		if o.Detected {
			det++
		}
		if o.AttrCorrect {
			attr++
		}
		if o.ExactTop {
			top++
		}
		fp += o.FalseStragglers
	}
	a, t := fmt.Sprintf("%d/%d", attr, n), fmt.Sprintf("%d/%d", top, n)
	if f == sim.Clean {
		a, t = "n/a", "n/a"
	}
	fmt.Printf("| %s | %d/%d | %s | %s | %d |\n", f, det, n, a, t, fp)
}

func writeTraces(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for _, f := range sim.AllFaults {
		evs, _ := sim.Generate(sim.Params{Fault: f, Seed: 0, Steps: 300})
		out, err := os.Create(fmt.Sprintf("%s/%s.jsonl", dir, f))
		if err != nil {
			return err
		}
		err = trace.Write(out, evs)
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return err
		}
	}
	return nil
}
