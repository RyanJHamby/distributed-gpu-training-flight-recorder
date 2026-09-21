package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"encoding/json"

	"github.com/spf13/cobra"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/agent"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/collector"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/coordinator"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/trace"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/transport"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "gfr",
		Short: "GPU Flight Recorder — distributed training observability",
	}

	rootCmd.AddCommand(agentCmd())
	rootCmd.AddCommand(coordinatorCmd())
	rootCmd.AddCommand(replayCmd())
	rootCmd.AddCommand(reportCmd())

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func agentCmd() *cobra.Command {
	var (
		coordinatorAddr string
		nodeID          string
		pollIntervalMs  int
		bufferSize      int
		replayFile      string
		replayRanks     []int
	)

	cmd := &cobra.Command{
		Use:   "agent",
		Short: "Run the per-node agent daemon",
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg := types.DefaultAgentConfig()
			if coordinatorAddr != "" {
				cfg.CoordinatorAddress = coordinatorAddr
			}
			if nodeID != "" {
				cfg.NodeID = nodeID
			}
			if pollIntervalMs > 0 {
				cfg.PollInterval = time.Duration(pollIntervalMs) * time.Millisecond
			}
			if bufferSize > 0 {
				cfg.BufferSize = bufferSize
			}

			ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer cancel()

			a, err := agent.New(cfg)
			if err != nil {
				return fmt.Errorf("initializing agent: %w", err)
			}
			if replayFile != "" {
				ranks := make([]uint32, len(replayRanks))
				for i, r := range replayRanks {
					ranks[i] = uint32(r)
				}
				a.AddCollector(collector.NewReplayCollector(replayFile, ranks))
			}
			if cfg.CoordinatorAddress != "" {
				cl, err := transport.NewClient(ctx, cfg.CoordinatorAddress)
				if err != nil {
					return fmt.Errorf("connecting to coordinator: %w", err)
				}
				defer cl.Close()
				a.SetClient(cl)
			}
			return a.Run(ctx)
		},
	}

	cmd.Flags().StringVar(&coordinatorAddr, "coordinator", "localhost:50051", "Coordinator gRPC address")
	cmd.Flags().StringVar(&nodeID, "node-id", "", "Unique node identifier")
	cmd.Flags().IntVar(&pollIntervalMs, "poll-interval-ms", 100, "Metric poll interval in milliseconds")
	cmd.Flags().IntVar(&bufferSize, "buffer-size", 1<<20, "Ring buffer capacity (events)")
	cmd.Flags().StringVar(&replayFile, "replay", "", "Replay a JSONL trace instead of live collectors")
	cmd.Flags().IntSliceVar(&replayRanks, "ranks", nil, "With --replay: only emit these ranks")

	return cmd
}

func coordinatorCmd() *cobra.Command {
	var (
		listenAddr     string
		thresholdSigma float64
	)

	cmd := &cobra.Command{
		Use:   "coordinator",
		Short: "Run the coordinator for cross-rank correlation",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer cancel()

			live := coordinator.NewLive(coordinator.Options{ThresholdSigma: thresholdSigma}, 0)
			srv := transport.NewServer(listenAddr, live)
			if err := srv.Listen(); err != nil {
				return err
			}
			go func() {
				<-ctx.Done()
				log.Println("coordinator shutting down")
				srv.Stop()
			}()
			return srv.Start()
		},
	}

	cmd.Flags().StringVar(&listenAddr, "listen", ":50051", "gRPC listen address")
	cmd.Flags().Float64Var(&thresholdSigma, "threshold-sigma", 4.0, "Block-median z threshold")

	return cmd
}

func replayCmd() *cobra.Command {
	var (
		thresholdSigma float64
		attrWindow     time.Duration
		asJSON         bool
	)
	cmd := &cobra.Command{
		Use:   "replay <trace.jsonl>",
		Short: "Analyse a recorded JSONL trace offline (no GPU needed)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			f, err := os.Open(args[0])
			if err != nil {
				return err
			}
			defer f.Close()
			events, err := trace.Read(f)
			if err != nil {
				return fmt.Errorf("reading %s: %w", args[0], err)
			}
			rep := coordinator.Analyze(events, coordinator.Options{
				ThresholdSigma: thresholdSigma, AttributionWindow: attrWindow})
			if asJSON {
				enc := json.NewEncoder(cmd.OutOrStdout())
				enc.SetIndent("", "  ")
				return enc.Encode(rep)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "%d events, %d ranks, %d anomalies\n", rep.Events, len(rep.Ranks), len(rep.Attributions))
			for _, a := range rep.Attributions {
				fmt.Fprintln(cmd.OutOrStdout(), "  "+a.Summary)
			}
			return nil
		},
	}
	cmd.Flags().Float64Var(&thresholdSigma, "threshold-sigma", 4, "Robust-z threshold per collective")
	cmd.Flags().DurationVar(&attrWindow, "attribution-window", 5*time.Second, "Look-back before the first flagged collective")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Emit the full report as JSON")
	return cmd
}

func reportCmd() *cobra.Command {
	var addr string
	cmd := &cobra.Command{
		Use:   "report",
		Short: "Query a running coordinator for current anomalies",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := context.WithTimeout(cmd.Context(), 30*time.Second)
			defer cancel()
			cl, err := transport.NewClient(ctx, addr)
			if err != nil {
				return err
			}
			defer cl.Close()
			rep, err := cl.GetReport(ctx)
			if err != nil {
				return fmt.Errorf("querying %s: %w", addr, err)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "%d anomalies\n", len(rep.Anomalies))
			for _, a := range rep.Anomalies {
				fmt.Fprintln(cmd.OutOrStdout(), "  "+a.AttributionSummary)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&addr, "coordinator", "localhost:50051", "Coordinator gRPC address")
	return cmd
}
