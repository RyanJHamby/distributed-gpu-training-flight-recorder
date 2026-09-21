package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/agent"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "gfr",
		Short: "GPU Flight Recorder — distributed training observability",
	}

	rootCmd.AddCommand(agentCmd())
	rootCmd.AddCommand(coordinatorCmd())

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
			return a.Run(ctx)
		},
	}

	cmd.Flags().StringVar(&coordinatorAddr, "coordinator", "localhost:50051", "Coordinator gRPC address")
	cmd.Flags().StringVar(&nodeID, "node-id", "", "Unique node identifier")
	cmd.Flags().IntVar(&pollIntervalMs, "poll-interval-ms", 100, "Metric poll interval in milliseconds")
	cmd.Flags().IntVar(&bufferSize, "buffer-size", 1<<20, "Ring buffer capacity (events)")

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
			// TODO: init gRPC server, correlator, attribution engine, wire together
			_ = listenAddr
			_ = thresholdSigma

			ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
			defer cancel()

			log.Printf("coordinator starting on %s", listenAddr)
			<-ctx.Done()
			log.Println("coordinator shutting down")
			return nil
		},
	}

	cmd.Flags().StringVar(&listenAddr, "listen", ":50051", "gRPC listen address")
	cmd.Flags().Float64Var(&thresholdSigma, "threshold-sigma", 2.0, "Anomaly threshold (standard deviations)")

	return cmd
}
