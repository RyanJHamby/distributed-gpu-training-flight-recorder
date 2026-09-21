package transport

import (
	"context"
	"errors"
	"io"
	"log"
	"net"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/RyanJHamby/distributed-gpu-training-flight-recorder/api/proto"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/attribution"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// Handler receives decoded events on the coordinator.
type Handler interface {
	Ingest(types.Event)
	// Report analyses everything ingested so far.
	Report() []attribution.Attribution
}

// Server is the coordinator-side gRPC server.
type Server struct {
	pb.UnimplementedEventStreamServer
	pb.UnimplementedAnomalyReportServer

	listenAddr string
	grpcServer *grpc.Server
	handler    Handler
	lis        net.Listener
	received   atomic.Uint64
}

// NewServer returns a server; handler may be nil for a server that only
// counts events (used by tests of the transport itself).
func NewServer(listenAddr string, handler Handler) *Server {
	s := &Server{listenAddr: listenAddr, grpcServer: grpc.NewServer(), handler: handler}
	pb.RegisterEventStreamServer(s.grpcServer, s)
	pb.RegisterAnomalyReportServer(s.grpcServer, s)
	return s
}

// Listen binds the address; Addr is valid afterwards (useful with ":0").
func (s *Server) Listen() error {
	lis, err := net.Listen("tcp", s.listenAddr)
	s.lis = lis
	return err
}

func (s *Server) Addr() string { return s.lis.Addr().String() }

// Start listens if needed and serves until Stop.
func (s *Server) Start() error {
	if s.lis == nil {
		if err := s.Listen(); err != nil {
			return err
		}
	}
	log.Printf("gRPC server listening on %s", s.lis.Addr())
	return s.grpcServer.Serve(s.lis)
}

func (s *Server) Stop() { s.grpcServer.GracefulStop() }

// StreamEvents drains a client stream. Malformed messages are counted and
// skipped: one bad event must not tear down an agent's whole stream.
func (s *Server) StreamEvents(stream pb.EventStream_StreamEventsServer) error {
	var n uint64
	for {
		m, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return stream.SendAndClose(&pb.StreamAck{EventsReceived: n})
		}
		if err != nil {
			return err
		}
		e, err := FromProto(m)
		if err != nil {
			log.Printf("dropping malformed event from rank %d: %v", m.Rank, err)
			continue
		}
		if s.handler != nil {
			s.handler.Ingest(e)
		}
		n++
		s.received.Add(1)
	}
}

// Received is the total number of events accepted across all streams.
func (s *Server) Received() uint64 { return s.received.Load() }

func (s *Server) GetReport(_ context.Context, _ *pb.ReportRequest) (*pb.ReportResponse, error) {
	resp := &pb.ReportResponse{}
	if s.handler == nil {
		return resp, nil
	}
	for _, a := range s.handler.Report() {
		d := &pb.AnomalyDetail{
			DetectedAtNs: a.Anomaly.DetectedAt, StragglerRank: a.Anomaly.StragglerRank, OpType: a.Anomaly.OpType,
			LagNs:          a.Anomaly.LagNs,
			DeviationSigma: a.Anomaly.DeviationSigma, AttributionSummary: a.Summary,
			Hits: int64(a.Anomaly.Hits), Groups: int64(a.Anomaly.Groups),
		}
		for _, l := range a.CausalChain {
			d.CausalChain = append(d.CausalChain, &pb.CausalLink{Cause: string(l.Cause), Confidence: l.Confidence, Detail: l.Detail})
		}
		resp.Anomalies = append(resp.Anomalies, d)
	}
	return resp, nil
}

// Client is the agent-side gRPC client for streaming events to the coordinator.
type Client struct {
	conn    *grpc.ClientConn
	events  pb.EventStreamClient
	reports pb.AnomalyReportClient
}

// NewClient dials lazily (grpc.NewClient does not connect until first use).
// Plaintext: development only; production would use mTLS credentials.
func NewClient(_ context.Context, coordinatorAddr string) (*Client, error) {
	conn, err := grpc.NewClient(coordinatorAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, events: pb.NewEventStreamClient(conn), reports: pb.NewAnomalyReportClient(conn)}, nil
}

// StreamEvents sends events from ch until it closes or ctx ends, then returns
// the server's ack count. It retries the stream open with backoff, but events
// already sent on a broken stream are not replayed: the ring buffer, not the
// wire, is the durable record.
func (c *Client) StreamEvents(ctx context.Context, ch <-chan types.Event) (uint64, error) {
	backoff := 100 * time.Millisecond
	for {
		stream, err := c.events.StreamEvents(ctx)
		if err == nil {
			return c.pump(ctx, stream, ch)
		}
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-time.After(backoff):
			if backoff < 5*time.Second {
				backoff *= 2
			}
		}
	}
}

func (c *Client) pump(ctx context.Context, stream pb.EventStream_StreamEventsClient, ch <-chan types.Event) (uint64, error) {
	for {
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case e, ok := <-ch:
			if !ok {
				ack, err := stream.CloseAndRecv()
				if err != nil {
					return 0, err
				}
				return ack.EventsReceived, nil
			}
			m, err := ToProto(e)
			if err != nil {
				log.Printf("skipping unsendable event: %v", err)
				continue
			}
			if err := stream.Send(m); err != nil {
				return 0, err
			}
		}
	}
}

// GetReport asks the coordinator for its current anomalies.
func (c *Client) GetReport(ctx context.Context) (*pb.ReportResponse, error) {
	return c.reports.GetReport(ctx, &pb.ReportRequest{})
}

func (c *Client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}
