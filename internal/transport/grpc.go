package transport

import (
	"context"
	"log"
	"net"

	"google.golang.org/grpc"
)

// Server is the coordinator-side gRPC server.
type Server struct {
	listenAddr string
	grpcServer *grpc.Server
}

func NewServer(listenAddr string) *Server {
	return &Server{
		listenAddr: listenAddr,
		grpcServer: grpc.NewServer(),
	}
}

func (s *Server) Start() error {
	lis, err := net.Listen("tcp", s.listenAddr)
	if err != nil {
		return err
	}
	log.Printf("gRPC server listening on %s", s.listenAddr)
	// TODO: register EventStream and AnomalyReport service implementations
	return s.grpcServer.Serve(lis)
}

func (s *Server) Stop() {
	s.grpcServer.GracefulStop()
}

// Client is the agent-side gRPC client for streaming events to the coordinator.
type Client struct {
	conn *grpc.ClientConn
}

func NewClient(ctx context.Context, coordinatorAddr string) (*Client, error) {
	// TODO: dial with insecure creds for dev, mTLS for prod
	_ = ctx
	_ = coordinatorAddr
	return &Client{}, nil
}

func (c *Client) StreamEvents(ctx context.Context) error {
	// TODO: open StreamEvents RPC, read from event channel, convert to proto, send
	return nil
}

func (c *Client) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}
