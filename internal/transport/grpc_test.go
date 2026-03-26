package transport

import (
	"testing"
)

func TestNewServer(t *testing.T) {
	s := NewServer(":0")
	if s == nil {
		t.Fatal("NewServer returned nil")
	}
}

// TODO: test with bufconn, stream events end-to-end, reconnect behavior
