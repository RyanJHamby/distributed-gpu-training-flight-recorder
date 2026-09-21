.PHONY: sim build test lint proto docker-build run-agent run-coordinator clean

BINARY := gfr
BUILD_DIR := bin
GO := go
GOFLAGS := -trimpath

# Build the gfr binary
build:
	$(GO) build $(GOFLAGS) -o $(BUILD_DIR)/$(BINARY) ./cmd/gfr

# Run all tests
test:
	$(GO) test -race -count=1 ./...

# Run linter
lint:
	golangci-lint run ./...

# Generate protobuf code
proto:
	protoc \
		--go_out=. --go_opt=paths=source_relative \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative \
		api/proto/flightrecorder.proto

# Build Docker image
docker-build:
	docker build -t gpu-flight-recorder:latest -f deploy/Dockerfile .

# Run agent locally
run-agent: build
	$(BUILD_DIR)/$(BINARY) agent --node-id=$$(hostname)

# Run coordinator locally
run-coordinator: build
	$(BUILD_DIR)/$(BINARY) coordinator

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Regenerate the synthetic scorecard
sim:
	go run ./cmd/gfr-sim > docs/scorecard.md

# Static Linux binary for rented GPU boxes (scoring runs there; no Go toolchain needed)
build-linux:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -trimpath -o bin/gfr-linux ./cmd/gfr
