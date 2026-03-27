#!/usr/bin/env bash
# Dev environment setup for GPU Flight Recorder.
set -euo pipefail

echo "Setting up dev environment..."

if ! command -v go &> /dev/null; then
    echo "Go not found. Install Go 1.22+ from https://go.dev/dl/"
    exit 1
fi

echo "Go version: $(go version)"

# protoc plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# linter
if ! command -v golangci-lint &> /dev/null; then
    echo "Installing golangci-lint..."
    go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
fi

go mod download
go build ./...

echo "Done. Run 'make build' to build, 'make test' to test."
