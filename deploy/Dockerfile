# Stage 1: Build
FROM golang:1.25-bookworm AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -o /bin/gfr ./cmd/gfr

# Stage 2: Runtime
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /bin/gfr /usr/local/bin/gfr

USER nonroot:nonroot
ENTRYPOINT ["gfr"]
