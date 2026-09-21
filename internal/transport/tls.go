package transport

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"

	"google.golang.org/grpc/credentials"
)

func loadPool(caFile string) (*x509.CertPool, error) {
	pem, err := os.ReadFile(caFile)
	if err != nil {
		return nil, err
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("no certificates found in %s", caFile)
	}
	return pool, nil
}

// ServerCredentials builds mutual-TLS credentials for the coordinator: it
// presents certFile/keyFile and requires every agent to present a certificate
// signed by clientCAFile. TLS 1.3 only.
func ServerCredentials(certFile, keyFile, clientCAFile string) (credentials.TransportCredentials, error) {
	if clientCAFile == "" {
		return nil, errors.New("mTLS requires a client CA: refusing to start TLS without client authentication")
	}
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}
	pool, err := loadPool(clientCAFile)
	if err != nil {
		return nil, err
	}
	return credentials.NewTLS(&tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientCAs:    pool,
		ClientAuth:   tls.RequireAndVerifyClientCert,
		MinVersion:   tls.VersionTLS13,
	}), nil
}

// ClientCredentials builds mutual-TLS credentials for an agent: it verifies the
// coordinator against caFile and presents certFile/keyFile. serverName overrides
// the name checked against the coordinator certificate (empty: use the dial host).
func ClientCredentials(caFile, certFile, keyFile, serverName string) (credentials.TransportCredentials, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}
	pool, err := loadPool(caFile)
	if err != nil {
		return nil, err
	}
	return credentials.NewTLS(&tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      pool,
		ServerName:   serverName,
		MinVersion:   tls.VersionTLS13,
	}), nil
}
