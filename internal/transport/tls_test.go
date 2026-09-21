package transport

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type ca struct {
	cert *x509.Certificate
	key  *ecdsa.PrivateKey
	pem  []byte
}

func newCA(t *testing.T, cn string) *ca {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(1), Subject: pkix.Name{CommonName: cn},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour),
		IsCA: true, BasicConstraintsValid: true, KeyUsage: x509.KeyUsageCertSign}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	c, _ := x509.ParseCertificate(der)
	return &ca{c, key, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})}
}

// issue writes cert/key PEM files signed by c and returns their paths.
func (c *ca) issue(t *testing.T, dir, name string, server bool) (certPath, keyPath string) {
	t.Helper()
	key, _ := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	tmpl := &x509.Certificate{SerialNumber: big.NewInt(time.Now().UnixNano()), Subject: pkix.Name{CommonName: name},
		NotBefore: time.Now().Add(-time.Hour), NotAfter: time.Now().Add(time.Hour)}
	if server {
		tmpl.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
		tmpl.DNSNames = []string{"localhost"}
		tmpl.IPAddresses = []net.IP{net.ParseIP("127.0.0.1")}
	} else {
		tmpl.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, c.cert, &key.PublicKey, c.key)
	if err != nil {
		t.Fatal(err)
	}
	kb, _ := x509.MarshalECPrivateKey(key)
	certPath, keyPath = filepath.Join(dir, name+".crt"), filepath.Join(dir, name+".key")
	os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600)
	os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: kb}), 0o600)
	return
}

func (c *ca) file(t *testing.T, dir, name string) string {
	p := filepath.Join(dir, name)
	os.WriteFile(p, c.pem, 0o600)
	return p
}

// call makes a fail-fast unary RPC. Unlike the streaming client (which retries
// until its context ends), a refused handshake surfaces immediately as an
// error, so a rejection cannot be confused with a slow or absent server.
func call(t *testing.T, cl *Client) error {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, err := cl.GetReport(ctx)
	return err
}

func TestMutualTLS(t *testing.T) {
	dir := t.TempDir()
	good, other := newCA(t, "gfr-ca"), newCA(t, "rogue-ca")
	srvCert, srvKey := good.issue(t, dir, "server", true)
	srvCreds, err := ServerCredentials(srvCert, srvKey, good.file(t, dir, "ca.pem"))
	if err != nil {
		t.Fatal(err)
	}
	srv := NewServer("127.0.0.1:0", nil, srvCreds)
	if err := srv.Listen(); err != nil {
		t.Fatal(err)
	}
	go srv.Start()
	defer srv.Stop()

	dial := func(caPath, cert, key string) *Client {
		cc, err := ClientCredentials(caPath, cert, key, "localhost")
		if err != nil {
			t.Fatal(err)
		}
		cl, err := NewClient(context.Background(), srv.Addr(), cc)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cl.Close() })
		return cl
	}

	t.Run("valid client certificate is accepted", func(t *testing.T) {
		c, k := good.issue(t, dir, "agent-ok", false)
		if err := call(t, dial(good.file(t, dir, "ca.pem"), c, k)); err != nil {
			t.Fatalf("valid mTLS client rejected: %v", err)
		}
	})
	t.Run("client certificate from an untrusted CA is rejected", func(t *testing.T) {
		c, k := other.issue(t, dir, "agent-rogue", false)
		err := call(t, dial(good.file(t, dir, "ca.pem"), c, k))
		if status.Code(err) != codes.Unavailable {
			t.Fatalf("want handshake refusal (Unavailable), got %v", err)
		}
	})
	t.Run("plaintext client is rejected", func(t *testing.T) {
		cl, _ := NewClient(context.Background(), srv.Addr())
		defer cl.Close()
		if status.Code(call(t, cl)) != codes.Unavailable {
			t.Fatal("plaintext client was accepted by a TLS server")
		}
	})
	t.Run("client refuses a server it does not trust", func(t *testing.T) {
		c, k := good.issue(t, dir, "agent-ok2", false)
		err := call(t, dial(other.file(t, dir, "wrong-ca.pem"), c, k))
		if status.Code(err) != codes.Unavailable {
			t.Fatalf("client trusted a server signed by an unknown CA: %v", err)
		}
	})
}

func TestServerCredentialsRequireClientCA(t *testing.T) {
	if _, err := ServerCredentials("c", "k", ""); err == nil {
		t.Fatal("TLS without client auth must be refused")
	}
}
