#!/usr/bin/env bash
# Stage 16 — TLS + client authentication (ADR-4).
#
# Generates, entirely locally:
#   1. A self-signed local certificate authority (certs/ca.{key,crt}).
#   2. A SuperLink server certificate signed by that CA (certs/server.{key,pem}),
#      with SANs covering localhost/127.0.0.1/::1 (host-only deployment, Stage 16)
#      AND the `superlink` Docker Compose service name (Stage 17) — found
#      necessary via a real failed deployment run, not by inspection: TLS
#      hostname verification silently fails (repeated "Connection attempt
#      failed" with no clear error surfaced) when a client connects via a
#      hostname absent from the server cert's SAN list, which `superlink`
#      is when containers reach it over the Compose network by service name
#      rather than 127.0.0.1.
#   3. One ECDSA-384 keypair per hospital (certs/hospital_{A,B,C}[.pub]) for
#      Flower's node-authentication mechanism (`--enable-supernode-auth` on the
#      SuperLink, `--auth-supernode-private-key` on each SuperNode, the public
#      half pre-registered via `flwr supernode register`).
#
# Uses only openssl and ssh-keygen (ADR-3's "no custom cryptography" spirit
# applies equally here — no hand-rolled key generation).
#
# The exact SuperLink/SuperNode auth mechanism was verified against the
# pinned flwr==1.35.0's actual CLI (`flower-superlink --help`,
# `flower-supernode --help`) and cross-checked against Flower's own tested
# reference (`framework/e2e/e2e-bare-auth/generate.sh` in flwrlabs/flower on
# GitHub) — per ADR-4's own note that this area has changed across releases,
# and ADR-5's rule to verify against the pinned version, not memory. The old
# `--auth-list-public-keys` flag this project's earlier CLAUDE.md drafts might
# assume is gone entirely at this version (hard error, not just deprecated);
# `--enable-supernode-auth` + `flwr supernode register` is the current,
# working mechanism.
#
# Output goes to certs/ (gitignored — verify with `git check-ignore -v
# certs/*` before ever running `git add` near this directory; nothing this
# script produces may be committed).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Optional first argument overrides the output directory (default certs/) —
# used by tests/test_tls_auth.py to generate throwaway certs/keys in an
# isolated tmp_path rather than depending on (or polluting) the real certs/.
CERT_DIR="${1:-certs}"
mkdir -p "$CERT_DIR"
rm -f "$CERT_DIR"/*

CERT_CONF="$(mktemp)"
trap 'rm -f "$CERT_CONF"' EXIT
cat > "$CERT_CONF" <<'EOF'
[req]
default_bits = 4096
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn

[dn]
O = pneumonia-fl
CN = localhost

[req_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = superlink
IP.1 = ::1
IP.2 = 127.0.0.1
IP.3 = 0.0.0.0
EOF

echo "Generating local certificate authority..."
openssl genrsa -out "$CERT_DIR/ca.key" 4096
openssl req -new -x509 -key "$CERT_DIR/ca.key" -sha256 \
    -subj "/O=pneumonia-fl-local-ca" -days 365 -out "$CERT_DIR/ca.crt"

echo "Generating SuperLink server certificate..."
openssl genrsa -out "$CERT_DIR/server.key" 4096
openssl req -new -key "$CERT_DIR/server.key" -out "$CERT_DIR/server.csr" -config "$CERT_CONF"
openssl x509 -req -in "$CERT_DIR/server.csr" \
    -CA "$CERT_DIR/ca.crt" -CAkey "$CERT_DIR/ca.key" -CAcreateserial \
    -out "$CERT_DIR/server.pem" -days 365 -sha256 \
    -extfile "$CERT_CONF" -extensions req_ext
rm -f "$CERT_DIR/server.csr" "$CERT_DIR/ca.srl"

echo "Generating per-hospital node-authentication keypairs..."
for hospital in A B C; do
    ssh-keygen -t ecdsa -b 384 -N "" -f "$CERT_DIR/hospital_${hospital}" -C "hospital-${hospital}" -q
done

chmod 600 "$CERT_DIR"/ca.key "$CERT_DIR"/server.key
for hospital in A B C; do
    chmod 600 "$CERT_DIR/hospital_${hospital}"
done

echo
echo "Done. Generated in $CERT_DIR/ (gitignored — never commit):"
ls -la "$CERT_DIR"
