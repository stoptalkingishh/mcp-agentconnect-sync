"""Per-identity ASGI allowlist middleware, shared by every mTLS-fronted app.

Pure ASGI with zero dependencies, so it lives in core (the model manager and
the worker transport both mount it). NOTE: uvicorn does not yet surface the
peer certificate to the ASGI scope, so the identity is read from the ASGI-TLS
extension when present, otherwise from a trusted reverse-proxy header
(``X-Client-Cert-DN`` / ``X-SPIFFE-ID``). With pure uvicorn and no proxy, the
effective identity boundary is CA issuance and this middleware is a
defense-in-depth no-op unless a proxy populates the header.
"""

from __future__ import annotations

from typing import Optional


def _tls_extension_identity(scope) -> Optional[str]:
    """Peer identity from the ASGI-TLS extension ONLY — i.e. the real client
    certificate as surfaced by the ASGI server. This is a trustworthy anchor: it
    cannot be set by the remote client, only by the terminating TLS server."""
    ext = (scope.get("extensions") or {}).get("tls") or {}
    # ASGI-TLS extension (PEP-ish): client cert subject may be exposed here.
    subject = ext.get("client_cert_name") or ext.get("client_cert_subject")
    return str(subject) if subject else None


def _forwarded_header_identity(scope) -> Optional[str]:
    """Peer identity from a reverse-proxy header (``X-Client-Cert-DN`` /
    ``X-SPIFFE-ID``). Trustworthy ONLY if a header-stripping mTLS-terminating
    proxy sits in front — otherwise it is client-settable and MUST NOT be trusted
    as an authorization anchor. Callers gate this behind an explicit opt-in."""
    for name, value in scope.get("headers", []):
        lname = name.decode().lower() if isinstance(name, bytes) else str(name).lower()
        if lname in ("x-client-cert-dn", "x-spiffe-id"):
            return value.decode() if isinstance(value, bytes) else str(value)
    return None


def _peer_identity(scope) -> Optional[str]:
    """Best-effort peer identity from the ASGI-TLS extension or a proxy header.

    NOTE: the header fallback is client-spoofable without a header-stripping
    proxy; :class:`ClientIdentityMiddleware` uses it only as an allowlist
    hardening on top of the CA (and defers when absent). A surface that makes
    identity the SOLE authorization determinant must NOT accept the header
    fallback without an explicit trusted-proxy opt-in — see
    ``runtime.transport.add_pull_routes``."""
    return _tls_extension_identity(scope) or _forwarded_header_identity(scope)


class ClientIdentityMiddleware:
    """ASGI middleware: reject requests whose peer identity is not in ``allowed``.

    Only enforces when an identity can actually be determined (extension or proxy
    header). If none is available it defers to the transport-layer CA check rather
    than blocking every request — see module docstring.
    """

    def __init__(self, app, allowed: set[str]):
        self.app = app
        self.allowed = allowed

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            identity = _peer_identity(scope)
            if identity is not None and identity not in self.allowed:
                await self._forbid(send, f"client identity {identity!r} not allowed")
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _forbid(send, detail: str) -> None:
        body = f'{{"detail":"{detail}"}}'.encode()
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})
