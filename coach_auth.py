"""coach_auth.py — sign in with a Google or Microsoft account.

Why an account at all: the athlete profile, the workout log and the web
dashboard are personal. Signing in gives the coach a verified name/e-mail
for the profile ("identity" facts), lets the dashboard be served beyond
localhost (Docker, the Azure demo) to the right person only, and is the
same identity the mobile apps can use. Nothing about training is sent to
Google or Microsoft — they only tell us who you are.

Conventions followed (deliberately, and tested):
* OAuth 2.0 for Native Apps, RFC 8252 — the system browser (never an
  embedded web view), a loopback redirect on 127.0.0.1 with an ephemeral
  port, exact redirect-URI matching.
* PKCE, RFC 7636 (S256) on every flow — no authorization-code interception.
* OpenID Connect Core — discovery document, `state` against CSRF, `nonce`
  bound into the ID token, and full ID-token validation: RS256 signature
  against the provider's JWKS (pure standard library RSA verification),
  `iss`, `aud`, `exp`, `iat`, `nonce`.
* Secrets stay out of the repo: client IDs come from environment variables
  or the same google_credentials.json the Calendar connector uses; the
  identity file is written with mode 0600 and never logged.
* The dashboard's web flow uses the same code path, plus HttpOnly /
  SameSite=Lax (Secure over https) session cookies signed with HMAC-SHA256
  and an optional e-mail allow-list.

    python coach_auth.py --login google          # browser → signed in
    python coach_auth.py --login microsoft
    python coach_auth.py --whoami
    python coach_auth.py --logout
    python coach_auth.py --selftest              # offline, fake provider

Setup: docs/AUTH.md.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.server
import json
import os
import secrets
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
IDENTITY_FILE = os.environ.get("COACH_IDENTITY_FILE",
                               os.path.join(HERE, "coach_identity.json"))
GOOGLE_CREDS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE",
                                   os.path.join(HERE, "google_credentials.json"))
LOGIN_TIMEOUT_S = 300
CLOCK_SKEW_S = 120

SETUP_HELP = """\
No sign-in client configured. One-time setup (docs/AUTH.md):
  Google:    console.cloud.google.com → Credentials → OAuth client ID →
             "Desktop app" → set COACH_GOOGLE_CLIENT_ID and
             COACH_GOOGLE_CLIENT_SECRET (or reuse google_credentials.json
             from the Calendar connector)
  Microsoft: portal.azure.com → Entra ID → App registrations → New →
             "Mobile and desktop applications" with redirect
             http://localhost, enable "Allow public client flows" →
             set COACH_MICROSOFT_CLIENT_ID (COACH_MICROSOFT_TENANT
             defaults to "common")"""


class AuthError(RuntimeError):
    """Sign-in failed or a token did not validate."""


# ------------------------------------------------------------- providers
@dataclass
class Provider:
    name: str                      # "google" | "microsoft"
    label: str
    client_id: str
    client_secret: str = ""        # Google "Desktop app" clients require one
    discovery_url: str = ""
    scopes: str = "openid email profile"
    extra_auth_params: dict = field(default_factory=dict)
    _config: dict | None = None
    _jwks: dict | None = None

    def config(self) -> dict:
        if self._config is None:
            self._config = _http_json("GET", self.discovery_url)
            for k in ("issuer", "authorization_endpoint", "token_endpoint", "jwks_uri"):
                if k not in self._config:
                    raise AuthError(f"{self.label}: discovery document lacks {k}")
        return self._config

    def jwks(self, refresh: bool = False) -> dict:
        if self._jwks is None or refresh:
            self._jwks = _http_json("GET", self.config()["jwks_uri"])
        return self._jwks

    def issuer_matches(self, iss: str) -> bool:
        exp = self.config()["issuer"]
        if self.name == "microsoft" and "{tenantid}" in exp:
            # the "common" endpoint issues per-tenant issuers
            head = exp.split("{tenantid}")[0]
            return iss.startswith(head)
        return iss == exp or (self.name == "google" and iss == "accounts.google.com")


def _google_client() -> tuple[str, str]:
    cid = os.environ.get("COACH_GOOGLE_CLIENT_ID", "")
    sec = os.environ.get("COACH_GOOGLE_CLIENT_SECRET", "")
    if not cid and os.path.exists(GOOGLE_CREDS_FILE):
        try:
            with open(GOOGLE_CREDS_FILE, encoding="utf-8") as fh:
                raw = json.load(fh)
            c = raw.get("installed") or raw.get("web") or raw
            cid, sec = c.get("client_id", ""), c.get("client_secret", "")
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return cid, sec


def providers(env: dict | None = None) -> dict[str, Provider]:
    """The providers that have a client configured (empty = not set up)."""
    env = os.environ if env is None else env
    out: dict[str, Provider] = {}
    gid, gsec = (env.get("COACH_GOOGLE_CLIENT_ID", ""),
                 env.get("COACH_GOOGLE_CLIENT_SECRET", ""))
    if env is os.environ and not gid:
        gid, gsec = _google_client()
    if gid:
        out["google"] = Provider(
            "google", "Google", gid, gsec,
            "https://accounts.google.com/.well-known/openid-configuration",
            extra_auth_params={"prompt": "select_account"})
    mid = env.get("COACH_MICROSOFT_CLIENT_ID", "")
    if mid:
        tenant = env.get("COACH_MICROSOFT_TENANT", "common")
        out["microsoft"] = Provider(
            "microsoft", "Microsoft", mid, "",
            f"https://login.microsoftonline.com/{tenant}/v2.0/"
            ".well-known/openid-configuration",
            extra_auth_params={"prompt": "select_account"})
    return out


# ----------------------------------------------------------------- http
def _http_json(method: str, url: str, payload: dict | None = None,
               form: bool = False, timeout: float = 20.0) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        if form:
            data = urllib.parse.urlencode(payload).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise AuthError(f"{url}: HTTP {e.code} {body}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise AuthError(f"{url}: {e}") from None


# ----------------------------------------------------------------- PKCE
def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def pkce_pair(verifier: str | None = None) -> tuple[str, str]:
    """(code_verifier, code_challenge) per RFC 7636 §4.1–4.2, S256."""
    verifier = verifier or b64url(secrets.token_bytes(48))     # 64 chars
    challenge = b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


# ------------------------------------------------- ID token verification
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


def rsa_pkcs1v15_sha256_verify(n: int, e: int, message: bytes, signature: bytes) -> bool:
    """RSASSA-PKCS1-v1_5 verify (RFC 8017 §8.2.2) with SHA-256 — standard
    library only: modular exponentiation and a constant-time compare."""
    k = (n.bit_length() + 7) // 8
    if len(signature) != k:
        return False
    s = int.from_bytes(signature, "big")
    if s >= n:
        return False
    em = pow(s, e, n).to_bytes(k, "big")
    t = _SHA256_DIGEST_INFO + hashlib.sha256(message).digest()
    if k < len(t) + 11:
        return False
    expected = b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t
    return hmac.compare_digest(em, expected)


def decode_jwt(token: str) -> tuple[dict, dict, bytes, bytes]:
    """(header, claims, signature, signing_input) — no validation yet."""
    try:
        h, p, s = token.split(".")
        header = json.loads(b64url_decode(h))
        claims = json.loads(b64url_decode(p))
        return header, claims, b64url_decode(s), f"{h}.{p}".encode("ascii")
    except (ValueError, json.JSONDecodeError) as e:
        raise AuthError(f"malformed ID token: {e}") from None


def verify_id_token(token: str, provider: Provider, nonce: str,
                    now: float | None = None) -> dict:
    """OIDC Core §3.1.3.7: signature (RS256 via JWKS), iss, aud, exp, iat,
    nonce. Returns the claims."""
    now = time.time() if now is None else now
    header, claims, sig, signing_input = decode_jwt(token)
    if header.get("alg") != "RS256":
        raise AuthError(f"unsupported ID token alg {header.get('alg')}")
    key = None
    for attempt in (False, True):                 # refresh JWKS once on miss
        keys = provider.jwks(refresh=attempt).get("keys", [])
        key = next((k for k in keys if k.get("kid") == header.get("kid")), None)
        if key or attempt:
            break
    if not key:
        raise AuthError("ID token signed with an unknown key")
    n = int.from_bytes(b64url_decode(key["n"]), "big")
    e = int.from_bytes(b64url_decode(key["e"]), "big")
    if not rsa_pkcs1v15_sha256_verify(n, e, signing_input, sig):
        raise AuthError("ID token signature invalid")
    if not provider.issuer_matches(str(claims.get("iss", ""))):
        raise AuthError(f"ID token issuer {claims.get('iss')} not trusted")
    aud = claims.get("aud")
    if provider.client_id not in (aud if isinstance(aud, list) else [aud]):
        raise AuthError("ID token audience mismatch")
    if float(claims.get("exp", 0)) < now - CLOCK_SKEW_S:
        raise AuthError("ID token expired")
    if float(claims.get("iat", now)) > now + CLOCK_SKEW_S:
        raise AuthError("ID token issued in the future")
    if claims.get("nonce") != nonce:
        raise AuthError("ID token nonce mismatch")
    return claims


# ------------------------------------------------------------- identity
def identity_from_claims(provider: str, claims: dict, tokens: dict) -> dict:
    email = claims.get("email") or claims.get("preferred_username") or ""
    return {
        "provider": provider,
        "sub": str(claims.get("sub", "")),
        "email": email,
        "email_verified": bool(claims.get("email_verified", provider == "microsoft")),
        "name": claims.get("name") or email.split("@")[0],
        "picture": claims.get("picture", ""),
        "signed_in": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expires": time.time() + float(tokens.get("expires_in", 3600)),
        "refresh_token": tokens.get("refresh_token", ""),
    }


def save_identity(identity: dict, path: str = IDENTITY_FILE) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(identity, fh, indent=1)
    try:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    os.replace(tmp, path)


def load_identity(path: str = IDENTITY_FILE) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) and d.get("sub") else None
    except (OSError, json.JSONDecodeError):
        return None


def logout(path: str = IDENTITY_FILE) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def describe(identity: dict | None) -> str:
    if not identity:
        return "Not signed in."
    return (f"Signed in as {identity.get('name')} <{identity.get('email')}> "
            f"via {identity.get('provider', '?').title()} "
            f"(since {identity.get('signed_in', '?')})")


# ----------------------------------------------------- authorization URL
def build_auth_request(provider: Provider, redirect_uri: str) -> tuple[str, dict]:
    """(url, pending) — pending holds state/nonce/verifier for the callback."""
    verifier, challenge = pkce_pair()
    state, nonce = b64url(secrets.token_bytes(24)), b64url(secrets.token_bytes(24))
    params = {
        "client_id": provider.client_id, "redirect_uri": redirect_uri,
        "response_type": "code", "scope": provider.scopes, "state": state,
        "nonce": nonce, "code_challenge": challenge, "code_challenge_method": "S256",
        **provider.extra_auth_params,
    }
    url = provider.config()["authorization_endpoint"] + "?" + urllib.parse.urlencode(params)
    return url, {"state": state, "nonce": nonce, "verifier": verifier,
                 "redirect_uri": redirect_uri, "provider": provider.name,
                 "created": time.time()}


def exchange_code(provider: Provider, pending: dict, code: str, state: str) -> dict:
    """Callback → tokens → validated identity dict."""
    if not hmac.compare_digest(state, pending["state"]):
        raise AuthError("state mismatch (possible CSRF) — sign in again")
    if time.time() - pending["created"] > LOGIN_TIMEOUT_S:
        raise AuthError("sign-in took too long — start again")
    payload = {"grant_type": "authorization_code", "code": code,
               "redirect_uri": pending["redirect_uri"],
               "client_id": provider.client_id, "code_verifier": pending["verifier"]}
    if provider.client_secret:
        payload["client_secret"] = provider.client_secret
    tokens = _http_json("POST", provider.config()["token_endpoint"], payload, form=True)
    if "id_token" not in tokens:
        raise AuthError("token response has no ID token (is 'openid' in scope?)")
    claims = verify_id_token(tokens["id_token"], provider, pending["nonce"])
    return identity_from_claims(provider.name, claims, tokens)


# ------------------------------------------------ native (loopback) login
def login(provider: Provider, open_browser=None, identity_file: str = IDENTITY_FILE,
          timeout_s: float = LOGIN_TIMEOUT_S) -> dict:
    """RFC 8252 loopback flow: ephemeral 127.0.0.1 port, system browser."""
    got: dict = {}
    lock = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            if u.path != "/callback" or not (q.get("code") or q.get("error")):
                self.send_response(404)
                self.end_headers()
                return
            got.update({k: v[0] for k, v in q.items()})
            ok = "code" in got
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write((
                "<h2>✅ Signed in — close this tab and go back to your coach.</h2>"
                if ok else f"<h2>Sign-in cancelled: {got.get('error')}</h2>").encode())
            lock.set()

        def log_message(self, *_a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    srv.timeout = 1
    redirect = f"http://127.0.0.1:{srv.server_address[1]}/callback"
    url, pending = build_auth_request(provider, redirect)
    if open_browser is None:
        import webbrowser
        print(f"Opening your browser to sign in with {provider.label}…\n"
              "If nothing opens, paste this URL yourself:\n" + url)
        open_browser = webbrowser.open
    open_browser(url)
    deadline = time.time() + timeout_s
    try:
        while not lock.is_set() and time.time() < deadline:
            srv.handle_request()
    finally:
        srv.server_close()
    if "code" not in got:
        raise AuthError(got.get("error_description") or got.get("error")
                        or "no authorization code received (timeout or consent denied)")
    identity = exchange_code(provider, pending, got["code"], got.get("state", ""))
    if identity_file:
        save_identity(identity, identity_file)
    return identity


# ----------------------------------------------------- web sessions
class SessionSigner:
    """HMAC-SHA256 signed, expiring session values for the dashboard cookie.
    The secret comes from COACH_SESSION_SECRET or is random per process
    (sessions then end with the process — fine for a personal page)."""

    def __init__(self, secret: str | bytes | None = None, ttl_s: float = 7 * 24 * 3600):
        secret = secret or os.environ.get("COACH_SESSION_SECRET") or secrets.token_hex(32)
        self.key = secret.encode() if isinstance(secret, str) else secret
        self.ttl_s = ttl_s

    def sign(self, payload: dict) -> str:
        body = b64url(json.dumps({**payload, "exp": time.time() + self.ttl_s},
                                 separators=(",", ":")).encode())
        mac = b64url(hmac.new(self.key, body.encode(), hashlib.sha256).digest())
        return f"{body}.{mac}"

    def verify(self, value: str | None) -> dict | None:
        if not value or "." not in value:
            return None
        body, mac = value.rsplit(".", 1)
        good = b64url(hmac.new(self.key, body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(mac, good):
            return None
        try:
            payload = json.loads(b64url_decode(body))
        except (ValueError, json.JSONDecodeError):
            return None
        if float(payload.get("exp", 0)) < time.time():
            return None
        return payload


class WebAuth:
    """Server-side flow for the dashboard: /login → provider → /auth/callback
    → session cookie. `allowed` is a set of e-mails (empty = anyone who can
    sign in with a configured provider)."""

    COOKIE = "coach_session"

    def __init__(self, providers_: dict[str, Provider], allowed: set[str] | None = None,
                 signer: SessionSigner | None = None):
        self.providers = providers_
        self.allowed = {a.lower() for a in (allowed or set()) if a}
        self.signer = signer or SessionSigner()
        self.pending: dict[str, dict] = {}       # state -> pending request
        self._lock = threading.Lock()

    @staticmethod
    def allowed_from_env(env: dict | None = None) -> set[str]:
        env = os.environ if env is None else env
        return {e.strip().lower() for e in env.get("COACH_ALLOWED_EMAILS", "").split(",")
                if e.strip()}

    def start(self, provider_name: str, base_url: str) -> str:
        """URL to redirect the browser to (base_url = scheme://host[:port])."""
        provider = self.providers.get(provider_name)
        if provider is None:
            raise AuthError(f"unknown provider {provider_name}")
        url, pending = build_auth_request(provider, base_url.rstrip("/") + "/auth/callback")
        with self._lock:
            now = time.time()
            self.pending = {s: p for s, p in self.pending.items()
                            if now - p["created"] < LOGIN_TIMEOUT_S}
            self.pending[pending["state"]] = pending
        return url

    def finish(self, query: dict) -> tuple[dict, str]:
        """Callback query → (identity, cookie value). Raises AuthError."""
        state = query.get("state", "")
        with self._lock:
            pending = self.pending.pop(state, None)
        if pending is None:
            raise AuthError("unknown or expired sign-in state — start again")
        if query.get("error"):
            raise AuthError(query.get("error_description") or query["error"])
        identity = exchange_code(self.providers[pending["provider"]], pending,
                                 query.get("code", ""), state)
        email = identity.get("email", "").lower()
        if self.allowed and email not in self.allowed:
            raise AuthError(f"{email} is not on the allow-list for this dashboard")
        cookie = self.signer.sign({"sub": identity["sub"], "email": email,
                                   "name": identity["name"],
                                   "provider": identity["provider"]})
        return identity, cookie

    def session(self, cookie_header: str | None) -> dict | None:
        if not cookie_header:
            return None
        for part in cookie_header.split(";"):
            k, _, v = part.strip().partition("=")
            if k == self.COOKIE:
                return self.signer.verify(v)
        return None

    def cookie_headers(self, value: str | None, secure: bool) -> str:
        """Set-Cookie value; value=None clears the cookie."""
        base = f"{self.COOKIE}={value or ''}; Path=/; HttpOnly; SameSite=Lax"
        if secure:
            base += "; Secure"
        if value is None:
            base += "; Max-Age=0"
        else:
            base += f"; Max-Age={int(self.signer.ttl_s)}"
        return base

    def login_page(self, error: str = "") -> str:
        import html as _html
        buttons = "".join(
            f'<a class="btn" href="/login/{name}">Sign in with {p.label}</a>'
            for name, p in self.providers.items())
        err = f'<p class="err">{_html.escape(error)}</p>' if error else ""
        return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Gym Coach — sign in</title><style>
body{{font:15px/1.5 system-ui,sans-serif;background:#0f1115;color:#e5e7eb;display:flex;
align-items:center;justify-content:center;min-height:100vh;margin:0}}
.card{{background:#171a21;border:1px solid #262b36;border-radius:14px;padding:28px 32px;
max-width:380px;text-align:center}} h1{{font-size:22px;margin:0 0 6px}}
p{{color:#94a3b8}} .btn{{display:block;margin:10px 0;padding:11px;border-radius:10px;
background:#4ade80;color:#0f1115;font-weight:600;text-decoration:none}}
.btn:hover{{background:#86efac}} .err{{color:#f87171}}</style></head><body>
<div class="card"><h1>&#127947; AI Gym Coach</h1><p>This dashboard is private.
Sign in to see your progress.</p>{err}{buttons or
'<p class="err">No sign-in provider configured — see docs/AUTH.md.</p>'}
<p style="font-size:12px">Only your name and e-mail are read. Nothing about your
training is shared with the provider.</p></div></body></html>"""


# --------------------------------------------------------------- chat cmd
def handle_command(text: str) -> str | None:
    """/login google|microsoft, /whoami, /logout — None if not ours."""
    parts = text.strip().split()
    if not parts or parts[0].lower() not in ("/login", "/whoami", "/logout"):
        return None
    cmd = parts[0].lower()
    if cmd == "/whoami":
        return describe(load_identity())
    if cmd == "/logout":
        return "Signed out." if logout() else "You were not signed in."
    avail = providers()
    if not avail:
        return SETUP_HELP
    want = parts[1].lower() if len(parts) > 1 else next(iter(avail))
    if want not in avail:
        return f"usage: /login {'|'.join(avail)}"
    try:
        ident = login(avail[want])
    except AuthError as e:
        return f"Sign-in failed: {e}"
    return describe(ident)


def identity_prompt_block() -> str:
    """One line for the coach's system prompt (data, not instructions)."""
    ident = load_identity()
    if not ident:
        return ""
    import coach_ops
    return ("SIGNED-IN ATHLETE: " + coach_ops.neutralize(
        f"{ident.get('name', '')} ({ident.get('email', '')}, via "
        f"{ident.get('provider', '')})"))


# ---------------------------------------------------------------- selftest
def _test_rsa_key(bits: int = 1024) -> tuple[int, int, int]:
    """Tiny pure-Python RSA keygen for the selftest (n, e, d)."""
    import random
    rng = random.Random(1234)

    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
            if n % p == 0:
                return n == p
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for _ in range(16):
            a = rng.randrange(2, n - 1)
            x = pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def prime(b: int) -> int:
        while True:
            c = rng.getrandbits(b) | (1 << (b - 1)) | 1
            if is_prime(c):
                return c
    e = 65537
    while True:
        p, q = prime(bits // 2), prime(bits // 2)
        phi = (p - 1) * (q - 1)
        if p != q and phi % e:
            return p * q, e, pow(e, -1, phi)


def _sign_jwt(claims: dict, n: int, e: int, d: int, kid: str = "k1", alg="RS256") -> str:
    header = b64url(json.dumps({"alg": alg, "kid": kid, "typ": "JWT"}).encode())
    body = b64url(json.dumps(claims).encode())
    signing_input = f"{header}.{body}".encode()
    k = (n.bit_length() + 7) // 8
    t = _SHA256_DIGEST_INFO + hashlib.sha256(signing_input).digest()
    em = b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t
    sig = pow(int.from_bytes(em, "big"), d, n).to_bytes(k, "big")
    return f"{header}.{body}.{b64url(sig)}"


class _FakeProvider(threading.Thread):
    """Discovery + token + JWKS endpoints of an OIDC provider, in-process."""

    def __init__(self, n: int, e: int, d: int, client_id: str = "test-client"):
        super().__init__(daemon=True)
        self.n, self.e, self.d, self.client_id = n, e, d, client_id
        self.codes: dict[str, str] = {}         # code -> nonce
        self.seen_verifiers: list[str] = []
        self.srv = http.server.HTTPServer(("127.0.0.1", 0), self._handler())
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def _handler(self):
        fp = self

        class H(http.server.BaseHTTPRequestHandler):
            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/.well-known/openid-configuration":
                    self._json({"issuer": fp.base, "authorization_endpoint": fp.base + "/auth",
                                "token_endpoint": fp.base + "/token",
                                "jwks_uri": fp.base + "/jwks"})
                elif self.path == "/jwks":
                    self._json({"keys": [{"kty": "RSA", "kid": "k1", "alg": "RS256",
                                          "n": b64url(fp.n.to_bytes((fp.n.bit_length() + 7) // 8, "big")),
                                          "e": b64url(fp.e.to_bytes(3, "big"))}]})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                ln = int(self.headers.get("Content-Length", 0))
                form = urllib.parse.parse_qs(self.rfile.read(ln).decode())
                code = form.get("code", [""])[0]
                fp.seen_verifiers.append(form.get("code_verifier", [""])[0])
                if code not in fp.codes:
                    return self._json({"error": "invalid_grant"}, 400)
                now = int(time.time())
                tok = _sign_jwt({"iss": fp.base, "aud": fp.client_id, "sub": "user-1",
                                 "email": "ath@example.com", "email_verified": True,
                                 "name": "Ath Lete", "nonce": fp.codes.pop(code),
                                 "iat": now, "exp": now + 600}, fp.n, fp.e, fp.d)
                self._json({"access_token": "at", "id_token": tok, "expires_in": 3600,
                            "token_type": "Bearer"})

            def log_message(self, *_a):
                pass
        return H

    def run(self):
        self.srv.serve_forever()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()

    def provider(self) -> Provider:
        return Provider("google", "Fake", self.client_id, "",
                        self.base + "/.well-known/openid-configuration")


def selftest():
    import tempfile
    print("== coach_auth selftests ==")

    print("1) PKCE matches the RFC 7636 test vector:", end=" ")
    v, c = pkce_pair("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
    assert c == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM", c
    v2, c2 = pkce_pair()
    assert 43 <= len(v2) <= 128 and c2 != c and pkce_pair(v2)[1] == c2
    print("OK")

    print("2) RS256 verify (pure stdlib) accepts a good signature, rejects tampering:", end=" ")
    n, e, d = _test_rsa_key()
    tok = _sign_jwt({"sub": "x"}, n, e, d)
    header, claims, sig, si = decode_jwt(tok)
    assert rsa_pkcs1v15_sha256_verify(n, e, si, sig)
    assert not rsa_pkcs1v15_sha256_verify(n, e, si + b"x", sig)
    assert not rsa_pkcs1v15_sha256_verify(n, e, si, bytes(len(sig)))
    assert not rsa_pkcs1v15_sha256_verify(n, e, si, sig[:-1])
    print("OK")

    fp = _FakeProvider(n, e, d)
    fp.start()
    try:
        prov = fp.provider()
        print("3) ID token validation: iss, aud, exp, iat, nonce, kid, alg:", end=" ")
        now = int(time.time())
        good = {"iss": fp.base, "aud": "test-client", "sub": "u", "nonce": "N",
                "iat": now, "exp": now + 60}
        assert verify_id_token(_sign_jwt(good, n, e, d), prov, "N")["sub"] == "u"
        for bad, why in ((dict(good, iss="https://evil"), "issuer"),
                         (dict(good, aud="other"), "audience"),
                         (dict(good, exp=now - 1000), "expired"),
                         (dict(good, iat=now + 1000), "future"),
                         (dict(good, nonce="M"), "nonce")):
            try:
                verify_id_token(_sign_jwt(bad, n, e, d), prov, "N")
                raise AssertionError(f"accepted bad {why}")
            except AuthError as err:
                assert why in str(err).lower() or why == "future" and "future" in str(err), (why, err)
        try:
            verify_id_token(_sign_jwt(good, n, e, d, kid="zz"), prov, "N")
            raise AssertionError("unknown kid accepted")
        except AuthError as err:
            assert "unknown key" in str(err)
        try:
            verify_id_token(_sign_jwt(good, n, e, d, alg="none"), prov, "N")
            raise AssertionError("alg none accepted")
        except AuthError as err:
            assert "alg" in str(err)
        # "aud" may be a list; Microsoft per-tenant issuers under "common"
        assert verify_id_token(_sign_jwt(dict(good, aud=["a", "test-client"]), n, e, d),
                               prov, "N")
        ms = Provider("microsoft", "Microsoft", "cid", "", "x")
        ms._config = {"issuer": "https://login.microsoftonline.com/{tenantid}/v2.0"}
        assert ms.issuer_matches("https://login.microsoftonline.com/9188-abc/v2.0")
        assert not ms.issuer_matches("https://evil.example/v2.0")
        print("OK")

        print("4) loopback login: state check, PKCE verifier sent, identity saved 0600:", end=" ")
        with tempfile.TemporaryDirectory() as td:
            idf = os.path.join(td, "id.json")
            captured = {}

            def fake_browser(url):
                # act as the user's browser + provider consent: parse the
                # request, mint a code bound to the nonce, hit the redirect
                q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                assert q["code_challenge_method"] == ["S256"] and q["state"] and q["nonce"]
                assert q["redirect_uri"][0].startswith("http://127.0.0.1:")
                assert q["scope"] == ["openid email profile"]
                captured.update({k: v[0] for k, v in q.items()})
                code = secrets.token_hex(8)
                fp.codes[code] = q["nonce"][0]
                cb = q["redirect_uri"][0] + "?" + urllib.parse.urlencode(
                    {"code": code, "state": q["state"][0]})
                threading.Thread(target=lambda: urllib.request.urlopen(cb, timeout=5).read(),
                                 daemon=True).start()
            ident = login(prov, open_browser=fake_browser, identity_file=idf, timeout_s=10)
            assert ident["email"] == "ath@example.com" and ident["name"] == "Ath Lete"
            assert ident["provider"] == "google" and ident["sub"] == "user-1"
            assert pkce_pair(fp.seen_verifiers[-1])[1] == captured["code_challenge"]
            saved = load_identity(idf)
            assert saved and saved["email"] == "ath@example.com"
            if os.name != "nt":
                assert stat.S_IMODE(os.stat(idf).st_mode) == 0o600
            assert "Ath Lete" in describe(saved)
            assert logout(idf) and load_identity(idf) is None

            # wrong state on the callback must be rejected before any token call
            def evil_browser(url):
                q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                code = secrets.token_hex(8)
                fp.codes[code] = q["nonce"][0]
                cb = q["redirect_uri"][0] + "?" + urllib.parse.urlencode(
                    {"code": code, "state": "forged"})
                threading.Thread(target=lambda: urllib.request.urlopen(cb, timeout=5).read(),
                                 daemon=True).start()
            calls = len(fp.seen_verifiers)
            try:
                login(prov, open_browser=evil_browser, identity_file=idf, timeout_s=10)
                raise AssertionError("forged state accepted")
            except AuthError as err:
                assert "state" in str(err)
            assert len(fp.seen_verifiers) == calls, "token endpoint must not be called"
        print("OK")

        print("5) web flow for the dashboard: allow-list, signed cookie, expiry:", end=" ")
        web = WebAuth({"google": prov}, allowed={"ath@example.com"},
                      signer=SessionSigner("unit-test-secret", ttl_s=60))
        url = web.start("google", "https://coach.example")
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        assert q["redirect_uri"] == ["https://coach.example/auth/callback"]
        code = secrets.token_hex(8)
        fp.codes[code] = q["nonce"][0]
        ident, cookie = web.finish({"code": code, "state": q["state"][0]})
        assert ident["email"] == "ath@example.com"
        sess = web.session(f"other=1; {WebAuth.COOKIE}={cookie}")
        assert sess and sess["email"] == "ath@example.com" and sess["provider"] == "google"
        assert web.session(f"{WebAuth.COOKIE}={cookie[:-2]}xx") is None    # tampered
        assert web.session(None) is None
        assert SessionSigner("other-secret").verify(cookie) is None          # wrong key
        old = SessionSigner("unit-test-secret", ttl_s=-1).sign({"sub": "x"})
        assert SessionSigner("unit-test-secret").verify(old) is None         # expired
        hdr = web.cookie_headers(cookie, secure=True)
        assert "HttpOnly" in hdr and "SameSite=Lax" in hdr and "Secure" in hdr
        assert "Max-Age=0" in web.cookie_headers(None, secure=False)
        try:
            web.finish({"code": "c", "state": "nope"})
            raise AssertionError("unknown state accepted")
        except AuthError:
            pass
        strict = WebAuth({"google": prov}, allowed={"someone@else.com"})
        url = strict.start("google", "http://localhost:7788")
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        code = secrets.token_hex(8)
        fp.codes[code] = q["nonce"][0]
        try:
            strict.finish({"code": code, "state": q["state"][0]})
            raise AssertionError("allow-list ignored")
        except AuthError as err:
            assert "allow-list" in str(err)
        page = web.login_page("bad thing <script>")
        assert "Sign in with Fake" in page and "<script>" not in page
        assert WebAuth.allowed_from_env({"COACH_ALLOWED_EMAILS": "A@x.com, b@y.org,"}) == \
            {"a@x.com", "b@y.org"}
        print("OK")
    finally:
        fp.stop()

    print("6) provider configuration from the environment:", end=" ")
    assert providers({}) == {}
    p = providers({"COACH_GOOGLE_CLIENT_ID": "g", "COACH_GOOGLE_CLIENT_SECRET": "s",
                   "COACH_MICROSOFT_CLIENT_ID": "m", "COACH_MICROSOFT_TENANT": "org.example"})
    assert set(p) == {"google", "microsoft"} and p["google"].client_secret == "s"
    assert "org.example/v2.0" in p["microsoft"].discovery_url
    assert p["microsoft"].client_secret == ""                                # public client
    assert handle_command("hello") is None
    assert handle_command("/whoami").startswith(("Not signed in", "Signed in"))
    print("OK")

    print("\nAll coach_auth selftests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--login", choices=("google", "microsoft"))
    ap.add_argument("--whoami", action="store_true")
    ap.add_argument("--logout", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.login:
        avail = providers()
        if args.login not in avail:
            sys.exit(SETUP_HELP)
        try:
            print(describe(login(avail[args.login])))
        except AuthError as e:
            sys.exit(f"Sign-in failed: {e}")
    elif args.whoami:
        print(describe(load_identity()))
    elif args.logout:
        print("Signed out." if logout() else "You were not signed in.")
    else:
        ap.print_help()
        sys.exit(1)
