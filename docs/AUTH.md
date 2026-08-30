# Sign in with Google or Microsoft

`coach_auth.py` lets the athlete sign in with a Google or a Microsoft
account. It is optional: the app works fully without an account. What an
account adds:

| Where | What signing in does |
|---|---|
| Coach chat (`--coach`, `coach_chat.py`) | `/login google` · `/login microsoft` · `/whoami` · `/logout`; the verified name and e-mail land in the athlete profile (`identity` facts) and the coach greets you by name |
| Web dashboard (`coach_dashboard.py --auth`) | the page and `/data.json` require a session; a login page with "Sign in with Google / Microsoft"; an e-mail allow-list so only you (or your athletes) can see the log when it is served beyond localhost — Docker, the Azure demo |
| iOS app | Sign in with Apple, Google or Microsoft on the Account screen (see [IOS.md §7](IOS.md)) |

Nothing about your training is sent to Google or Microsoft: the provider
only tells the app who you are (`openid email profile`).

## 1. The conventions the implementation follows

- **OAuth 2.0 for Native Apps (RFC 8252)** — the system browser (never an
  embedded web view), a loopback redirect `http://127.0.0.1:<random port>/callback`,
  exact redirect-URI matching. The app listens only for the one callback.
- **PKCE (RFC 7636, S256)** on every flow, so an intercepted authorization
  code is useless. The RFC's own test vector is in the selftest.
- **OpenID Connect Core** — provider discovery documents, a random `state`
  (CSRF) checked with a constant-time compare *before* any token call, a
  `nonce` bound into the ID token, and full ID-token validation: RS256
  signature against the provider's JWKS (implemented with the standard
  library — RSASSA-PKCS1-v1_5 per RFC 8017), `iss`, `aud`, `exp`, `iat`,
  `nonce`, `alg` pinned to RS256, unknown `kid` → one JWKS refresh then reject.
- **Secrets out of the repo** — client IDs/secrets come from environment
  variables or `google_credentials.json` (git-ignored); the identity file
  `coach_identity.json` is written with mode 0600 and git-ignored; tokens are
  never printed or traced.
- **Web sessions** — HMAC-SHA256-signed cookie, `HttpOnly; SameSite=Lax`,
  `Secure` behind https (`X-Forwarded-Proto`), 7-day expiry, secret from
  `COACH_SESSION_SECRET` (random per process otherwise), optional
  `COACH_ALLOWED_EMAILS` allow-list, forged/expired/tampered cookies are
  rejected — all covered by `coach_auth.py --selftest` and dashboard
  selftest 9 against an in-process fake OpenID provider.
- **Microsoft public client** — no client secret (public clients cannot keep
  one; PKCE is the protection). Google "Desktop app" clients require a
  secret by Google's design; it is not treated as confidential.

## 2. Setup

### Google

1. [console.cloud.google.com](https://console.cloud.google.com) → the same
   project as the Calendar connector (or a new one) → **APIs & Services →
   OAuth consent screen** (External; add yourself as a test user while in
   testing).
2. **Credentials → Create credentials → OAuth client ID → Desktop app**.
3. Either download the JSON as `google_credentials.json` next to the app
   (the Calendar connector's file works as-is), or set

   ```bash
   export COACH_GOOGLE_CLIENT_ID="…apps.googleusercontent.com"
   export COACH_GOOGLE_CLIENT_SECRET="…"
   ```

   For the **web dashboard** create a second client of type **Web
   application** and add the authorised redirect URI
   `https://<your-host>/auth/callback` (or `http://localhost:7788/auth/callback`).

### Microsoft (Entra ID / personal Microsoft accounts)

1. [portal.azure.com](https://portal.azure.com) → **Microsoft Entra ID → App
   registrations → New registration**; supported account types: *Accounts in
   any organizational directory and personal Microsoft accounts* (or your
   tenant only).
2. **Authentication → Add a platform → Mobile and desktop applications** →
   redirect URI `http://localhost` (Microsoft accepts any loopback port for
   that entry) → enable **Allow public client flows**.
3. For the **web dashboard** add a **Web** platform with
   `https://<your-host>/auth/callback`.
4. Set

   ```bash
   export COACH_MICROSOFT_CLIENT_ID="<Application (client) ID>"
   export COACH_MICROSOFT_TENANT="common"     # or your tenant ID / domain
   ```

### Use it

```bash
python coach_auth.py --login google          # browser opens, sign in, done
python coach_auth.py --whoami
python coach_auth.py --logout

python pose_coach.py --exercise squat --coach   # then type: /login microsoft
python coach_dashboard.py --auth                # login page on http://localhost:7788
COACH_ALLOWED_EMAILS="you@gmail.com" COACH_DASHBOARD_AUTH=1 \
    python coach_dashboard.py --host 0.0.0.0    # shared on the LAN, you only
```

Docker / compose: pass the `COACH_*` variables through the environment
(`docker compose up dashboard` reads them from your shell or a `.env` file —
never commit them). Behind a TLS proxy (the Azure Container App) keep
`X-Forwarded-Proto: https` so the cookie gets the `Secure` flag.

## 3. What is stored where

| File | Content | Protection |
|---|---|---|
| `coach_identity.json` | provider, subject id, e-mail, name, picture URL, sign-in time, refresh token if the provider issued one | mode 0600, git-ignored, `--logout` deletes it |
| `coach_profile.db` | `identity/name`, `identity/email` facts after a chat `/login` | local SQLite, `/forget` removes |
| dashboard cookie | signed `{sub, email, name, provider, exp}` — no tokens | HttpOnly, SameSite=Lax, Secure on https |

Access and refresh tokens are not used for anything after sign-in (the app
calls no Google/Microsoft API with them), so revoking the app in your
account settings has no side effects beyond the next `/login`.

## 4. Threat notes (see SECURITY.md)

- A local process could read `coach_identity.json` — same trust level as
  the workout log and the profile database it sits next to.
- The dashboard without `--auth` is unauthenticated by design for
  `127.0.0.1`; the server prints a warning when bound elsewhere without it.
- Sign-in pages are served by the app itself; the provider's consent page is
  always the real one in the system browser (no embedded views), which is
  what makes phishing-resistant passkeys and hardware keys work.
