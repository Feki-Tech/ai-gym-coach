"""Google Calendar for the AI Gym Coach — the coach reads your week and
books training sessions with you ("book me a leg day Tuesday at 6pm").

Setup (one time, ~3 minutes):
1. https://console.cloud.google.com → create a project → APIs & Services →
   enable the "Google Calendar API".
2. OAuth consent screen → External → add your own Gmail as a test user.
3. Credentials → Create credentials → OAuth client ID → **Desktop app** →
   download the JSON → save it as  google_credentials.json  next to this
   file (or point GOOGLE_CREDENTIALS_FILE at it).
4. Run:  python coach_calendar.py --connect   (opens your browser once).

Then start the app as usual — the coach announces "Calendar connected".

Privacy: the only scope requested is calendar.events (read/write events,
nothing else in your Google account). Tokens live in google_token.json on
your machine (git-ignored) and, with the default Ollama backend, your
agenda is only ever shown to the local LLM.

No third-party dependencies — plain stdlib urllib, like the LLM client.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
CREDS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE",
                            os.path.join(HERE, "google_credentials.json"))
TOKEN_FILE = os.environ.get("GOOGLE_TOKEN_FILE",
                            os.path.join(HERE, "google_token.json"))
SCOPE = "https://www.googleapis.com/auth/calendar.events"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
API = "https://www.googleapis.com/calendar/v3"

SETUP_HELP = f"""\
Google Calendar isn't connected yet. One-time setup:
  1. console.cloud.google.com → new project → enable "Google Calendar API"
  2. OAuth consent screen → External → add yourself as a test user
  3. Credentials → OAuth client ID → Desktop app → download JSON
     → save as {CREDS_FILE}
  4. python coach_calendar.py --connect"""


class CalendarError(RuntimeError):
    """Google Calendar unreachable / rejected the request."""


def _http_json(method: str, url: str, payload: dict | None = None,
               token: str | None = None, form: bool = False) -> dict:
    headers = {}
    data = None
    if payload is not None:
        if form:
            data = urllib.parse.urlencode(payload).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode()).get("error", {})
            msg = str(err.get("message", err))[:200]
        except Exception:
            msg = f"HTTP {e.code}"
        raise CalendarError(f"Google Calendar: {msg}") from e
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise CalendarError(f"network problem reaching Google: {e}") from e


def _save_token(path: str, tok: dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(tok, fh, indent=2)


def _fmt_start(iso: str) -> str:
    if "T" not in iso:
        return iso + " (all day)"       # date-only events
    try:
        return datetime.fromisoformat(iso).strftime("%a %d %b %H:%M")
    except ValueError:
        return iso


def _fmt_end(iso: str) -> str:
    if "T" not in iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except ValueError:
        return ""


class CalendarClient:
    """Minimal Google Calendar v3 client with automatic token refresh."""

    def __init__(self, token_file: str = TOKEN_FILE, http=None):
        self.token_file = token_file
        self._http = http or _http_json
        self._lock = threading.Lock()
        with open(token_file, encoding="utf-8") as fh:
            self.tok = json.load(fh)

    def _access_token(self) -> str:
        with self._lock:
            if time.time() > float(self.tok.get("expiry", 0)) - 60:
                data = self._http("POST", self.tok["token_uri"], {
                    "client_id": self.tok["client_id"],
                    "client_secret": self.tok["client_secret"],
                    "refresh_token": self.tok["refresh_token"],
                    "grant_type": "refresh_token"}, form=True)
                self.tok["access_token"] = data["access_token"]
                self.tok["expiry"] = time.time() + float(
                    data.get("expires_in", 3600))
                try:
                    _save_token(self.token_file, self.tok)
                except OSError:
                    pass                       # keep going with the in-memory token
            return self.tok["access_token"]

    def events(self, days: int = 7) -> list[dict]:
        now = datetime.now().astimezone()
        q = urllib.parse.urlencode({
            "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(days=days)).isoformat(),
            "singleEvents": "true", "orderBy": "startTime",
            "maxResults": "50"})
        data = self._http("GET", f"{API}/calendars/primary/events?{q}",
                          token=self._access_token())
        out = []
        for it in data.get("items", []):
            start = it.get("start") or {}
            end = it.get("end") or {}
            out.append({
                "start": start.get("dateTime") or start.get("date", ""),
                "end": end.get("dateTime") or end.get("date", ""),
                "title": it.get("summary", "(no title)")})
        return out

    def agenda(self, days: int = 7) -> str:
        """Human/LLM-readable agenda for the next N days."""
        evs = self.events(days)
        if not evs:
            return (f"No events in the next {days} days — the schedule is "
                    "wide open.")
        lines = []
        for e in evs:
            end = _fmt_end(e["end"])
            lines.append(f"- {_fmt_start(e['start'])}"
                         + (f" to {end}" if end else "")
                         + f": {e['title']}")
        return "\n".join(lines)

    def book(self, title: str, start: str, minutes: int = 60,
             description: str = "") -> str:
        """Create an event; start is local time 'YYYY-MM-DDTHH:MM'.

        Returns a human confirmation like 'Tuesday 14 Jul 18:00-19:00'."""
        try:
            begin = datetime.fromisoformat(str(start))
        except (ValueError, TypeError):
            raise CalendarError(
                "start must look like 2026-07-14T18:00 (local time)")
        if begin.tzinfo is None:
            begin = begin.astimezone()         # interpret as local time
        end = begin + timedelta(minutes=minutes)
        body = {"summary": title,
                "description": description or "Booked by your AI Gym Coach 🏋️",
                "start": {"dateTime": begin.isoformat()},
                "end": {"dateTime": end.isoformat()}}
        self._http("POST", f"{API}/calendars/primary/events", body,
                   token=self._access_token())
        return (begin.strftime("%A %d %b %H:%M") + "–"
                + end.strftime("%H:%M"))


def connect(creds_file: str = CREDS_FILE,
            token_file: str = TOKEN_FILE) -> str:
    """One-time OAuth: browser consent → tokens saved locally."""
    if not os.path.exists(creds_file):
        raise CalendarError(SETUP_HELP)
    with open(creds_file, encoding="utf-8") as fh:
        raw = json.load(fh)
    c = raw.get("installed") or raw.get("web") or raw
    client_id, client_secret = c["client_id"], c["client_secret"]
    token_uri = c.get("token_uri", "https://oauth2.googleapis.com/token")

    import http.server
    import webbrowser
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if "code" not in q and "error" not in q:
                self.send_response(404)
                self.end_headers()
                return                        # favicon etc. — keep waiting
            got["code"] = (q.get("code") or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("<h2>✅ Calendar connected — close this tab "
                             "and go back to your coach.</h2>".encode())

        def log_message(self, *_a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    srv.timeout = 5
    redirect = f"http://127.0.0.1:{srv.server_address[1]}"
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id, "redirect_uri": redirect,
        "response_type": "code", "scope": SCOPE,
        "access_type": "offline", "prompt": "consent"})
    print("Opening your browser to connect Google Calendar…\n"
          "If nothing opens, paste this URL yourself:\n" + url)
    webbrowser.open(url)
    deadline = time.time() + 300
    while "code" not in got and time.time() < deadline:
        srv.handle_request()
    srv.server_close()
    if not got.get("code"):
        raise CalendarError("No authorization code received (5 min timeout "
                            "or consent denied). Try --connect again.")
    data = _http_json("POST", token_uri, {
        "code": got["code"], "client_id": client_id,
        "client_secret": client_secret, "redirect_uri": redirect,
        "grant_type": "authorization_code"}, form=True)
    _save_token(token_file, {
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "expiry": time.time() + float(data.get("expires_in", 3600)),
        "client_id": client_id, "client_secret": client_secret,
        "token_uri": token_uri})
    return token_file


def connect_if_configured(token_file: str = TOKEN_FILE) -> CalendarClient | None:
    """CalendarClient if --connect was done before, else None (no browser)."""
    if not os.path.exists(token_file):
        return None
    try:
        return CalendarClient(token_file)
    except Exception:
        return None


# ------------------------------------------------------------- selftests
def selftest():
    import tempfile
    print("coach_calendar selftests")

    now = datetime.now().astimezone()
    calls: list[tuple] = []

    def fake_http(method, url, payload=None, token=None, form=False):
        calls.append((method, url.split("?")[0], payload, token))
        if "token" in url:
            return {"access_token": "AT2", "expires_in": 3600}
        if method == "GET":
            return {"items": [
                {"summary": "Standup",
                 "start": {"dateTime": (now + timedelta(hours=2)).isoformat()},
                 "end": {"dateTime": (now + timedelta(hours=3)).isoformat()}},
                {"summary": "Trip",
                 "start": {"date": "2099-01-02"},
                 "end": {"date": "2099-01-03"}}]}
        return {"id": "evt1"}

    print("1) expired token refreshes, persists, agenda reads:", end=" ")
    with tempfile.TemporaryDirectory() as td:
        tf = os.path.join(td, "tok.json")
        _save_token(tf, {"access_token": "AT1", "refresh_token": "RT",
                         "expiry": 0, "client_id": "CID",
                         "client_secret": "CS",
                         "token_uri": "https://example.test/token"})
        cal = CalendarClient(tf, http=fake_http)
        agenda = cal.agenda(7)
        assert "Standup" in agenda and "Trip" in agenda, agenda
        assert "all day" in agenda                       # date-only event
        assert calls[0][1] == "https://example.test/token"
        assert calls[1][3] == "AT2"                      # refreshed token used
        with open(tf, encoding="utf-8") as fh:
            assert json.load(fh)["access_token"] == "AT2"    # persisted

        print("ok")
        print("2) booking: local timezone attached, correct length:", end=" ")
        when = cal.book("Leg day", "2026-07-14T18:00", 45)
        method, url, body, _tok = calls[-1]
        assert method == "POST" and url.endswith("/events")
        assert body["summary"] == "Leg day"
        s = datetime.fromisoformat(body["start"]["dateTime"])
        e = datetime.fromisoformat(body["end"]["dateTime"])
        assert s.tzinfo is not None and e.tzinfo is not None
        assert (e - s).total_seconds() == 45 * 60
        assert "18:00" in when and "19" not in when.split("–")[0]
        try:
            cal.book("x", "tomorrow at six")
            raise AssertionError("bad start accepted")
        except CalendarError as err:
            assert "must look like" in str(err)
        print("ok")

        print("3) empty agenda + unconfigured detection:", end=" ")

        def empty_http(method, url, payload=None, token=None, form=False):
            return ({"access_token": "AT", "expires_in": 3600}
                    if "token" in url else {"items": []})
        cal2 = CalendarClient(tf, http=empty_http)
        assert "wide open" in cal2.agenda(3)
        assert connect_if_configured(os.path.join(td, "nope.json")) is None
        assert connect_if_configured(tf) is not None
    print("ok")

    print("\nAll coach_calendar selftests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Google Calendar link for the AI Gym Coach")
    ap.add_argument("--connect", action="store_true",
                    help="one-time browser sign-in, saves tokens locally")
    ap.add_argument("--agenda", type=int, metavar="DAYS", default=0,
                    help="print the next DAYS days of your calendar")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.connect:
        try:
            path = connect()
            print(f"✅ Connected — tokens in {path}. The coach can now see "
                  "and book your training sessions.")
        except CalendarError as e:
            sys.exit(str(e))
    elif args.agenda:
        cal = connect_if_configured()
        if cal is None:
            sys.exit(SETUP_HELP)
        try:
            print(cal.agenda(args.agenda))
        except CalendarError as e:
            sys.exit(str(e))
    else:
        ap.print_help()
