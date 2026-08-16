"""Sensor fusion for AI Gym Coach — heart rate, IMU, and whatever's next.

Design: docs/SENSORS.md. The camera can't see effort, recovery or load;
this module adds the sensors that can. Everything is optional and degrades
to exactly the camera-only behaviour when absent (the hub just answers
"unknown").

    python coach_sensors.py --demo       # simulated set + recovery, printed
    python coach_sensors.py --selftest   # deterministic, runs in CI

Wire into a workout with pose_coach.py --sensors SPEC where SPEC is
    sim            deterministic simulated HR+IMU (no hardware)
    replay:FILE    JSONL replay: {"t": 1.0, "kind": "hr", "value": 91}
    udp:PORT       one JSON object per datagram (phone streaming apps)
    ble            a real BLE heart-rate strap (pip install -r
                   requirements-sensors.txt; standard GATT HRS, so any
                   Polar/Garmin/Wahoo strap or watch in broadcast mode)

Local-first: samples live in memory, summaries land in the local workout
log, nothing is ever uploaded (docs/INFRA.md §2).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque, namedtuple

# One record for everything: host-monotonic arrival time, a kind tag
# ("hr", "imu", ...), and a float value (bpm; acceleration magnitude m/s²).
Sample = namedtuple("Sample", "t kind value")

KIND_HR = "hr"
KIND_IMU = "imu"


# ------------------------------------------------------------------ sources
class SensorSource:
    """A sensor stream. start() may spawn a thread; push via callback."""

    name = "source"

    def start(self, push) -> None:      # push: callable(Sample) -> None
        raise NotImplementedError

    def stop(self) -> None:
        pass


class SimulatedSession(SensorSource):
    """Deterministic HR + IMU generator — the synth_frames of sensors.

    Models rest (hr_rest), a climb while "working" and exponential recovery
    after; IMU emits an acceleration burst per simulated rep. time_fn is
    injectable so tests run instantly on a fake clock."""

    name = "sim"

    def __init__(self, hr_rest: float = 62.0, hr_peak: float = 158.0,
                 seed: int = 0, rate_hz: float = 4.0, time_fn=time.monotonic):
        self.hr_rest, self.hr_peak = hr_rest, hr_peak
        self.rate_hz, self.time_fn = rate_hz, time_fn
        self.seed = seed
        self.working = False            # flipped by the demo / the app
        self._hr = hr_rest
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self, dt: float, t: float) -> list[Sample]:
        """Advance the model by dt seconds and emit samples (pure, testable)."""
        target = self.hr_peak if self.working else self.hr_rest
        # first-order approach: fast climb (~15 s), slower recovery (~40 s)
        tau = 15.0 if self.working else 40.0
        self._hr += (target - self._hr) * (1 - math.exp(-dt / tau))
        wobble = math.sin(t * 1.3 + self.seed) * 1.5
        out = [Sample(t, KIND_HR, round(self._hr + wobble, 1))]
        if self.working:                # one accel burst per ~3 s rep
            phase = (t % 3.0) / 3.0
            accel = 9.81 + 6.0 * math.exp(-((phase - 0.35) ** 2) / 0.01)
        else:
            accel = 9.81 + math.sin(t * 5 + self.seed) * 0.05
        out.append(Sample(t, KIND_IMU, round(accel, 3)))
        return out

    def start(self, push):
        def _loop():
            prev = self.time_fn()
            while not self._stop.is_set():
                time.sleep(1.0 / self.rate_hz)
                now = self.time_fn()
                for s in self.sample_once(now - prev, now):
                    push(s)
                prev = now
        self._stop.clear()
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()


class ReplaySource(SensorSource):
    """Replay a JSONL recording in real time (or instantly for tests)."""

    name = "replay"

    def __init__(self, path: str, realtime: bool = True):
        self.path, self.realtime = path, realtime
        self._stop = threading.Event()

    def _rows(self):
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    yield (float(row["t"]), str(row["kind"]),
                           float(row["value"]))
                except (json.JSONDecodeError, KeyError, TypeError,
                        ValueError):
                    continue            # bad lines skipped, like --collect

    def start(self, push):
        def _loop():
            t0, wall0 = None, time.monotonic()
            for t, kind, value in self._rows():
                if self._stop.is_set():
                    return
                if self.realtime:
                    if t0 is None:
                        t0 = t
                    delay = (t - t0) - (time.monotonic() - wall0)
                    if delay > 0:
                        time.sleep(min(delay, 5.0))
                push(Sample(time.monotonic(), kind, value))
        self._stop.clear()
        threading.Thread(target=_loop, daemon=True).start()

    def stop(self):
        self._stop.set()


class UdpJsonSource(SensorSource):
    """One JSON object per UDP datagram: {"kind": "hr", "value": 128}.

    Lets any phone sensor-streaming app (or a 5-line script) feed the hub —
    no pairing, no dependency. Binds localhost by default."""

    name = "udp"

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.host, self.port = host, port
        self._sock: socket.socket | None = None
        self._stop = threading.Event()

    def start(self, push):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)
        self._sock.bind((self.host, self.port))

        def _loop():
            while not self._stop.is_set():
                try:
                    data, _ = self._sock.recvfrom(2048)
                    row = json.loads(data.decode("utf-8", "replace"))
                    push(Sample(time.monotonic(), str(row["kind"]),
                                float(row["value"])))
                except socket.timeout:
                    continue
                except Exception:
                    continue            # malformed datagrams never kill us
        self._stop.clear()
        threading.Thread(target=_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass


def parse_hr_measurement(data: bytes) -> int | None:
    """GATT Heart Rate Measurement (0x2A37): flags bit0 picks uint8/uint16.

    Pure and spec-tested — the only part of the BLE path that has logic."""
    if not data:
        return None
    if data[0] & 0x01:
        return int.from_bytes(data[1:3], "little") if len(data) >= 3 else None
    return data[1] if len(data) >= 2 else None


class BleHeartRate(SensorSource):
    """Real chest strap / broadcasting watch via the standard GATT HRS.

    Optional dependency (bleak); import errors surface as a friendly
    message, never a crash — the session just runs camera-only."""

    name = "ble"
    HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"
    HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"

    def __init__(self, address: str | None = None):
        self.address = address
        self._stop = threading.Event()

    def start(self, push):
        try:
            import asyncio

            import bleak
        except ImportError:
            print("BLE needs the sensor extras: "
                  "pip install -r requirements-sensors.txt")
            return

        async def _run():
            addr = self.address
            if not addr:
                print("Scanning for a heart-rate sensor (5 s)...")
                devices = await bleak.BleakScanner.discover(
                    timeout=5.0, service_uuids=[self.HR_SERVICE])
                if not devices:
                    print("No BLE heart-rate sensor found — is the strap "
                          "worn and awake?")
                    return
                addr = devices[0].address
                print(f"Using {devices[0].name or addr}")
            async with bleak.BleakClient(addr) as client:
                def _cb(_h, data: bytearray):
                    bpm = parse_hr_measurement(bytes(data))
                    if bpm:
                        push(Sample(time.monotonic(), KIND_HR, float(bpm)))
                await client.start_notify(self.HR_MEASUREMENT, _cb)
                while not self._stop.is_set():
                    await asyncio.sleep(0.5)
                await client.stop_notify(self.HR_MEASUREMENT)

        def _loop():
            import asyncio
            try:
                asyncio.run(_run())
            except Exception as e:
                print(f"(BLE heart rate stopped: {e})")
        self._stop.clear()
        threading.Thread(target=_loop, daemon=True).start()

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------- hub
class SensorHub:
    """Dumb on purpose: thread-safe per-kind ring buffers, time-windowed
    reads, zero interpretation (docs/SENSORS.md §2)."""

    def __init__(self, keep_s: float = 300.0, maxlen: int = 4096):
        self.keep_s = keep_s
        self._buf: dict[str, deque[Sample]] = {}
        self._lock = threading.Lock()
        self._sources: list[SensorSource] = []

    def add(self, source: SensorSource) -> "SensorHub":
        self._sources.append(source)
        return self

    def start(self):
        for s in self._sources:
            s.start(self.push)

    def stop(self):
        for s in self._sources:
            s.stop()

    def push(self, sample: Sample):
        with self._lock:
            buf = self._buf.setdefault(sample.kind, deque(maxlen=4096))
            buf.append(sample)
            cutoff = sample.t - self.keep_s
            while buf and buf[0].t < cutoff:
                buf.popleft()

    def latest(self, kind: str, max_age_s: float | None = None,
               now: float | None = None) -> Sample | None:
        with self._lock:
            buf = self._buf.get(kind)
            if not buf:
                return None
            s = buf[-1]
        if max_age_s is not None:
            now = time.monotonic() if now is None else now
            if now - s.t > max_age_s:
                return None             # stale = unknown, not "last value"
        return s

    def window(self, kind: str, seconds: float,
               now: float | None = None) -> list[Sample]:
        now = time.monotonic() if now is None else now
        with self._lock:
            buf = self._buf.get(kind, ())
            return [s for s in buf if s.t >= now - seconds]

    def status(self) -> dict:
        with self._lock:
            return {k: len(v) for k, v in self._buf.items()}


# ------------------------------------------------------------------ fusion
class EffortModel:
    """HR → training zone and per-set strain. hr_max/hr_rest from the
    athlete profile when known; the age formula is a tagged estimate."""

    def __init__(self, hr_max: float | None = None,
                 hr_rest: float | None = None, age: int | None = None):
        self.estimated = hr_max is None
        self.hr_max = hr_max or (220.0 - (age or 30))
        self.hr_rest = hr_rest or 60.0
        self.peak = 0.0                 # rolling per-set peak
        self.current: float | None = None

    def update(self, sample: Sample | None):
        if sample is None or sample.kind != KIND_HR:
            return
        self.current = sample.value
        self.peak = max(self.peak, sample.value)

    def zone(self, hr: float | None = None) -> int | None:
        hr = self.current if hr is None else hr
        if hr is None:
            return None
        pct = hr / self.hr_max
        for z, top in enumerate((0.6, 0.7, 0.8, 0.9), start=1):
            if pct < top:
                return z
        return 5

    def new_set(self):
        self.peak = self.current or 0.0

    def snapshot(self) -> dict:
        """live_state block for the LLM coach; absent keys mean unknown."""
        if self.current is None:
            return {}
        return {"heart_rate": round(self.current),
                "hr_zone": self.zone(),
                "hr_peak_set": round(self.peak),
                "hr_max" + ("_estimated" if self.estimated else ""):
                    round(self.hr_max)}


class RestAdvisor:
    """Heart-rate-recovery rest: ready when HR drops below
    hr_rest + frac × (set_peak − hr_rest); a hard cap keeps a flaky strap
    from ever blocking training. Pure logic — feed it samples."""

    def __init__(self, effort: EffortModel, frac: float = 0.35,
                 cap_s: float = 240.0, min_s: float = 30.0):
        self.effort = effort
        self.frac, self.cap_s, self.min_s = frac, cap_s, min_s
        self.pending = False
        self._t_set_done: float | None = None
        self._set_peak = 0.0

    def set_done(self, now: float | None = None):
        self._t_set_done = time.monotonic() if now is None else now
        self._set_peak = max(self.effort.peak, self.effort.hr_rest + 1.0)
        self.pending = True

    def threshold(self) -> float:
        return (self.effort.hr_rest
                + self.frac * (self._set_peak - self.effort.hr_rest))

    def check(self, hr: float | None, now: float | None = None) -> str | None:
        """Returns a spoken-ready message ONCE when recovery is reached."""
        if not self.pending or self._t_set_done is None:
            return None
        now = time.monotonic() if now is None else now
        elapsed = now - self._t_set_done
        if elapsed < self.min_s:
            return None
        if hr is not None and hr <= self.threshold():
            self.pending = False
            return (f"Heart rate is back down to {round(hr)} — "
                    "ready when you are.")
        if elapsed >= self.cap_s:
            self.pending = False
            return "Long enough — let's get back to it."
        return None


class VelocityFuser:
    """Per-rep fusion of vision velocity with an IMU window (PoC scope:
    agreement surface, not silent averaging — docs/SENSORS.md §2)."""

    GRAVITY = 9.81

    @staticmethod
    def imu_rep_energy(window: list[Sample]) -> float:
        """Mean |accel − g| over a window — crude 'movement energy'."""
        vals = [abs(s.value - VelocityFuser.GRAVITY) for s in window
                if s.kind == KIND_IMU]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    @classmethod
    def fuse(cls, vision_vel_deg_s: float | None,
             imu_window: list[Sample]) -> dict:
        out: dict = {}
        if vision_vel_deg_s is not None:
            out["vision_vel_deg_s"] = round(vision_vel_deg_s, 1)
        energy = cls.imu_rep_energy(imu_window)
        if imu_window:
            out["imu_energy"] = energy
            out["imu_moving"] = energy > 0.4
        if vision_vel_deg_s is not None and imu_window:
            # disagreement is a signal (loose strap / occlusion), not noise
            out["agree"] = (energy > 0.4) == (vision_vel_deg_s > 20.0)
        return out


# ------------------------------------------------------------------- wiring
def hub_from_spec(spec: str) -> SensorHub | None:
    """--sensors SPEC -> configured hub (sim | replay:FILE | udp:PORT | ble
    | ble:ADDRESS). Unknown/broken specs return None with a message —
    the workout must never die because of a sensor flag."""
    try:
        hub = SensorHub()
        kind, _, arg = spec.partition(":")
        if kind == "sim":
            src: SensorSource = SimulatedSession()
            src.working = True          # workouts are work by default
        elif kind == "replay":
            if not os.path.exists(arg):
                print(f"--sensors replay: no such file {arg}")
                return None
            src = ReplaySource(arg)
        elif kind == "udp":
            src = UdpJsonSource(int(arg or "9999"))
        elif kind == "ble":
            src = BleHeartRate(arg or None)
        else:
            print(f"--sensors: unknown spec {spec!r} "
                  "(use sim | replay:FILE | udp:PORT | ble)")
            return None
        hub.add(src)
        hub.start()
        return hub
    except Exception as e:
        print(f"--sensors {spec}: could not start ({e}) — continuing "
              "camera-only")
        return None


def demo():
    """A simulated set + recovery, fast-forwarded: shows zones climbing,
    the set peak, and the rest advisor firing on actual recovery."""
    clock = [0.0]
    sim = SimulatedSession(time_fn=lambda: clock[0])
    hub = SensorHub()
    effort = EffortModel(hr_max=190.0, hr_rest=60.0)
    advisor = RestAdvisor(effort)

    def advance(seconds: float, working: bool, label: str):
        sim.working = working
        end = clock[0] + seconds
        while clock[0] < end:
            clock[0] += 0.25
            for s in sim.sample_once(0.25, clock[0]):
                hub.push(s)
            effort.update(hub.latest(KIND_HR, now=clock[0]))
        hr = hub.latest(KIND_HR, now=clock[0])
        print(f"t={clock[0]:5.0f}s  {label:12s} hr={hr.value:5.1f} "
              f"zone={effort.zone()}  imu_energy="
              f"{VelocityFuser.imu_rep_energy(hub.window(KIND_IMU, 3.0, now=clock[0]))}")

    print("Simulated session (fast-forward):")
    advance(10, False, "warm-up")
    effort.new_set()
    advance(45, True, "set 1")
    advisor.set_done(now=clock[0])
    print(f"          set done — recovery threshold "
          f"{advisor.threshold():.0f} bpm")
    while advisor.pending:
        advance(10, False, "resting")
        msg = advisor.check(hub.latest(KIND_HR, now=clock[0]).value,
                            now=clock[0])
        if msg:
            print(f"          🏋️  Coach: {msg}")
    print("Demo done — same pipeline the app uses with --sensors sim.")


# ----------------------------------------------------------------- selftest
def selftest():
    print("1) GATT heart-rate parser (spec vectors):", end=" ")
    assert parse_hr_measurement(bytes([0x00, 72])) == 72          # uint8
    assert parse_hr_measurement(bytes([0x01, 0x2C, 0x01])) == 300  # uint16
    assert parse_hr_measurement(bytes([0x16, 95, 0x10, 0x02])) == 95
    assert parse_hr_measurement(b"") is None
    assert parse_hr_measurement(bytes([0x01, 0x2C])) is None      # truncated
    print("ok")

    print("2) hub: ring buffers, windows, staleness:", end=" ")
    hub = SensorHub(keep_s=10.0)
    for i in range(20):
        hub.push(Sample(float(i), KIND_HR, 60.0 + i))
    assert hub.latest(KIND_HR).value == 79.0
    assert len(hub.window(KIND_HR, 5.0, now=19.0)) == 6           # 14..19
    assert hub.latest(KIND_HR, max_age_s=2.0, now=30.0) is None   # stale
    assert hub.latest(KIND_IMU) is None                           # unknown kind
    assert min(s.t for s in hub.window(KIND_HR, 999, now=19.0)) >= 9.0
    print("ok")

    print("3) simulated session is deterministic and physiological:", end=" ")
    a, b = SimulatedSession(seed=3), SimulatedSession(seed=3)
    sa = [s for t in range(60) for s in a.sample_once(1.0, float(t))]
    sb = [s for t in range(60) for s in b.sample_once(1.0, float(t))]
    assert sa == sb                                              # seeded
    a.working = True
    rest_hr = sa[-2].value
    work = [s for t in range(60, 120) for s in a.sample_once(1.0, float(t))]
    work_hr = [s.value for s in work if s.kind == KIND_HR]
    assert work_hr[-1] > rest_hr + 40, (rest_hr, work_hr[-1])    # climbs
    bursts = [s.value for s in work if s.kind == KIND_IMU]
    assert max(bursts) > 12.0 and min(bursts) < 10.5             # rep bursts
    print("ok")

    print("4) effort zones + set peak:", end=" ")
    eff = EffortModel(hr_max=190.0, hr_rest=60.0)
    assert eff.zone(100.0) == 1 and eff.zone(120.0) == 2
    assert eff.zone(140.0) == 3 and eff.zone(160.0) == 4
    assert eff.zone(180.0) == 5 and eff.zone() is None
    eff.update(Sample(0.0, KIND_HR, 150.0))
    eff.update(Sample(1.0, KIND_HR, 130.0))
    assert eff.peak == 150.0 and eff.snapshot()["heart_rate"] == 130
    assert eff.snapshot()["hr_zone"] == 2          # 130/190 = 68 % of max
    est = EffortModel(age=40)
    assert est.estimated and "hr_max_estimated" not in eff.snapshot()
    eff2 = EffortModel(age=40)
    eff2.update(Sample(0.0, KIND_HR, 120.0))
    assert eff2.snapshot()["hr_max_estimated"] == 180
    print("ok")

    print("5) rest advisor: recovery-based, capped, fires once:", end=" ")
    eff = EffortModel(hr_max=190.0, hr_rest=60.0)
    eff.update(Sample(0.0, KIND_HR, 160.0))                     # set peak
    adv = RestAdvisor(eff, cap_s=240.0, min_s=30.0)
    adv.set_done(now=100.0)
    assert abs(adv.threshold() - 95.0) < 1e-6                   # 60+0.35*100
    assert adv.check(90.0, now=110.0) is None                   # min_s gate
    assert adv.check(120.0, now=140.0) is None                  # not recovered
    msg = adv.check(94.0, now=150.0)
    assert msg and "94" in msg
    assert adv.check(80.0, now=160.0) is None                   # fires once
    adv.set_done(now=200.0)
    assert adv.check(None, now=250.0) is None                   # no data yet
    cap = adv.check(None, now=200.0 + 240.0)
    assert cap and not adv.pending                              # capped
    print("ok")

    print("6) velocity fusion surfaces agreement:", end=" ")
    moving = [Sample(float(i) / 10, KIND_IMU, 9.81 + 3.0) for i in range(10)]
    still = [Sample(float(i) / 10, KIND_IMU, 9.81) for i in range(10)]
    f = VelocityFuser.fuse(90.0, moving)
    assert f["agree"] and f["imu_moving"] and f["vision_vel_deg_s"] == 90.0
    assert not VelocityFuser.fuse(90.0, still)["agree"]         # mismatch!
    assert VelocityFuser.fuse(None, []) == {}                   # no sensors
    assert "agree" not in VelocityFuser.fuse(None, moving)
    print("ok")

    print("7) replay + udp sources feed the hub:", end=" ")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "rec.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write('{"t": 0.0, "kind": "hr", "value": 88}\n')
            fh.write("not json\n")
            fh.write('{"t": 0.1, "kind": "imu", "value": 11.2}\n')
        hub = SensorHub()
        src = ReplaySource(p, realtime=False)
        src.start(hub.push)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(hub.status()) < 2:
            time.sleep(0.01)
        assert hub.latest(KIND_HR).value == 88.0
        assert hub.latest(KIND_IMU).value == 11.2
    hub2 = SensorHub()
    udp = UdpJsonSource(0)              # pick a free port
    udp.start(hub2.push)
    port = udp._sock.getsockname()[1]
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    out.sendto(b'{"kind": "hr", "value": 131}', ("127.0.0.1", port))
    out.sendto(b"garbage", ("127.0.0.1", port))
    out.sendto(b'{"kind": "hr", "value": 132}', ("127.0.0.1", port))
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and (
            hub2.latest(KIND_HR) is None
            or hub2.latest(KIND_HR).value != 132.0):
        time.sleep(0.01)
    assert hub2.latest(KIND_HR).value == 132.0
    udp.stop()
    out.close()
    print("ok")

    print("8) hub_from_spec never kills the workout:", end=" ")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert hub_from_spec("nonsense") is None
        assert hub_from_spec("replay:/no/such/file.jsonl") is None
        hub = hub_from_spec("sim")
    assert hub is not None
    hub.stop()
    print("ok")

    print("\nAll coach_sensors selftests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sensor fusion PoC for AI Gym Coach (docs/SENSORS.md)")
    ap.add_argument("--demo", action="store_true",
                    help="fast-forward a simulated set + recovery")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.demo:
        demo()
    else:
        ap.print_help()
        sys.exit(1)
