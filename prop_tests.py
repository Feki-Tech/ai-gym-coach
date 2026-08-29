"""Property-based tests — invariants under seeded random inputs.

The selftests pin known-good examples; these pin what must hold for EVERY
input: parsers that never throw on garbage, an FSM that cannot count more
reps than the signal crossed lockout, partitions that lose nothing. In the
spirit of the rest of the repo the harness is dependency-free (no
hypothesis): a seeded generator, N cases per property, and a reproducible
failure report (property name + case seed) instead of shrinking.

    python prop_tests.py                 # default: 300 cases per property
    PROP_CASES=2000 python prop_tests.py # deeper CI/nightly run
    PROP_SEED=7 python prop_tests.py     # different deterministic universe
"""
from __future__ import annotations

import json
import math
import os
import random
import string
import sys
import tempfile

import numpy as np

import coach_chat
import coach_sensors
import pose_coach

CASES = int(os.environ.get("PROP_CASES", "300"))
BASE_SEED = int(os.environ.get("PROP_SEED", "0"))

_PROPS: list = []


def prop(fn):
    _PROPS.append(fn)
    return fn


# ------------------------------------------------------------- generators
def rand_angles(rng: random.Random) -> list[float]:
    """Mixed regimes: pure noise, random walk, wave+noise — the FSM must
    behave under all of them."""
    n = rng.randrange(0, 400)
    mode = rng.randrange(3)
    out, x = [], rng.uniform(60, 180)
    for i in range(n):
        if mode == 0:
            x = rng.uniform(0, 200)
        elif mode == 1:
            x = min(200.0, max(0.0, x + rng.uniform(-25, 25)))
        else:
            x = (130 + 45 * math.cos(2 * math.pi * i / rng.uniform(20, 120))
                 + rng.uniform(-8, 8))
        out.append(round(x, 3))
    return out


def rand_text(rng: random.Random, n: int = 300) -> str:
    pool = (string.ascii_letters + string.digits + ' \n\t{}[]":,.!?ÄöüßЩ中文'
            + "ACTION:do{}")
    return "".join(rng.choice(pool) for _ in range(rng.randrange(n)))


# ------------------------------------------------------------- properties
@prop
def rep_counter_bounds(rng):
    ex = rng.choice([e for e in pose_coach.SPECS.values() if e.mode == "reps"])
    angles = rand_angles(rng)
    fps = rng.choice([10, 30, 60])
    c = pose_coach.RepCounter(ex)
    events = []
    for i, a in enumerate(angles):
        ev = c.update(a, i / fps)
        if ev is not None:
            events.append(ev)
    assert c.count == len(events)
    upcross = sum(1 for i in range(1, len(angles))
                  if angles[i - 1] <= ex.lockout_above < angles[i])
    assert c.count <= upcross, (c.count, upcross)
    for ev in events:
        assert ev.duration >= ex.min_rep_s - 1e-9
        assert abs((ev.eccentric_s + ev.concentric_s) - ev.duration) < 1e-6
        assert ev.full_depth == (ev.min_angle < ex.bottom_below)
        assert ev.min_angle <= ex.start_below


@prop
def plank_time_conservation(rng):
    p = pose_coach.PlankTracker()
    t = 0.0
    for _ in range(rng.randrange(0, 300)):
        t += rng.uniform(0, 0.2)
        p.update(rng.uniform(100, 200), t)
    assert 0.0 <= p.best <= p.total + 1e-9
    assert p.total <= t + 1e-9


@prop
def parse_actions_total(rng):
    text = rand_text(rng)
    if rng.random() < 0.5:                    # seed some near-valid lines
        text += '\nACTION: {"do": "' + rand_text(rng, 12) + '"}'
    if rng.random() < 0.3:
        text += '\nACTION: {"broken": '
    clean, actions = coach_chat.parse_actions(text)
    for line in clean.splitlines():
        assert not line.strip().startswith("ACTION:"), line
    for a in actions:
        assert isinstance(a, dict) and a.get("do")


@prop
def gatt_parser_total(rng):
    data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 10)))
    out = coach_sensors.parse_hr_measurement(data)
    assert out is None or (isinstance(out, int) and 0 <= out <= 0xFFFF)


@prop
def split_collected_partition(rng):
    rows = [rng.randrange(1000) for _ in range(rng.randrange(0, 60))]
    train, ev = pose_coach._split_collected(rows)
    assert sorted(train + ev) == sorted(rows)
    assert ev == rows[::5]
    assert train == [r for i, r in enumerate(rows) if i % 5 != 0]


@prop
def collected_reader_total(rng):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.jsonl")
        lines = []
        for _ in range(rng.randrange(0, 12)):
            r = rng.random()
            if r < 0.4:
                lines.append(rand_text(rng, 60))
            elif r < 0.7:
                lines.append(json.dumps({"label": rng.choice(
                    list(pose_coach.ML_CLASSES) + ["nope"]),
                    "x": [0.0] * rng.choice([3, pose_coach.NDIM])}))
            elif r < 0.85:
                lines.append(json.dumps([1, 2, 3]))
            else:                          # right shape, poisoned payload
                lines.append(json.dumps({"label": "squat",
                    "x": ["a"] * pose_coach.NDIM}))
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        rows = pose_coach._read_collected(p)
        for label, x in rows:
            assert label in pose_coach.ML_CLASSES
            assert len(x) == pose_coach.NDIM
            assert all(isinstance(v, (int, float)) for v in x)


@prop
def replay_rows_total(rng):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "r.jsonl")
        with open(p, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(rand_text(rng, 400))
            if rng.random() < 0.5:
                fh.write('\n{"t": 1, "kind": "hr", "value": 88}\n')
        rows = list(coach_sensors.ReplaySource(p, realtime=False)._rows())
        for t, kind, value in rows:
            assert isinstance(t, float) and isinstance(kind, str)
            assert isinstance(value, float)


@prop
def sensor_hub_window_contract(rng):
    hub = coach_sensors.SensorHub(keep_s=rng.uniform(1, 50))
    last_by_kind: dict = {}
    for _ in range(rng.randrange(0, 200)):
        kind = rng.choice(["hr", "imu", "glucose"])
        s = coach_sensors.Sample(rng.uniform(0, 100), kind,
                                 rng.uniform(-1e6, 1e6))
        hub.push(s)
        last_by_kind[kind] = s
    now = rng.uniform(0, 120)
    horizon = rng.uniform(0, 60)
    for kind, latest in last_by_kind.items():
        got = hub.latest(kind)
        assert got == latest              # latest == last pushed, always
        for s in hub.window(kind, horizon, now=now):
            assert s.t >= now - horizon
            assert s.kind == kind


@prop
def rest_advisor_fires_once(rng):
    eff = coach_sensors.EffortModel(hr_max=190.0, hr_rest=60.0)
    adv = coach_sensors.RestAdvisor(eff)
    t = 0.0
    fired_since_set = 0
    for _ in range(rng.randrange(1, 120)):
        t += rng.uniform(0.5, 20)
        r = rng.random()
        if r < 0.15:
            eff.update(coach_sensors.Sample(t, "hr", rng.uniform(60, 190)))
            adv.set_done(now=t)
            fired_since_set = 0
        else:
            hr = None if r < 0.3 else rng.uniform(40, 200)
            if adv.check(hr, now=t):
                fired_since_set += 1
        assert fired_since_set <= 1
        z = eff.zone()
        assert z is None or 1 <= z <= 5


@prop
def history_stats_total(rng):
    def rand_val(depth=0):
        r = rng.random()
        if depth > 2 or r < 0.3:
            return rng.choice([None, True, rng.randrange(-5, 100),
                               rng.uniform(-1, 1), rand_text(rng, 12)])
        if r < 0.6:
            return {rand_text(rng, 8) or "k": rand_val(depth + 1)
                    for _ in range(rng.randrange(3))}
        return [rand_val(depth + 1) for _ in range(rng.randrange(3))]

    log = [rand_val() for _ in range(rng.randrange(6))]
    if rng.random() < 0.5:                    # mix in plausible sessions
        log.append({"started": "2026-08-01 10:00:00", "exercise": "squat",
                    "summary": {"reps": 5, "avg_score": 80.0,
                                "fault_counts": {"shallow": rng.randrange(3)}}})
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "log.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(log, fh)
        stats = coach_chat.history_stats(
            p, exercise=rng.choice([None, "squat", rand_text(rng, 6)]),
            days=rng.choice([None, 1, 30, 100000]))
        agg = stats["aggregate"]
        assert agg["sessions"] == len(stats["sessions"])
        assert agg["total_reps"] >= 0
        _, fb = coach_chat.execute_history_action(
            p, {"do": "history_query", "days": rng.choice(
                [None, "junk", -5, 40, 10**9])})
        assert fb is None or isinstance(fb, str)


@prop
def window_features_finite(rng):
    frames = [{k: rng.uniform(-10, 400) for k in pose_coach.FEAT_KEYS}
              for _ in range(rng.randrange(1, 90))]
    x = pose_coach.window_features(frames)
    assert x.shape == (pose_coach.NDIM,)
    assert np.isfinite(x).all()


@prop
def split_sentences_total(rng):
    sents, rest = coach_chat.split_sentences(rand_text(rng))
    for s in sents:
        assert s.strip()
    assert isinstance(rest, str)


# ---------------------------------------------------------------- harness
def main() -> int:
    failures = 0
    for fn in _PROPS:
        bad = None
        for i in range(CASES):
            seed = hash((BASE_SEED, fn.__name__, i)) & 0xFFFFFFFF
            try:
                fn(random.Random(seed))
            except Exception as e:              # noqa: BLE001 — report, don't die
                bad = (i, seed, e)
                break
        if bad:
            failures += 1
            print(f"FAIL {fn.__name__}: case {bad[0]} (seed {bad[1]}): "
                  f"{type(bad[2]).__name__}: {bad[2]}")
        else:
            print(f"ok   {fn.__name__} ({CASES} cases)")
    if failures:
        print(f"\n{failures} propert{'y' if failures == 1 else 'ies'} failed "
              f"(reproduce: PROP_SEED={BASE_SEED} and the seed above).")
        return 1
    print(f"\nAll {len(_PROPS)} properties held over {CASES} cases each.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
