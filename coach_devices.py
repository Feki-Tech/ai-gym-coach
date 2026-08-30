"""
Device discovery and selection for the AI Gym Coach.

Which webcam, which microphone, which speaker, which heart-rate strap —
one place that answers "what does this machine have?" and turns a human
spec ("1", "Camo", "/dev/video2", "rtsp://…") into what OpenCV /
sounddevice / bleak expect.

    python coach_devices.py              # cameras + audio devices found here
    python coach_devices.py --ble        # ...plus a 5 s BLE heart-rate scan
    python coach_devices.py --json       # machine-readable
    python coach_devices.py --selftest

Used by:
    python pose_coach.py --list-devices
    python pose_coach.py --camera 1 --coach --mic "Camo"
    python coach_chat.py --hands-free --mic 3

Everything is optional-dependency tolerant: without OpenCV the camera
probe is skipped, without sounddevice the audio section says so, without
bleak the BLE scan says so. Listing devices must never crash.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field

# Windows/MSMF: opening a camera with hardware transforms enabled can take
# 5-20 s per index while probing. The app itself is unaffected (it opens
# one camera once); set only if the user has not chosen otherwise.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

MAX_PROBE_INDEX = 6          # webcams beyond index 5 are vanishingly rare


# ------------------------------------------------------------------ camera
def parse_camera(spec: str | int | None) -> int | str:
    """--camera SPEC -> what cv2.VideoCapture wants.

    None/"" -> 0 (the default webcam); digits -> index; anything else is
    passed through as a path or URL (/dev/video2, rtsp://…, http://…/mjpg).
    """
    if spec is None:
        return 0
    if isinstance(spec, int):
        return spec
    s = str(spec).strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    if s.startswith("-") and s[1:].isdigit():
        raise ValueError(f"camera index must be >= 0, got {s}")
    return s


def describe_camera(cam: int | str) -> str:
    if isinstance(cam, int):
        return f"webcam {cam}"
    if re.match(r"^[a-z][a-z0-9+.-]*://", cam):
        return f"stream {cam}"
    return f"camera {cam}"


@dataclass
class CameraInfo:
    index: int
    name: str = ""
    path: str = ""
    opened: bool | None = None       # None = not probed
    width: int = 0
    height: int = 0
    fps: float = 0.0

    def label(self) -> str:
        bits = [f"[{self.index}]"]
        if self.name:
            bits.append(self.name)
        if self.path:
            bits.append(f"({self.path})")
        if self.opened:
            res = f"{self.width}x{self.height}" if self.width else "opened"
            if self.fps:
                res += f" @ {self.fps:.0f} fps"
            bits.append(f"— {res}")
        elif self.opened is False:
            bits.append("— could not open (busy or no permission?)")
        return " ".join(bits)


def linux_v4l2_devices() -> list[CameraInfo]:
    """/dev/videoN with the driver's name from sysfs (no OpenCV needed)."""
    out = []
    for path in sorted(glob.glob("/dev/video*"),
                       key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        m = re.search(r"(\d+)$", path)
        if not m:
            continue
        idx = int(m.group(1))
        name = ""
        try:
            with open(f"/sys/class/video4linux/video{idx}/name",
                      encoding="utf-8") as fh:
                name = fh.read().strip()
        except OSError:
            pass
        out.append(CameraInfo(index=idx, name=name, path=path))
    return out


def windows_camera_names(timeout: float = 8.0) -> list[str]:
    """Friendly names from PnP (best effort; order usually — not always —
    matches the OpenCV index order)."""
    if sys.platform != "win32":
        return []
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-PnpDevice -Class Camera,Image -Status OK | "
           "Select-Object -ExpandProperty FriendlyName"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
    except Exception:
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _default_opener():
    """() -> callable(index) -> (opened, width, height, fps), or None if
    OpenCV is unavailable."""
    try:
        import cv2
    except ImportError:
        return None
    try:                                  # silence "can't open camera" spam
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass

    def _open(index: int):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                return False, 0, 0, 0.0
            return (True, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                    float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
        finally:
            cap.release()
    return _open


def list_cameras(probe: bool = True, max_index: int = MAX_PROBE_INDEX,
                 opener=None, platform: str | None = None,
                 names: list[str] | None = None) -> list[CameraInfo]:
    """Cameras this machine can see.

    Linux: exactly the /dev/video* nodes (names from sysfs). Elsewhere:
    indices 0..max_index-1, kept only if they open (or if `probe` is off,
    all of them, unverified). `opener`/`platform`/`names` are injectable
    for tests.
    """
    platform = platform or sys.platform
    if platform.startswith("linux"):
        cams = linux_v4l2_devices()
        if not cams:
            return []
        candidates = cams
    else:
        if names is None:
            names = windows_camera_names() if platform == "win32" else []
        candidates = [CameraInfo(index=i,
                                 name=names[i] if i < len(names) else "")
                      for i in range(max_index)]
    if not probe:
        return candidates
    if opener is None:
        opener = _default_opener()
    if opener is None:                    # no OpenCV: report unverified
        return candidates
    out = []
    for cam in candidates:
        try:
            cam.opened, cam.width, cam.height, cam.fps = opener(cam.index)
        except Exception:
            cam.opened = False
        # Linux: keep every node (metadata nodes show as "could not open",
        # which is useful info). Elsewhere a closed index means "no camera".
        if cam.opened or cam.path:
            out.append(cam)
    return out


# ------------------------------------------------------------------- audio
@dataclass
class AudioDevice:
    index: int
    name: str
    kind: str                     # "input" | "output"
    channels: int = 0
    hostapi: str = ""
    default: bool = False

    def label(self) -> str:
        star = " *default*" if self.default else ""
        api = f" [{self.hostapi}]" if self.hostapi else ""
        return f"[{self.index}] {self.name}{api} ({self.channels} ch){star}"


def list_audio_devices(sd=None) -> tuple[list[AudioDevice], list[AudioDevice]]:
    """(inputs, outputs) via sounddevice; ([], []) when it is not installed.
    `sd` is injectable for tests (needs .query_devices/.query_hostapis/
    .default.device)."""
    if sd is None:
        try:
            import sounddevice as sd  # type: ignore
        except Exception:
            return [], []
    try:
        devs = sd.query_devices()
        apis = sd.query_hostapis()
        d_in, d_out = sd.default.device
    except Exception:
        return [], []
    ins, outs = [], []
    for i, d in enumerate(devs):
        api = ""
        try:
            api = apis[d.get("hostapi", -1)]["name"]
        except Exception:
            pass
        if d.get("max_input_channels", 0) > 0:
            ins.append(AudioDevice(i, d["name"], "input",
                                   d["max_input_channels"], api, i == d_in))
        if d.get("max_output_channels", 0) > 0:
            outs.append(AudioDevice(i, d["name"], "output",
                                    d["max_output_channels"], api, i == d_out))
    return ins, outs


def resolve_audio_device(spec: str | int | None, kind: str = "input",
                         devices: list[AudioDevice] | None = None) -> int | None:
    """--mic SPEC -> sounddevice index (None = system default).

    Digits select by index; anything else is a case-insensitive substring
    of the device name (exact match wins, then the default device, then
    the first hit). Raises ValueError with the candidates when nothing
    matches, so the user sees what to type instead.
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    s = str(spec).strip()
    if not s or s.lower() == "default":
        return None
    if devices is None:
        ins, outs = list_audio_devices()
        devices = ins if kind == "input" else outs
    if s.isdigit():
        idx = int(s)
        if devices and idx not in {d.index for d in devices}:
            raise ValueError(
                f"no {kind} device with index {idx}; available:\n  "
                + "\n  ".join(d.label() for d in devices))
        return idx
    if not devices:
        raise ValueError(
            f"cannot look up {kind} device {s!r} by name — install the "
            "voice extras (pip install -r requirements-voice.txt) or "
            "give a numeric index")
    low = s.lower()
    hits = [d for d in devices if low in d.name.lower()]
    if not hits:
        raise ValueError(
            f"no {kind} device matching {s!r}; available:\n  "
            + "\n  ".join(d.label() for d in devices))
    exact = [d for d in hits if d.name.lower() == low]
    default = [d for d in hits if d.default]
    return (exact or default or hits)[0].index


# --------------------------------------------------------------------- BLE
def scan_ble_heart_rate(timeout: float = 5.0) -> list[dict] | None:
    """Nearby GATT heart-rate straps/watches. None = bleak not installed."""
    try:
        import asyncio

        import bleak
    except ImportError:
        return None
    try:
        from coach_sensors import BleHeartRate
        service = BleHeartRate.HR_SERVICE
    except Exception:
        service = "0000180d-0000-1000-8000-00805f9b34fb"

    async def _scan():
        devs = await bleak.BleakScanner.discover(timeout=timeout,
                                                 service_uuids=[service])
        return [{"address": d.address, "name": d.name or ""} for d in devs]
    try:
        return asyncio.run(_scan())
    except Exception as e:
        print(f"(BLE scan failed: {e})")
        return []


# ------------------------------------------------------------------ report
@dataclass
class DeviceReport:
    cameras: list[CameraInfo] = field(default_factory=list)
    camera_note: str = ""
    inputs: list[AudioDevice] = field(default_factory=list)
    outputs: list[AudioDevice] = field(default_factory=list)
    audio_note: str = ""
    ble: list[dict] | None = None
    ble_note: str = ""

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)


def gather(ble: bool = False, probe: bool = True) -> DeviceReport:
    rep = DeviceReport()
    rep.cameras = list_cameras(probe=probe)
    if not rep.cameras:
        if sys.platform.startswith("linux") and not glob.glob("/dev/video*"):
            rep.camera_note = ("no /dev/video* nodes — no webcam is exposed "
                               "to this Linux system")
            if "microsoft" in os.uname().release.lower():
                rep.camera_note += (" (WSL2: attach one with usbipd-win and "
                                    "`sudo modprobe uvcvideo`, or run "
                                    "natively on Windows — docs/WEBCAM.md)")
        else:
            rep.camera_note = "no camera opened at indices 0..%d" % (
                MAX_PROBE_INDEX - 1)
    elif sys.platform == "win32":
        rep.camera_note = ("names come from Windows PnP; the index order "
                           "usually matches — verify with a quick run")
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        rep.audio_note = ("sounddevice not installed — mic/speaker listing "
                          "needs: pip install -r requirements-voice.txt")
    except OSError as e:                  # wheel present, PortAudio missing
        rep.audio_note = (f"sounddevice unusable ({e}) — on Debian/Ubuntu: "
                          "sudo apt-get install libportaudio2")
    else:
        rep.inputs, rep.outputs = list_audio_devices()
        if not rep.inputs and not rep.outputs:
            rep.audio_note = "no audio devices reported by the OS"
    if ble:
        rep.ble = scan_ble_heart_rate()
        if rep.ble is None:
            rep.ble_note = ("bleak not installed — BLE scan needs: "
                            "pip install -r requirements-sensors.txt")
        elif not rep.ble:
            rep.ble_note = "no heart-rate sensor advertising nearby"
    return rep


def format_report(rep: DeviceReport) -> str:
    lines = ["Cameras  (--camera INDEX | PATH | URL)"]
    for c in rep.cameras:
        lines.append("  " + c.label())
    if rep.camera_note:
        lines.append("  " + rep.camera_note)
    lines.append("Microphones  (--mic INDEX | NAME)")
    for d in rep.inputs:
        lines.append("  " + d.label())
    lines.append("Speakers  (used via the OS default)")
    for d in rep.outputs:
        lines.append("  " + d.label())
    if rep.audio_note:
        lines.append("  " + rep.audio_note)
    if rep.ble is not None or rep.ble_note:
        lines.append("Heart-rate sensors  (--sensors ble | ble:ADDRESS)")
        for b in rep.ble or []:
            lines.append(f"  {b['address']}  {b['name']}")
        if rep.ble_note:
            lines.append("  " + rep.ble_note)
    return "\n".join(lines)


def print_report(ble: bool = False, as_json: bool = False) -> None:
    rep = gather(ble=ble)
    print(rep.to_json() if as_json else format_report(rep))


# ---------------------------------------------------------------- selftest
def selftest():
    print("1) parse_camera: index / path / URL / defaults:", end=" ")
    assert parse_camera(None) == 0 and parse_camera("") == 0
    assert parse_camera("2") == 2 and parse_camera(3) == 3
    assert parse_camera(" 1 ") == 1
    assert parse_camera("/dev/video2") == "/dev/video2"
    assert parse_camera("rtsp://cam/live") == "rtsp://cam/live"
    try:
        parse_camera("-1")
        raise AssertionError("negative index accepted")
    except ValueError:
        pass
    assert describe_camera(0) == "webcam 0"
    assert describe_camera("rtsp://x").startswith("stream ")
    assert describe_camera("/dev/video1").startswith("camera ")
    print("OK")

    print("2) list_cameras probes indices, drops closed ones:", end=" ")
    seen = []

    def fake_open(i):
        seen.append(i)
        return (i in (0, 2), 640 if i == 0 else 1280, 480 if i == 0 else 720,
                30.0)
    cams = list_cameras(opener=fake_open, platform="win32",
                        names=["HP HD Camera", "HP IR Camera", "Camo"])
    assert seen == list(range(MAX_PROBE_INDEX)), seen
    assert [c.index for c in cams] == [0, 2], cams
    assert cams[1].name == "Camo" and cams[1].width == 1280
    assert "1280x720" in cams[1].label() and "[2]" in cams[1].label()
    unverified = list_cameras(probe=False, platform="darwin", max_index=3)
    assert [c.index for c in unverified] == [0, 1, 2]
    assert all(c.opened is None for c in unverified)

    def boom(i):
        raise RuntimeError("driver exploded")
    assert list_cameras(opener=boom, platform="darwin") == []
    print("OK")

    print("3) audio devices parsed from a sounddevice-like table:", end=" ")

    class FakeSD:
        class default:
            device = (1, 3)

        @staticmethod
        def query_devices():
            return [
                {"name": "Microphone (Camo)", "hostapi": 0,
                 "max_input_channels": 1, "max_output_channels": 0},
                {"name": "Mikrofonarray (Intel Smart Sound)", "hostapi": 0,
                 "max_input_channels": 2, "max_output_channels": 0},
                {"name": "Headset", "hostapi": 1,
                 "max_input_channels": 1, "max_output_channels": 2},
                {"name": "Lautsprecher (Realtek)", "hostapi": 0,
                 "max_input_channels": 0, "max_output_channels": 2},
            ]

        @staticmethod
        def query_hostapis():
            return [{"name": "MME"}, {"name": "WASAPI"}]
    ins, outs = list_audio_devices(FakeSD)
    assert [d.index for d in ins] == [0, 1, 2], ins
    assert [d.index for d in outs] == [2, 3], outs
    assert ins[1].default and not ins[0].default
    assert outs[1].default and outs[1].hostapi == "MME"
    assert "*default*" in ins[1].label()
    assert list_audio_devices(object()) == ([], [])   # broken backend
    print("OK")

    print("4) resolve_audio_device: index, name, default, errors:", end=" ")
    assert resolve_audio_device(None, devices=ins) is None
    assert resolve_audio_device("", devices=ins) is None
    assert resolve_audio_device("default", devices=ins) is None
    assert resolve_audio_device("2", devices=ins) == 2
    assert resolve_audio_device(0, devices=ins) == 0
    assert resolve_audio_device("camo", devices=ins) == 0
    assert resolve_audio_device("MIKROFON", devices=ins) == 1
    assert resolve_audio_device("Headset", devices=ins) == 2
    # ambiguous substring: the default device wins
    assert resolve_audio_device("m", devices=ins) == 1
    # exact name beats default
    assert resolve_audio_device("microphone (camo)", devices=ins) == 0
    for bad in ("9", "webcam"):
        try:
            resolve_audio_device(bad, devices=ins)
            raise AssertionError(f"{bad!r} resolved")
        except ValueError as e:
            assert "available" in str(e) and "Camo" in str(e), e
    # numeric index is trusted when the backend can't be queried
    assert resolve_audio_device("5", devices=[]) == 5
    try:
        resolve_audio_device("camo", devices=[])
        raise AssertionError("name resolved without a backend")
    except ValueError as e:
        assert "voice extras" in str(e)
    print("OK")

    print("5) report renders and serializes with everything missing:",
          end=" ")
    rep = DeviceReport(camera_note="no camera", audio_note="no audio",
                       ble=None, ble_note="no bleak")
    txt = format_report(rep)
    assert "Cameras" in txt and "no camera" in txt and "no bleak" in txt
    assert json.loads(rep.to_json())["cameras"] == []
    rep2 = DeviceReport(cameras=cams, inputs=ins, outputs=outs,
                        ble=[{"address": "AA:BB", "name": "Polar H10"}])
    txt2 = format_report(rep2)
    assert "Camo" in txt2 and "Polar H10" in txt2 and "Realtek" in txt2
    assert json.loads(rep2.to_json())["cameras"][1]["index"] == 2
    print("OK")

    print("6) live gather() never raises on this machine:", end=" ")
    rep = gather(ble=False)
    assert isinstance(format_report(rep), str)
    json.loads(rep.to_json())
    print(f"OK ({len(rep.cameras)} camera(s), {len(rep.inputs)} mic(s), "
          f"{len(rep.outputs)} speaker(s))")

    print("\nAll coach_devices selftests passed.")


def main():
    ap = argparse.ArgumentParser(
        description="List the cameras, microphones, speakers and BLE "
                    "heart-rate sensors the coach can use.")
    ap.add_argument("--ble", action="store_true",
                    help="also scan for BLE heart-rate sensors (5 s)")
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--no-probe", action="store_true",
                    help="do not open cameras to verify them (faster)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    rep = gather(ble=args.ble, probe=not args.no_probe)
    print(rep.to_json() if args.json else format_report(rep))


if __name__ == "__main__":
    main()
