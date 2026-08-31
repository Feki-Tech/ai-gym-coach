# AI Gym Coach — headless container image.
# Processes exercise videos (rep counting, form analysis, annotated output).
# Webcam + GUI use is supported on Linux hosts via --device/X11 (see README).
FROM python:3.12-slim

# libgl1/libgles2/libegl1/libglib2.0-0 for OpenCV+MediaPipe; espeak-ng for optional TTS;
# fonts-dejavu-core so the HUD in annotated videos uses a real font
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgles2 libegl1 libglib2.0-0 libsm6 libxext6 espeak-ng \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# uv: reproducible installs pinned from the committed lock (docs/INFRA.md §3).
COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes -o /tmp/req.txt \
 && uv pip install --system --no-cache -r /tmp/req.txt

COPY pose_coach.py coach_chat.py coach_profile.py coach_calendar.py coach_dashboard.py coach_sensors.py coach_ops.py coach_eval.py coach_devices.py coach_hud.py coach_knowledge.py coach_mcp.py coach_auth.py coach_server.py parity_fixtures.py prop_tests.py ./
# coach eval scenarios (docs/LLMOPS.md) — data/ is otherwise a runtime mount
COPY data/coach_evals.jsonl data/parity_fixtures.json data/exercises.json data/
COPY data/knowledge data/knowledge
# bake the pose model into the image so containers run offline
RUN python -c "import pose_coach; pose_coach.ensure_model()"

# mount videos + receive logs/annotated output here
VOLUME /data

ENTRYPOINT ["python", "pose_coach.py"]
CMD ["--selftest"]
