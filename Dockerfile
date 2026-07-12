# AI Gym Coach — headless container image.
# Processes exercise videos (rep counting, form analysis, annotated output).
# Webcam + GUI use is supported on Linux hosts via --device/X11 (see README).
FROM python:3.12-slim

# libgl1/libgles2/libegl1/libglib2.0-0 for OpenCV+MediaPipe; espeak-ng for optional TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libgles2 libegl1 libglib2.0-0 libsm6 libxext6 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pose_coach.py coach_chat.py ./
# bake the pose model into the image so containers run offline
RUN python -c "import pose_coach; pose_coach.ensure_model()"

# mount videos + receive logs/annotated output here
VOLUME /data

ENTRYPOINT ["python", "pose_coach.py"]
CMD ["--selftest"]
