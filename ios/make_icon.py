"""Generate the app icon (1024x1024 PNG, no alpha) with OpenCV.

Usage:  python ios/make_icon.py
Writes: ios/GymCoach/Resources/Assets.xcassets/AppIcon.appiconset/icon-1024.png
"""
from pathlib import Path

import cv2
import numpy as np

S = 1024
img = np.zeros((S, S, 3), np.uint8)

# vertical gradient: deep navy -> teal-ish navy (BGR)
top = np.array([70, 34, 16], np.float64)
bot = np.array([34, 16, 8], np.float64)
for y in range(S):
    img[y, :] = (top + (bot - top) * (y / S)).astype(np.uint8)

LIME = (80, 220, 120)     # skeleton
ORANGE = (60, 140, 245)   # barbell
WHITE = (240, 240, 240)

def seg(a, b, color, th):
    cv2.line(img, a, b, color, th, cv2.LINE_AA)

def dot(p, r, color):
    cv2.circle(img, p, r, color, -1, cv2.LINE_AA)

# stick figure holding a barbell in a squat (side view)
head = (505, 250)
sho = (475, 360)
hip = (430, 560)
knee = (585, 660)
ankle = (455, 810)
grip = (600, 345)

seg((250, 340), (830, 340), ORANGE, 26)             # bar
cv2.circle(img, (265, 340), 58, ORANGE, 22, cv2.LINE_AA)   # plates
cv2.circle(img, (815, 340), 58, ORANGE, 22, cv2.LINE_AA)

seg(sho, hip, LIME, 42)                              # torso
seg(hip, knee, LIME, 42)                             # thigh
seg(knee, ankle, LIME, 42)                           # shin
seg(ankle, (560, 815), LIME, 30)                     # foot
seg(sho, grip, LIME, 34)                             # arm to bar
dot(head, 62, LIME)                                  # head

for p in (sho, hip, knee, ankle):                    # joint markers
    dot(p, 16, WHITE)

out = Path(__file__).parent / "GymCoach" / "Resources" / "Assets.xcassets" / \
    "AppIcon.appiconset" / "icon-1024.png"
cv2.imwrite(str(out), img)
print(f"wrote {out}")
