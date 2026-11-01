# DeepFER live: real-time emotion recognition from the webcam in an OpenCV window.
# Run from the repo folder:  python app/webcam.py      (press q or Esc to quit)
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import deepfer                                      # noqa: E402
from deepfer import CLASSES                         # noqa: E402

COLORS = {"angry": (50, 50, 220), "disgust": (40, 150, 40), "fear": (180, 60, 150), "happy": (0, 190, 240),
          "neutral": (150, 150, 150), "sad": (220, 110, 40), "surprise": (20, 130, 250)}      # BGR for OpenCV

ap = argparse.ArgumentParser(description="Live facial emotion recognition")
ap.add_argument("--camera", default="0", help="camera number, or a video file path")
ap.add_argument("--smooth", type=float, default=0.6, help="0 = no smoothing, 0.9 = very smooth (slower to react)")
ap.add_argument("--gpu", action="store_true", help="run the model on the GPU (default: CPU, fast enough for a few faces)")
ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = run until q)")
ap.add_argument("--no-window", action="store_true", help="don't open a window, only print the speed (for testing)")
args = ap.parse_args()

predictor = deepfer.EmotionPredictor(ROOT / "models", device=deepfer.get_device() if args.gpu else "cpu")
finder = deepfer.FaceFinder()
cap = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
if not cap.isOpened():
    sys.exit("Could not open the camera. On a Mac, allow camera access for the app you run this from "
             "(System Settings > Privacy & Security > Camera), then try again.")

tracks = []                 # faces of the previous frame: (centre x, centre y, width, smoothed probabilities)
fps, n, t_model = 0.0, 0, []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if args.camera.isdigit():
        frame = cv2.flip(frame, 1)                  # mirror view, like looking in a mirror
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, faces, probs = deepfer.analyse_image(gray, finder, predictor)

    # smoothing over time: match each face to the closest face of the last frame and blend the probabilities,
    # so the label doesn't flicker between two emotions from one frame to the next
    new_tracks = []
    for (x, y, w, h), p in zip(boxes, probs):
        cx, cy = x + w / 2, y + h / 2
        near = [t for t in tracks if abs(t[0] - cx) < 0.5 * w and abs(t[1] - cy) < 0.5 * w]
        if near:
            p = args.smooth * near[0][3] + (1 - args.smooth) * p
        new_tracks.append((cx, cy, w, p))
    tracks = new_tracks
    dt = time.perf_counter() - t0
    t_model.append(dt * 1000)
    fps = 0.9 * fps + 0.1 * (1 / max(dt, 1e-3)) if n else 1 / max(dt, 1e-3)
    n += 1

    for (x, y, w, h), (_, _, _, p) in zip(boxes, tracks):
        top = CLASSES[int(np.argmax(p))]
        col = COLORS[top]
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        cv2.putText(frame, f"{top} {p.max():.0%}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        for k, c in enumerate(CLASSES):             # small bar chart of all seven emotions next to the face
            yy = y + 14 + k * 16
            bx = x + w + 8
            cv2.putText(frame, c[:4], (bx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(frame, (bx + 38, yy - 9), (bx + 38 + int(80 * p[k]), yy), COLORS[c], -1)
    cv2.putText(frame, f"{fps:.0f} fps | {dt * 1000:.0f} ms per frame | {len(boxes)} face(s) | q to quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if not args.no_window:
        cv2.imshow("DeepFER live", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    if args.max_frames and n >= args.max_frames:
        break

cap.release()
cv2.destroyAllWindows()
if t_model:
    print(f"{n} frames, median {np.median(t_model):.1f} ms per frame (about {1000 / np.median(t_model):.0f} frames per second)")
