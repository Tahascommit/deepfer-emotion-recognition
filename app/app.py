# DeepFER: local deployment of the facial emotion model as a Streamlit web app.
# Run from the repo folder:  streamlit run app/app.py
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                       # deepfer.py (written by the notebook) lives in the repo folder
import deepfer                                      # noqa: E402
from deepfer import CLASSES                         # noqa: E402

st.set_page_config(page_title="DeepFER emotion recognition", layout="wide")
COLORS = {"angry": (220, 50, 50), "disgust": (40, 150, 40), "fear": (150, 60, 180), "happy": (240, 190, 0),
          "neutral": (150, 150, 150), "sad": (40, 110, 220), "surprise": (250, 130, 20)}      # RGB


@st.cache_resource
def load():
    """Load the final model and the face detector once, when the app starts."""
    return deepfer.EmotionPredictor(ROOT / "models"), deepfer.FaceFinder()


try:
    PRED, FINDER = load()
except FileNotFoundError:
    st.error("No saved model found. Run the notebook first: it writes the models to the models/ folder.")
    st.stop()
CARD = PRED.card


def decode(data):
    """Bytes of an uploaded file -> RGB array."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def analyse(rgb, whole_face):
    """Find the faces, crop them like the training faces and score them. Returns everything needed to show the result."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    t0 = time.perf_counter()
    if whole_face:                                  # the picture is already a cropped face
        boxes = [(0, 0, gray.shape[1], gray.shape[0])]
        faces = [cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)]
        probs = PRED.predict(faces)
    else:
        boxes, faces, probs = deepfer.analyse_image(gray, FINDER, PRED)
    ms = (time.perf_counter() - t0) * 1000
    out = rgb.copy()
    thick = max(2, rgb.shape[1] // 300)
    for k, ((x, y, w, h), p) in enumerate(zip(boxes, probs)):
        col = COLORS[CLASSES[p.argmax()]]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, thick)
        label = f"{k + 1}: {CLASSES[p.argmax()]} {p.max():.0%}"
        scale = max(0.5, w / 250)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        y0 = max(th + 6, y)
        cv2.rectangle(out, (x, y0 - th - 8), (x + tw + 6, y0), col, -1)
        cv2.putText(out, label, (x + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return out, boxes, faces, probs, ms


def show(rgb, key):
    """Show the annotated picture and one result card per face. The key changes with every new picture,
    so the 'cropped face' box starts from the right default each time."""
    small = max(rgb.shape[:2]) <= 120
    whole = st.checkbox("The picture is already a cropped face (skip face detection)", value=small, key=f"whole_{key}")
    out, boxes, faces, probs, ms = analyse(rgb, whole)
    left, right = st.columns([3, 2])
    with left:
        st.image(out, width="stretch" if max(rgb.shape[:2]) > 300 else 360)
        st.caption(f"{len(boxes)} face(s) found, analysed in {ms:.0f} ms on {PRED.device.type}")
    with right:
        if not boxes:
            st.warning("No face found. Try a brighter, front-facing photo, or tick the box above if the picture is a face crop.")
        for k, (face, p) in enumerate(zip(faces, probs)):
            top = CLASSES[p.argmax()]
            st.markdown(f"**Face {k + 1}: {top}** ({p.max():.0%} sure)")
            if p.max() < MIN_CONF:
                st.info(f"Not sure: the confidence is below {MIN_CONF:.0%}. The second guess is "
                        f"{CLASSES[np.argsort(p)[-2]]} ({np.sort(p)[-2]:.0%}).")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(cv2.resize(face, (120, 120), interpolation=cv2.INTER_NEAREST), caption="what the model sees (48x48)")
                if SHOW_CAM:
                    cam = PRED.explain(face)
                    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
                    mix = (0.55 * cv2.cvtColor(face, cv2.COLOR_GRAY2RGB) + 0.45 * heat).astype(np.uint8)
                    st.image(cv2.resize(mix, (120, 120)), caption="Grad-CAM: where it looked")
            with c2:
                st.bar_chart(pd.DataFrame({"probability": p}, index=CLASSES), height=220, horizontal=True)


# ---------- sidebar
with st.sidebar:
    st.header("Model")
    st.write(f"**{' + '.join(CARD['member_names'])}**" + (" with mirrored-face averaging" if CARD["tta"] else ""))
    ts = CARD["test_scores"]
    st.metric("Test accuracy", f"{ts['accuracy']:.1%}")
    st.metric("Test macro F1", f"{ts['macro F1']:.3f}")
    st.caption("Scores on the 6,952 test faces the model never saw. The time of every analysis is shown under the picture.")
    MIN_CONF = st.slider("Say 'not sure' below this confidence", 0.0, 1.0, 0.5, 0.05)
    SHOW_CAM = st.checkbox("Show Grad-CAM heat maps", value=True)
    st.caption("Emotions: " + ", ".join(CLASSES))

st.title("DeepFER: facial emotion recognition")
tab_photo, tab_cam, tab_live, tab_info = st.tabs(["Photo", "Webcam snapshot", "Live webcam", "Model insights"])

# ---------- tab 1: a photo
with tab_photo:
    up = st.file_uploader("Upload a photo with one or more faces", type=["jpg", "jpeg", "png", "webp", "bmp"])
    samples = sorted((ROOT / "outputs" / "sample_faces").glob("*.png"))
    demo = ROOT / "outputs" / "demo_frame.png"
    choices = ["(none)"] + (["demo frame with six test faces"] if demo.exists() else []) + [f.name for f in samples]
    pick = st.selectbox("...or try a test picture (the file name is the true emotion)", choices)
    rgb = None
    if up is not None:
        rgb, pic_id = decode(up.getvalue()), f"up_{up.file_id}"
        if rgb is None:
            st.error("This file could not be read as an image.")
    elif pick != "(none)":
        path = demo if pick.startswith("demo") else ROOT / "outputs" / "sample_faces" / pick
        rgb, pic_id = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), pick
    if rgb is not None:
        show(rgb, f"photo_{pic_id}")

# ---------- tab 2: a snapshot from the webcam
with tab_cam:
    st.write("Take a picture with your webcam. Face the camera with even light for the best result.")
    shot = st.camera_input("Webcam")
    if shot is not None:
        show(decode(shot.getvalue()), f"cam_{shot.file_id}")

# ---------- tab 3: the live webcam window
with tab_live:
    st.write("Live video runs in its own window (OpenCV), which is faster than sending every frame through the browser. "
             "Every frame: find the faces, crop them, predict the emotion, smooth it over the last frames and draw it, "
             "with the frames per second in the corner. Press **q** in the window to close it.")
    st.code("python app/webcam.py", language="bash")
    if st.button("Open the live webcam window"):
        subprocess.Popen([sys.executable, str(ROOT / "app" / "webcam.py")], cwd=str(ROOT))
        st.success("Starting... the window opens in a few seconds. On a Mac, allow camera access if asked.")

# ---------- tab 4: what the notebook found
with tab_info:
    out = ROOT / "outputs"
    if (out / "model_comparison.csv").exists():
        st.subheader("All models (test set)")
        comp = pd.read_csv(out / "model_comparison.csv", index_col=0)
        st.dataframe(comp[[c for c in comp.columns if c.startswith("test")]].round(3), width="stretch")
    for fname, caption in [("final_confusion_matrix.png", "Confusion matrix of the final model"),
                           ("explain_gradcam_average.png", "Average Grad-CAM per emotion"),
                           ("explain_occlusion.png", "Which face regions each emotion depends on"),
                           ("final_robustness.png", "How the model copes with lighting, noise, blur and a covered mouth"),
                           ("final_coverage_accuracy.png", "Accuracy when the model may say 'not sure'")]:
        if (out / "figures" / fname).exists():
            st.image(str(out / "figures" / fname), caption=caption)
    if (out / "per_emotion_report.csv").exists():
        st.subheader("Scores per emotion (test set)")
        st.dataframe(pd.read_csv(out / "per_emotion_report.csv", index_col=0).round(3), width="stretch")
