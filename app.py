import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
from model_v2 import EfficientNetB2
import gdown
import os

# ── MODEL CONFIG ─────────────────────────────────────────────
MODEL_PATH = "final_model_v2.pth"
GDRIVE_URL = "https://drive.google.com/uc?id=1C0TaYyVNat46tI76BCUFCtgxOljN0k96"

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Engagement Detector",
    page_icon=None,
    layout="centered"
)

# ── THEME STYLES ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Accent colour: indigo-blue */
    :root { --accent: #4F6EF7; }

    h1 { color: #1E2A5E; letter-spacing: -0.5px; }

    .result-card {
        border-left: 5px solid var(--accent);
        background: #F4F6FF;
        padding: 16px 20px;
        border-radius: 6px;
        margin: 16px 0;
    }
    .result-label {
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
    }
    .result-low    { color: #D7263D; border-color: #D7263D !important; }
    .result-medium { color: #E07B00; border-color: #E07B00 !important; }
    .result-high   { color: #1A7F4B; border-color: #1A7F4B !important; }

    .sidebar-section { font-size: 0.85rem; color: #555; }
    .sidebar-section b { color: #1E2A5E; }
</style>
""", unsafe_allow_html=True)

# ── LABELS ───────────────────────────────────────────────────
LABELS = {
    0: ("Low Engagement",    "result-low",    "#D7263D"),
    1: ("Medium Engagement", "result-medium", "#E07B00"),
    2: ("High Engagement",   "result-high",   "#1A7F4B"),
}

# ── TRANSFORM ────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── LOAD MODEL ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights…"):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)
    device = torch.device("cpu")
    model = EfficientNetB2()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    return model, device

model, device = load_model()

# ── FACE DETECTOR ────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── PREDICTION FUNCTION ──────────────────────────────────────
def predict(face_img):
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        label = torch.argmax(probs).item()

    return label, probs.cpu().numpy()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
<div class='sidebar-section'>
<b>Project:</b> Facial Engagement Detection<br>
<b>Model:</b> EfficientNet-B2<br>
<b>Dataset:</b> DAiSEE<br><br>
<b>Accuracy</b><br>
&nbsp;&nbsp;Test: 72.6%<br>
&nbsp;&nbsp;Validation: 74.75%<br><br>
<b>Classes</b><br>
&nbsp;&nbsp;Low Engagement<br>
&nbsp;&nbsp;Medium Engagement<br>
&nbsp;&nbsp;High Engagement
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='sidebar-section'>SASTRA Deemed University</div>",
                unsafe_allow_html=True)

# ── MAIN UI ──────────────────────────────────────────────────
st.title("Student Engagement Detector")
st.markdown("Take a photo to analyse your engagement level using EfficientNet-B2.")
st.markdown("---")

img_file = st.camera_input("Capture image")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3])[-1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (79, 110, 247), 2)

        face_crop = frame[y:y+h, x:x+w]
        label_idx, probs = predict(face_crop)

        label_name, css_class, color = LABELS[label_idx]

        st.image(image, caption="Captured image", use_container_width=True)

        st.markdown(
            f"<div class='result-card {css_class}'>"
            f"<p class='result-label'>{label_name}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("**Confidence**")
        for i, (name, _, _color) in LABELS.items():
            st.progress(float(probs[i]), text=f"{name}: {probs[i]*100:.1f}%")

    else:
        st.image(image, caption="Captured image", use_container_width=True)
        st.warning("No face detected — please face the camera and try again.")
