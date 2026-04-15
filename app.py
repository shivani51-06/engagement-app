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
    page_title="Engagement Detector",
    page_icon=None,
    layout="centered"
)

# ── STYLES ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: #111111;
    color: #EDEDED;
}

/* Red top stripe */
.stApp::before {
    content: '';
    display: block;
    height: 4px;
    background: #B91C1C;
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
}

/* Page title */
h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #F5F5F5 !important;
    margin-bottom: 2px !important;
}
.subtitle {
    font-size: 0.85rem;
    color: #666;
    margin-bottom: 24px;
}

/* Divider */
hr { border-color: #222 !important; }

/* Result display */
.result-wrap {
    margin: 20px 0 6px;
    padding: 18px 20px;
    border-left: 3px solid #B91C1C;
    background: #1A1A1A;
}
.result-label {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0 0 2px;
}
.result-note {
    font-size: 0.8rem;
    color: #777;
    margin: 0;
}
.label-low    { color: #EF4444; }
.label-medium { color: #F59E0B; }
.label-high   { color: #22C55E; }

/* Confidence */
.conf-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555;
    margin: 18px 0 8px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #181818;
    border-right: 1px solid #222;
}
.sb-heading {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #B91C1C;
    font-weight: 700;
    margin: 0 0 10px;
}
.sb-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #222;
    font-size: 0.82rem;
}
.sb-row .k { color: #555; }
.sb-row .v { color: #DDD; font-weight: 500; }
.sb-foot {
    font-size: 0.72rem;
    color: #444;
    margin-top: 24px;
}

/* Warning */
div[data-testid="stAlert"] {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-left: 3px solid #B91C1C !important;
    color: #AAA !important;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── LABELS ───────────────────────────────────────────────────
LABELS = {
    0: ("Low",    "label-low"),
    1: ("Medium", "label-medium"),
    2: ("High",   "label-high"),
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

# ── PREDICTION ───────────────────────────────────────────────
def predict(face_img):
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        label  = torch.argmax(probs).item()
    return label, probs.cpu().numpy()

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sb-heading'>Model</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='sb-row'><span class='k'>Architecture</span><span class='v'>EfficientNet-B2</span></div>
<div class='sb-row'><span class='k'>Dataset</span><span class='v'>DAiSEE</span></div>
<div class='sb-row'><span class='k'>Test accuracy</span><span class='v'>72.6%</span></div>
<div class='sb-row'><span class='k'>Val accuracy</span><span class='v'>74.75%</span></div>
""", unsafe_allow_html=True)

    st.markdown("<br><div class='sb-heading'>Classes</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='sb-row'><span class='k'>0</span><span class='v' style='color:#EF4444'>Low</span></div>
<div class='sb-row'><span class='k'>1</span><span class='v' style='color:#F59E0B'>Medium</span></div>
<div class='sb-row'><span class='k'>2</span><span class='v' style='color:#22C55E'>High</span></div>
""", unsafe_allow_html=True)

    st.markdown("<div class='sb-foot'>SASTRA Deemed University</div>", unsafe_allow_html=True)

# ── PAGE HEADER ──────────────────────────────────────────────
st.title("Engagement Analysis")
st.markdown("<div class='subtitle'>Facial engagement detection — EfficientNet-B2 / DAiSEE</div>", unsafe_allow_html=True)
st.markdown("---")

# ── CAMERA INPUT ─────────────────────────────────────────────
img_file = st.camera_input("Capture")

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (185, 28, 28), 2)

        face_crop        = frame[y:y+h, x:x+w]
        label_idx, probs = predict(face_crop)
        label_name, css_class = LABELS[label_idx]

        st.image(image, use_container_width=True)

        st.markdown(
            f"<div class='result-wrap'>"
            f"<p class='result-note'>Engagement level</p>"
            f"<p class='result-label {css_class}'>{label_name}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='conf-title'>Confidence</div>", unsafe_allow_html=True)
        for i, (name, _) in LABELS.items():
            st.progress(float(probs[i]), text=f"{name} — {probs[i]*100:.1f}%")

    else:
        st.image(image, use_container_width=True)
        st.warning("No face detected. Face the camera directly and try again.")
