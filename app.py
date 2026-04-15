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

# ── GLOBAL STYLES ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark navy background ── */
.stApp {
    background: #0D1117;
    color: #E6EDF3;
}

/* ── Gradient hero header ── */
.hero {
    background: linear-gradient(135deg, #6E3AFA 0%, #3A8BFA 60%, #00C9A7 100%);
    border-radius: 16px;
    padding: 40px 32px 32px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(110, 58, 250, 0.35);
}
.hero h1 {
    font-size: 2rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.82);
    margin: 0;
}

/* ── Camera widget label ── */
label[data-testid="stCameraInputLabel"] p {
    color: #A0AEC0 !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Result cards ── */
.result-card {
    border-radius: 14px;
    padding: 24px 28px;
    margin: 20px 0 12px;
    text-align: center;
}
.result-card .verdict {
    font-size: 1.7rem;
    font-weight: 900;
    letter-spacing: -0.3px;
    margin: 0 0 4px;
}
.result-card .sub {
    font-size: 0.85rem;
    opacity: 0.75;
    margin: 0;
}

.card-low {
    background: linear-gradient(135deg, #3D0C11, #6B1A22);
    border: 1px solid #C0392B;
    box-shadow: 0 0 24px rgba(192, 57, 43, 0.4);
    color: #FF6B6B;
}
.card-medium {
    background: linear-gradient(135deg, #3D2700, #6B4200);
    border: 1px solid #D4820A;
    box-shadow: 0 0 24px rgba(212, 130, 10, 0.4);
    color: #FFB347;
}
.card-high {
    background: linear-gradient(135deg, #003D22, #00622F);
    border: 1px solid #27AE60;
    box-shadow: 0 0 24px rgba(39, 174, 96, 0.4);
    color: #5CFF9D;
}

/* ── Confidence section ── */
.conf-header {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8B949E;
    margin: 20px 0 10px;
}

/* ── Progress bar colours ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6E3AFA, #3A8BFA) !important;
    border-radius: 99px;
}

/* ── Warning ── */
.stAlert {
    background: #1C2128 !important;
    border: 1px solid #30363D !important;
    color: #CDD9E5 !important;
    border-radius: 10px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #21262D;
}
.sidebar-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6E3AFA;
    font-weight: 700;
    margin-bottom: 12px;
}
.sidebar-row {
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid #21262D;
    font-size: 0.83rem;
}
.sidebar-row span:first-child { color: #8B949E; }
.sidebar-row span:last-child  { color: #E6EDF3; font-weight: 600; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 4px 2px 0;
}
.badge-low    { background: #3D1515; color: #FF6B6B; }
.badge-med    { background: #3D2700; color: #FFB347; }
.badge-high   { background: #003D22; color: #5CFF9D; }
.footer-text  { font-size: 0.75rem; color: #484F58; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ── LABELS ───────────────────────────────────────────────────
LABELS = {
    0: ("Low Engagement",    "card-low",    "Student appears disengaged."),
    1: ("Medium Engagement", "card-medium", "Student shows partial attention."),
    2: ("High Engagement",   "card-high",   "Student is actively engaged."),
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
    st.markdown("<div class='sidebar-title'>Project Info</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='sidebar-row'><span>Model</span><span>EfficientNet-B2</span></div>
<div class='sidebar-row'><span>Dataset</span><span>DAiSEE</span></div>
<div class='sidebar-row'><span>Test accuracy</span><span>72.6%</span></div>
<div class='sidebar-row'><span>Val accuracy</span><span>74.75%</span></div>
""", unsafe_allow_html=True)

    st.markdown("<br><div class='sidebar-title'>Classes</div>", unsafe_allow_html=True)
    st.markdown("""
<span class='badge badge-low'>Low</span>
<span class='badge badge-med'>Medium</span>
<span class='badge badge-high'>High</span>
""", unsafe_allow_html=True)

    st.markdown("<div class='footer-text'>SASTRA Deemed University</div>", unsafe_allow_html=True)

# ── HERO HEADER ──────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>Student Engagement Detector</h1>
    <p>Real-time facial engagement analysis powered by EfficientNet-B2</p>
</div>
""", unsafe_allow_html=True)

# ── CAMERA INPUT ─────────────────────────────────────────────
img_file = st.camera_input("Position your face in frame and capture")

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (110, 58, 250), 2)

        face_crop        = frame[y:y+h, x:x+w]
        label_idx, probs = predict(face_crop)
        label_name, css_class, subtitle = LABELS[label_idx]

        st.image(image, caption="Captured image", use_container_width=True)

        st.markdown(
            f"<div class='result-card {css_class}'>"
            f"  <p class='verdict'>{label_name}</p>"
            f"  <p class='sub'>{subtitle}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='conf-header'>Confidence breakdown</div>", unsafe_allow_html=True)
        for i, (name, _, _sub) in LABELS.items():
            st.progress(float(probs[i]), text=f"{name}  {probs[i]*100:.1f}%")

    else:
        st.image(image, caption="Captured image", use_container_width=True)
        st.warning("No face detected — make sure your face is clearly visible and try again.")
