import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
from model_v2 import EfficientNetB2
import gdown
import os

MODEL_PATH = "final_model_v2.pth"
url = "https://drive.google.com/uc?id=1C0TaYyVNat46tI76BCUFCtgxOljN0k96"

# Download ONLY if not present
if not os.path.exists(MODEL_PATH):
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Engagement Detector",
    page_icon="🎓",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────
LABELS = {
    0: ("Low Engagement",    "🔴", "#FF4B4B"),
    1: ("Medium Engagement", "🟡", "#FFA500"),
    2: ("High Engagement",   "🟢", "#00C853"),
}

TRANSFORM = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load model (cached so it loads only once) ─────────────────
@st.cache_resource
def load_model():
    device = torch.device("cpu")   # laptop inference on CPU is fine
    model = EfficientNetB2()
    model.load_state_dict(
        torch.load("final_model_v2.pth", map_location=device)
    )
    model.eval()
    return model, device

model, device = load_model()

# ── Face detector (OpenCV Haar Cascade) ───────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Predict engagement from a face crop ───────────────────────
def predict(face_img):
    """face_img: numpy BGR array"""
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor  = TRANSFORM(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        label  = torch.argmax(probs).item()
    return label, probs.cpu().numpy()

# ── UI ────────────────────────────────────────────────────────
st.title("🎓 Student Engagement Detector")
st.markdown("Real-time facial engagement classification using **EfficientNet-B2**")
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Project:** Facial Emotion Recognition  
    for E-Learning Engagement  
    
    **Model:** EfficientNet-B2  
    **Dataset:** DAiSEE  
    **Test Accuracy:** 72.6%  
    **Val Accuracy:** 74.75%
    
    **Classes:**
    - 🔴 Low Engagement
    - 🟡 Medium Engagement  
    - 🟢 High Engagement
    """)
    st.markdown("---")
    st.markdown("SASTRA Deemed University")

# Start/Stop buttons
col1, col2 = st.columns(2)
start = col1.button("▶ Start Camera", use_container_width=True)
stop  = col2.button("⏹ Stop Camera",  use_container_width=True)

# Placeholders
frame_placeholder      = st.empty()
label_placeholder      = st.empty()
confidence_placeholder = st.empty()
noface_placeholder     = st.empty()

# ── Webcam loop ───────────────────────────────────────────────
if start:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Make sure it is connected and not in use.")
    else:
        st.session_state["running"] = True

        while st.session_state.get("running", True):

            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read frame.")
                break

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) > 0:
                noface_placeholder.empty()

                # Use the largest detected face
                x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3])[-1]

                # Draw box around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Crop and predict
                face_crop = frame[y:y+h, x:x+w]
                label_idx, probs = predict(face_crop)

                label_name, emoji, color = LABELS[label_idx]

                # Annotate frame
                cv2.putText(
                    frame, f"{emoji} {label_name}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2
                )

                # Show frame
                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True
                )

                # Show engagement label
                label_placeholder.markdown(
                    f"<h2 style='text-align:center; color:{color}'>"
                    f"{emoji} {label_name}</h2>",
                    unsafe_allow_html=True
                )

                # Show confidence bars
                confidence_placeholder.markdown("**Confidence Scores:**")
                for i, (name, em, _) in LABELS.items():
                    confidence_placeholder.progress(
                        float(probs[i]),
                        text=f"{em} {name}: {probs[i]*100:.1f}%"
                    )

            else:
                # No face detected
                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", use_container_width=True
                )
                noface_placeholder.warning("⚠️ No face detected — please face the camera")
                label_placeholder.empty()
                confidence_placeholder.empty()

            # Check if stop was pressed
            if stop:
                st.session_state["running"] = False
                break

        cap.release()
        frame_placeholder.empty()
        label_placeholder.empty()
        confidence_placeholder.empty()
        noface_placeholder.empty()
        st.success("✅ Camera stopped.")
