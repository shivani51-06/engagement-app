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
    page_icon="🎓",
    layout="centered"
)

# ── LABELS ───────────────────────────────────────────────────
LABELS = {
    0: ("Low Engagement", "🔴", "#FF4B4B"),
    1: ("Medium Engagement", "🟡", "#FFA500"),
    2: ("High Engagement", "🟢", "#00C853"),
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

# ── UI ───────────────────────────────────────────────────────
st.title("🎓 Student Engagement Detector")
st.markdown("Capture your image and detect engagement level using **EfficientNet-B2**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Project:** Facial Engagement Detection  
    **Model:** EfficientNet-B2  
    **Dataset:** DAiSEE  
    
    **Accuracy:**
    - Test: 72.6%  
    - Validation: 74.75%  

    **Classes:**
    - 🔴 Low Engagement  
    - 🟡 Medium Engagement  
    - 🟢 High Engagement  
    """)
    st.markdown("---")
    st.markdown("SASTRA Deemed University")

# ── CAMERA INPUT ─────────────────────────────────────────────
img_file = st.camera_input("📸 Take a photo")

if img_file is not None:
    image = Image.open(img_file)
    frame = np.array(image)

    # Convert RGB → BGR (important)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3])[-1]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_crop = frame[y:y+h, x:x+w]
        label_idx, probs = predict(face_crop)

        label_name, emoji, color = LABELS[label_idx]

        # Show image
        st.image(image, caption="Captured Image", use_container_width=True)

        # Show label
        st.markdown(
            f"<h2 style='text-align:center; color:{color}'>"
            f"{emoji} {label_name}</h2>",
            unsafe_allow_html=True
        )

        # Confidence
        st.markdown("**Confidence Scores:**")
        for i, (name, em, _) in LABELS.items():
            st.progress(
                float(probs[i]),
                text=f"{em} {name}: {probs[i]*100:.1f}%"
            )

    else:
        st.image(image, caption="Captured Image", use_container_width=True)
        st.warning("⚠️ No face detected — please try again")
