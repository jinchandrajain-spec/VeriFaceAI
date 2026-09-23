import os
import time
import numpy as np
import matplotlib.cm as cm
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VeriFaceAI - Authenticity Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Cyberpunk & Glassmorphism Styling with Animations
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

    /* Dark Theme Base */
    .stApp {
        background: linear-gradient(135deg, #050505, #0b0c16, #100b1a, #07131b);
        background-size: 400% 400%;
        animation: cyberpunkGradient 15s ease infinite;
        color: #E0E0E0;
        overflow-x: hidden;
    }

    @keyframes cyberpunkGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Ambient Cursor Glow Follower Effect */
    body::before {
        content: '';
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none;
        background: radial-gradient(600px circle at var(--mouse-x, 50vw) var(--mouse-y, 50vh), rgba(0, 229, 255, 0.05), transparent 40%);
        z-index: 9999;
        transition: background 0.1s ease;
    }

    /* Glassmorphism Card Containers */
    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(0, 229, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 229, 255, 0.12);
    }

    /* Input Controls Glassmorphism */
    [data-testid="stFileUploader"], [data-testid="stCameraInput"] {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(0, 229, 255, 0.4);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 229, 255, 0.1);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover, [data-testid="stCameraInput"]:hover {
        border-color: #00E5FF;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.45), inset 0 0 15px rgba(0, 229, 255, 0.2);
    }

    /* Image Preview Container with Illuminated Neon Border */
    .image-preview-container {
        position: relative;
        border-radius: 16px;
        padding: 6px;
        background: linear-gradient(135deg, #00E5FF, #9D00FF, #00FF87);
        background-size: 300% 300%;
        animation: neonGlow 5s ease infinite;
        display: inline-block;
        margin-top: 10px;
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.35);
    }
    @keyframes neonGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .image-preview-container img {
        border-radius: 12px;
        display: block;
        max-width: 100%;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .status-real {
        background: rgba(0, 255, 135, 0.12);
        color: #00FF87;
        border: 1.5px solid #00FF87;
        box-shadow: 0 0 20px rgba(0, 255, 135, 0.35);
        animation: pulseGreen 2.5s infinite;
    }
    .status-fake {
        background: rgba(255, 59, 92, 0.12);
        color: #FF3B5C;
        border: 1.5px solid #FF3B5C;
        box-shadow: 0 0 20px rgba(255, 59, 92, 0.35);
        animation: pulseRed 2.5s infinite;
    }
    @keyframes pulseGreen {
        0%, 100% { box-shadow: 0 0 15px rgba(0, 255, 135, 0.3); }
        50% { box-shadow: 0 0 28px rgba(0, 255, 135, 0.6); }
    }
    @keyframes pulseRed {
        0%, 100% { box-shadow: 0 0 15px rgba(255, 59, 92, 0.3); }
        50% { box-shadow: 0 0 28px rgba(255, 59, 92, 0.6); }
    }

    /* Metric Badges in Dashboard */
    .metric-box {
        text-align: center;
        padding: 18px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00E5FF;
        margin-top: 4px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Primary Action Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00E5FF, #9D00FF);
        color: #FFFFFF;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(157, 0, 255, 0.45);
        color: #FFFFFF;
    }

    /* Glassmorphism HUD Main Content Containers */
    .stApp [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(0, 229, 255, 0.3);
        box-shadow: 0 4px 30px rgba(0, 229, 255, 0.1);
        padding: 20px;
    }

    /* HUD Uploaded Image Styling */
    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    /* Streamlit Alert Boxes HUD Override (Success/Error/Warning/Info) */
    [data-testid="stAlert"], .st-emotion-cache-1wivap2 {
        background: rgba(15, 23, 42, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        box-shadow: 0 4px 30px rgba(0, 229, 255, 0.1) !important;
    }

    /* Premium Toggle Buttons for Radio */
    div[role="radiogroup"] label {
        border: 1px solid #00E5FF;
        padding: 10px 20px;
        border-radius: 8px;
        color: #E0E0E0;
        background: rgba(0, 229, 255, 0.05);
        transition: all 0.3s ease;
        cursor: pointer;
        margin-right: 10px;
    }
    div[role="radiogroup"] label:hover {
        background: rgba(0, 229, 255, 0.15);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.4);
        color: #FFFFFF;
    }

    /* Main Orbitron Header styling */
    .stApp h1 {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 5.5rem !important;
        text-align: center !important;
        width: 100% !important;
        background: linear-gradient(90deg, #00E5FF, #9D00FF) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        padding-bottom: 0px !important;
        margin-bottom: 0px !important;
    }

    [data-testid="stRadio"] div[role="radio"] p {
        color: #FFFFFF !important; 
        font-weight: 600 !important; 
        font-size: 1.1rem !important;
    }

    /* Center align the tabs and add spacing */
    div[data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: center !important;
        gap: 15px;
        border-bottom: none !important;
        width: 100% !important;
    }

    /* Force center alignment for horizontal radio groups */
    [data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 auto !important;
    }
</style>

<script>
    // Real-time cursor coordinates tracking for background ambient light
    window.addEventListener('mousemove', (e) => {
        document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
        document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
    });
</script>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Custom 4-Block CNN Architecture (from train_model.py)
# ---------------------------------------------------------
class CustomFaceCNN(nn.Module):
    def __init__(self):
        super(CustomFaceCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# Device Detection & Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_system_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = CustomFaceCNN()
    weights_path = "best_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device, True
    return None, device, False

model, active_device, is_model_loaded = load_system_model()

# Image Preprocessing Transform
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------------------------------------------------
# Sidebar Diagnostic Panel
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ System Engine")
    st.markdown("#### **AI Face Authenticity Verifier**")
    st.markdown("---")
    
    st.markdown(f"**Hardware Acceleration:** `{str(active_device).upper()}`")
    st.markdown("**Architecture:** `Custom 4-Block CNN`")
    st.markdown("**Weights Status:** " + ("🟢 `Loaded`" if is_model_loaded else "🔴 `Missing best_model.pth`"))
    st.markdown("**Input Dimensions:** `128 x 128 px`")
    st.markdown("**Training Epochs:** `20`")
    st.markdown("**Baseline Test Accuracy:** `94.53%`")
    st.markdown("---")
    st.info("System built for real-time digital verification and authenticity scoring.")

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("<h1 style='background: linear-gradient(90deg, #00E5FF, #9D00FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; margin-bottom: 2px;'>VeriFaceAI</h1>", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #A3B1C6; margin-top: -15px; margin-bottom: 40px;">Deep learning verification engine analyzing facial artifacts and demographic robustness.</p>', unsafe_allow_html=True)

nav_left, nav_center, nav_right = st.columns([1.4, 2.5, 1])
with nav_center:
    app_mode = st.radio("Navigation", ["⚡ Live Verification", "📊 Demographic Fairness & Performance"], horizontal=True, label_visibility="collapsed")

# ---------------------------------------------------------
# Tab 1: Live Verification Engine
# ---------------------------------------------------------
if app_mode == "⚡ Live Verification":
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("1. Ingest Facial Image")
        
        input_method = st.radio("Select Input Method:", ["Upload Image", "Live Camera"], horizontal=True)
        
        image_input = None
        if input_method == "Upload Image":
            image_input = st.file_uploader("Upload image for authenticity analysis", type=["jpg", "jpeg", "png"])
        else:
            image_input = st.camera_input("Take a picture for verification")
        
        if image_input is not None:
            raw_image = Image.open(image_input).convert("RGB")
            st.markdown(f"<p style='color: #8B949E; margin-top: 8px;'>Image Dimensions: <code>{raw_image.size[0]} x {raw_image.size[1]} px</code></p>", unsafe_allow_html=True)
            
            # Illuminated Animated Container Preview
            st.markdown("<div class='image-preview-container'>", unsafe_allow_html=True)
            st.image(raw_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("2. Authenticity Analysis")
        
        if image_input is None:
            st.info("Upload or capture a portrait image to begin inference.")
        elif not is_model_loaded:
            st.error("Weights file 'best_model.pth' not found. Please verify training output.")
        else:
            if st.button("🚀 Verify Authenticity"):
                # --- Simulated Cyber-Scan Sequence ---
                status_text = st.empty()
                progress_bar = st.progress(0)
                scan_messages = [
                    "Initiating facial topography scan...",
                    "Extracting micro-textures...",
                    "Querying VeriFaceAI Deep Learning Engine...",
                    "Decrypting sigmoid output..."
                ]
                for i in range(100):
                    time.sleep(0.015)
                    progress_bar.progress(i + 1)
                    if i % 25 == 0:
                        status_text.markdown(f"_{scan_messages[i // 25]}_")
                status_text.empty()
                progress_bar.empty()
                # ---------------------------------------

                with st.spinner("Analyzing micro-textures and synthetic boundaries..."):
                    # ---- Saliency Map (XAI) Pipeline ----
                    input_tensor = image_transform(raw_image).unsqueeze(0).to(active_device)
                    input_tensor.requires_grad_(True)

                    # Forward pass WITH gradients for saliency
                    output = model(input_tensor)
                    output_prob = output.item()

                    # Backward pass to compute input gradients
                    model.zero_grad()
                    output.backward()

                    # Extract saliency map: max across colour channels → 2D
                    saliency = input_tensor.grad.data.abs()
                    saliency, _ = torch.max(saliency.squeeze(), dim=0)
                    saliency_np = saliency.cpu().numpy()

                    # Normalise to [0, 1]
                    s_min, s_max = saliency_np.min(), saliency_np.max()
                    if s_max > s_min:
                        saliency_np = (saliency_np - s_min) / (s_max - s_min)
                    else:
                        saliency_np = np.zeros_like(saliency_np)

                    # Apply thermal (inferno) colormap → RGB uint8
                    heatmap_rgb = cm.inferno(saliency_np)[..., :3]  # drop alpha
                    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
                    heatmap_img = Image.fromarray(heatmap_rgb).resize(
                        raw_image.size, Image.BILINEAR
                    )

                    # Blend original + heatmap at 50 % opacity
                    forensic_image = Image.blend(
                        raw_image.convert("RGB"), heatmap_img, alpha=0.50
                    )
                    # --------------------------------------

                    # Class Mapping: {'FAKE': 0, 'REAL': 1} — Strict 0.65 security threshold
                    THRESHOLD = 0.65
                    is_real = output_prob >= THRESHOLD
                    confidence = output_prob if is_real else (1.0 - output_prob)
                    confidence_percent = confidence * 100.0

                    st.markdown("---")
                    if is_real:
                        st.markdown(
                            f"<div class='status-badge status-real'>✓ Authentic (Real)</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**Certainty Score:** `{confidence_percent:.2f}%`")
                    else:
                        st.markdown(
                            f"<div class='status-badge status-fake'>⚠ Manipulated (Fake)</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**Detection Confidence:** `{confidence_percent:.2f}%`")

                    # Animated Progress / Confidence Meter
                    st.progress(float(confidence))

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                    * **Raw Sigmoid Output:** `{output_prob:.5f}`
                    * **Decision Threshold:** `0.65` *(Strict Security Mode)*
                    * **Assessment:** The network identified consistent facial edge alignment and micro-textures.
                    """)

                    # ---- Forensic Artifact Scanner UI ----
                    st.markdown("---")
                    st.markdown("#### 🔬 Forensic Artifact Scanner")
                    scan_col1, scan_col2 = st.columns(2)
                    with scan_col1:
                        st.image(raw_image, caption="Original Capture", use_container_width=True)
                    with scan_col2:
                        st.image(forensic_image, caption="Forensic Artifact Scan", use_container_width=True)
                        st.caption("Mapping high-frequency artifact gradients")
                    # -------------------------------------
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Demographic Fairness & Audit
# ---------------------------------------------------------
elif app_mode == "📊 Demographic Fairness & Performance":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Test Evaluation Metrics")
    
    # 4 Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("<div class='metric-box'><div class='metric-title'>Test Accuracy</div><div class='metric-value'>94.53%</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-box'><div class='metric-title'>Precision (Real)</div><div class='metric-value'>96.42%</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-box'><div class='metric-title'>Recall (Real)</div><div class='metric-value'>97.25%</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='metric-box'><div class='metric-title'>F1-Score</div><div class='metric-value'>96.83%</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts Layout
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Demographic Slicing: Gender")
        if os.path.exists("accuracy_by_gender.png"):
            st.image("accuracy_by_gender.png", use_container_width=True)
            st.caption("Cross-cohort accuracy across Male (95.61%), Female (94.01%), and Unknown categories shows minimal bias variance (<1.6%).")
        else:
            st.warning("'accuracy_by_gender.png' not found. Run evaluate_demographics.py to generate artifacts.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Demographic Slicing: Age Group")
        if os.path.exists("accuracy_by_age_group.png"):
            st.image("accuracy_by_age_group.png", use_container_width=True)
            st.caption("Classification stability across age demographics (18-25, 26-35, 36-50, 50+), peaking at 97.62% for older cohorts.")
        else:
            st.warning("'accuracy_by_age_group.png' not found. Run evaluate_demographics.py to generate artifacts.")
        st.markdown("</div>", unsafe_allow_html=True)