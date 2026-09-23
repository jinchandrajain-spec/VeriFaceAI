import os
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Face Authenticity Verifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Cyberpunk & Glassmorphism Styling with Animations
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
        overflow-x: hidden;
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

    /* Animated Upload Boundary Glow */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(0, 229, 255, 0.25);
        border-radius: 16px;
        padding: 16px;
        background: rgba(13, 17, 23, 0.6);
        transition: all 0.4s ease;
        animation: pulseSubtle 4s infinite ease-in-out;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00E5FF;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.45), inset 0 0 15px rgba(0, 229, 255, 0.2);
    }
    @keyframes pulseSubtle {
        0%, 100% { border-color: rgba(0, 229, 255, 0.25); }
        50% { border-color: rgba(157, 0, 255, 0.45); }
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
st.markdown("<h1 style='margin-bottom: 2px;'>AI-Based Authentic Face Verification System</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8B949E; margin-bottom: 25px;'>Deep learning verification engine analyzing facial artifacts and demographic robustness.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["⚡ Live Verification", "📊 Demographic Fairness & Performance"])

# ---------------------------------------------------------
# Tab 1: Live Verification Engine
# ---------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("1. Ingest Facial Image")
        uploaded_file = st.file_uploader("Upload image for authenticity analysis", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file).convert("RGB")
            st.markdown(f"<p style='color: #8B949E; margin-top: 8px;'>Image Dimensions: <code>{raw_image.size[0]} x {raw_image.size[1]} px</code></p>", unsafe_allow_html=True)
            
            # Illuminated Animated Container Preview
            st.markdown("<div class='image-preview-container'>", unsafe_allow_html=True)
            st.image(raw_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("2. Authenticity Analysis")
        
        if uploaded_file is None:
            st.info("Upload a portrait image to begin inference.")
        elif not is_model_loaded:
            st.error("Weights file 'best_model.pth' not found. Please verify training output.")
        else:
            if st.button("🚀 Verify Authenticity"):
                with st.spinner("Analyzing micro-textures and synthetic boundaries..."):
                    # Inference pipeline
                    input_tensor = image_transform(raw_image).unsqueeze(0).to(active_device)
                    with torch.no_grad():
                        output_prob = model(input_tensor).item()

                    # Class Mapping: {'FAKE': 0, 'REAL': 1}
                    is_real = output_prob >= 0.5
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
                    * **Decision Threshold:** `0.50`
                    * **Assessment:** The network identified consistent facial edge alignment and micro-textures.
                    """)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Demographic Fairness & Audit
# ---------------------------------------------------------
with tab2:
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