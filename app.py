import streamlit as st
import numpy as np
import pandas as pd
import json
import requests
import tempfile
import os
import time
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Agri-Health Vision: Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- BASIC STYLES ----------
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(18px);
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg,#00b894,#00cec9);
        border-radius: 16px;
        padding: 1.5rem;
        color: #fff;
        text-align: center;
    }
    .prediction-box h1 {font-size: 2rem; margin-bottom: 0.2rem;}
    .prediction-box p {margin: 0; font-size: 1.1rem;}
    .top-bar {margin-bottom: 1rem;}
    .metric-card {
        background: rgba(0,0,0,0.15);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- GOOGLE DRIVE DOWNLOAD HELPERS ----------
GDRIVE_FILE_ID = "1DAWfGqtpzGT9khOvH79McW4SG_UBheft"  # your model file

def download_from_google_drive(file_id: str, destination: Path):
    """
    Download a (large) file from Google Drive handling the confirm token.
    Uses the classic pattern from StackOverflow. [web:11]
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_model_and_classes():
    """
    Ensure model.h5 exists locally (download once from Drive) and load it,
    along with class_names.json from the working directory.
    """
    try:
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "plant_disease_model.h5"

        # Download only if not already cached on disk
        if not model_path.exists():
            with st.spinner("⬇️ Downloading CNN model from Google Drive (one‑time)..."):
                download_from_google_drive(GDRIVE_FILE_ID, model_path)

        # Load Keras model from file path
        model = tf.keras.models.load_model(str(model_path))

        # Load class names from repo
        with open("class_names.json", "r") as f:
            class_names = json.load(f)

        return model, class_names

    except Exception as e:
        st.error(f"Model loading failed: {e}. Using demo mode.")
        return None, ["Healthy", "Bacterial Spot", "Early Blight", "Late Blight"]


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_disease(model, image_array, class_names):
    """
    If model is None → demo mode.
    Otherwise returns top‑3 (label, confidence) predictions.
    """
    if model is None:
        # Demo only
        return [("Healthy", 0.923), ("Bacterial Spot", 0.044), ("Early Blight", 0.033)], True

    preds = model.predict(image_array, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:3]
    results = [(class_names[i], float(preds[i])) for i in top_idx]
    return results, False

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        """
        <div class="glass-card">
            <h2 style="color:#2ecc71; text-align:center; margin-bottom:0.2rem;">
                🌱 Agri‑Health Vision
            </h2>
            <p style="text-align:center; color:#bdc3c7; margin-bottom:0.6rem;">
                SIH 2025 Prototype
            </p>
            <hr>
            <p><b>Team:</b> AI Innovators</p>
            <p><b>Stack:</b> CNN · Streamlit · MobileNetV2</p>
            <p><b>Model Source:</b> Google Drive @ runtime</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='top-bar'><h1 style='color:#ecf0f1;'>Agri‑Health Vision: AI‑Powered Plant Disease Classifier</h1></div>",
    unsafe_allow_html=True,
)

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(
    ["🔎 AI Diagnosis & Classifier", "🌿 Integrated Treatment Protocol", "📊 Farm Health Monitor"]
)

# ---------- TREATMENT DATA ----------
TREATMENT_DATA = {
    "Tomato___Bacterial_spot": {
        "symptoms": [
            "Small, water‑soaked leaf spots with yellow halos",
            "Lesions on fruits that become scabby",
        ],
        "causes": [
            "Bacterial infection (Xanthomonas spp.)",
            "Warm, humid, rainy weather",
        ],
        "chemical": [
            "Copper‑based bactericides (follow label)",
            "Fixed copper plus mancozeb sprays",
        ],
        "organic": [
            "Use certified disease‑free seed",
            "Remove and destroy infected debris",
            "Avoid overhead irrigation; improve spacing",
        ],
    },
    "Tomato___Early_blight": {
        "symptoms": [
            "Dark concentric rings on older leaves",
            "Yellowing and defoliation from bottom upwards",
        ],
        "causes": [
            "Fungus Alternaria solani",
            "High humidity with 20‑30°C temperature",
        ],
        "chemical": [
            "Mancozeb or chlorothalonil sprays",
            "Strobilurin fungicides where permitted",
        ],
        "organic": [
            "Crop rotation with non‑solanaceous crops",
            "Mulching to reduce soil splash",
            "Remove lower infected leaves",
        ],
    },
    "Tomato___Late_blight": {
        "symptoms": [
            "Large water‑soaked lesions with pale borders",
            "White fungal growth on underside in humid weather",
        ],
        "causes": [
            "Oomycete Phytophthora infestans",
            "Cool, moist conditions and leaf wetness",
        ],
        "chemical": [
            "Systemic fungicides (metalaxyl‑M, cymoxanil mixes)",
            "Protectant fungicides (mancozeb, chlorothalonil)",
        ],
        "organic": [
            "Use resistant varieties where available",
            "Destroy volunteer potatoes and tomatoes",
            "Avoid late evening irrigation",
        ],
    },
    "Healthy": {
        "symptoms": ["Uniform green leaves, no lesions, no wilting"],
        "causes": ["Balanced nutrition and good agronomy"],
        "chemical": ["No chemical control required"],
        "organic": ["Continue regular monitoring and good field hygiene"],
    },
}

# ---------- TAB 1: AI DIAGNOSIS ----------
with tab1:
    with st.container():
        left, right = st.columns([1.1, 1.4])

        with left:
            st.subheader("Upload Plant Leaf Image")
            uploaded = st.file_uploader(
                "Browse files",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear close‑up of a single leaf",
            )

        if uploaded is not None:
            image = Image.open(uploaded)

            with st.spinner("⚙️ Processing image using MobileNetV2‑based CNN..."):
                # Load model & class names (cached after first time)
                model, class_names = load_model_and_classes()
                arr = preprocess_image(image)
                preds, demo_flag = predict_disease(model, arr, class_names)
                time.sleep(0.5)

            img_col, pred_col = st.columns([1.2, 1.3])

            with img_col:
                st.image(image, caption="Uploaded Image", use_column_width=True)

            with pred_col:
                main_label, main_conf = preds[0]
                display_name = main_label.replace("___", " ").replace("_", " ")
                conf_pct = main_conf * 100

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h1>{display_name}</h1>
                        <p>Confidence: {conf_pct:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if demo_flag:
                    st.warning(
                        "Model is in demo mode (CNN failed to load). "
                        "Predictions are simulated – please check Drive or internet connectivity.",
                        icon="⚠️",
                    )
                else:
                    st.success("✅ Using your trained CNN model from Google Drive at runtime.")

                st.markdown("<br>", unsafe_allow_html=True)
                metric_wrap = st.container()
                with metric_wrap:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Prediction confidence", f"{conf_pct:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.subheader("🔍 Differential Diagnosis (Top 3)")
                for rank, (lbl, c) in enumerate(preds, start=1):
                    nm = lbl.replace("___", " ").replace("_", " ")
                    st.write(f"{rank}. {nm} ({c*100:.1f}%)")
                    st.progress(min(float(c), 1.0))

        else:
            st.info("Upload a leaf image on the left to start diagnosis.", icon="📷")

# ---------- TAB 2: INTEGRATED TREATMENT ----------
with tab2:
    st.subheader("Select a Disease for Protocol")

    all_diseases = list(TREATMENT_DATA.keys())
    selected = st.selectbox(
        "Disease",
        all_diseases,
        format_func=lambda x: x.replace("___", " ").replace("_", " "),
    )

    data = TREATMENT_DATA[selected]
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📋 Symptoms & Causes")
        st.write("**Key visual symptoms:**")
        for s in data["symptoms"]:
            st.write(f"- {s}")
        st.write("**Likely causes / triggers:**")
        for c in data["causes"]:
            st.write(f"- {c}")

    with c2:
        st.markdown("### 💊 Solution Protocol")
        st.write("**Chemical (use label‑recommended dose & safety):**")
        for chem in data["chemical"]:
            st.write(f"- {chem}")
        st.write("**Organic / cultural practices:**")
        for org in data["organic"]:
            st.write(f"- {org}")
        st.write("**Illustration placeholder:**")
        st.image(
            "https://via.placeholder.com/400x220.png?text=Disease+Image",
            caption="Representative disease image (replace with real field photo).",
            use_column_width=True,
        )

# ---------- TAB 3: FARM HEALTH MONITOR ----------
with tab3:
    st.subheader("Simulated 7‑Day Farm Micro‑Climate")

    days = pd.date_range(end=pd.Timestamp.today().normalize(), periods=7)
    np.random.seed(42)
    humidity = 70 + 12 * np.sin(np.linspace(0, 3, 7)) + np.random.normal(0, 2, 7)
    temperature = 26 + 4 * np.sin(np.linspace(0, 2.5, 7)) + np.random.normal(0, 1.5, 7)
    soil_ph = 6.5 + 0.3 * np.sin(np.linspace(0, 3, 7)) + np.random.normal(0, 0.15, 7)

    df_h = pd.DataFrame({"Day": days, "Humidity (%)": humidity}).set_index("Day")
    df_t = pd.DataFrame({"Day": days, "Temperature (°C)": temperature}).set_index("Day")
    df_p = pd.DataFrame({"Day": days, "Soil pH": soil_ph}).set_index("Day")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("💧 Humidity")
        st.line_chart(df_h)
    with c2:
        st.write("🌡️ Temperature")
        st.line_chart(df_t)
    with c3:
        st.write("🧪 Soil pH")
        st.line_chart(df_p)

    avg_h = float(humidity.mean())
    avg_t = float(temperature.mean())

    if avg_h > 80 and 22 <= avg_t <= 30:
        st.error(
            "HIGH FUNGAL RISK: Sustained high humidity with moderate temperature – "
            "intensify scouting and preventive fungicide sprays where needed."
        )
    elif avg_h > 75:
        st.warning(
            "MODERATE RISK: High humidity detected – ensure good ventilation and avoid leaf wetness."
        )
    else:
        st.success("LOW RISK: Current micro‑climate is not highly favorable for foliar fungal diseases.")

st.markdown(
    "<hr><p style='text-align:center; color:#7f8c8d;'>Built for SIH 2025 | Powered by CNN & Streamlit 🌱</p>",
    unsafe_allow_html=True,
)
