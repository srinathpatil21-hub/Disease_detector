import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import gdown
from tensorflow.keras.models import load_model as keras_load_model

# ============================================================
# Model download & loading (Google Drive)
# ============================================================
MODEL_PATH = "plant_disease_model.h5"
DRIVE_ID = "1DAWfGqtpzGT9khOvH79McW4SG_UBheft"
# Using the correct direct download link format for gdown
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}" 

def ensure_model_downloaded():
    """Checks if the model file exists and downloads it if it does not."""
    if not os.path.exists(MODEL_PATH):
        st.info(f"Downloading model (ID: {DRIVE_ID}). This may take a moment...")
        try:
            # gdown.download handles the download from the constructed DRIVE_URL
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model from Google Drive. Please ensure the Drive ID is correct and the file is publicly accessible. Error: {e}")

# Default disease list (used if class_names.json missing)
DISEASE_LIST = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___Mosaic_virus",
    "Tomato___healthy",
]

@st.cache_resource(show_spinner="Loading CNN model and class names...")
def load_model_and_classes():
    """Loads the Keras model and the class names list."""
    ensure_model_downloaded()
    
    # Check if model path exists again, in case download failed
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found after attempted download. Cannot load model.")
        return None, DISEASE_LIST

    model = keras_load_model(MODEL_PATH)

    class_file = "class_names.json"
    class_names = DISEASE_LIST
    if os.path.exists(class_file):
        try:
            with open(class_file, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        except Exception as e:
            st.warning(f"Could not load class names from JSON. Using default list. Error: {e}")

    return model, class_names

# ============================================================
# Page configuration & base CSS
# ============================================================
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Styles for Dark/Agronomy Theme */
    body {
        background: radial-gradient(circle at top left, #052b2f 0, #000000 55%, #052b2f 100%);
        color: #f5f5f5;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    .main {
        padding: 1rem 2rem;
    }
    
    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(18px);
    }
    
    /* Highlighted Header Style */
    .glass-header {
        background: linear-gradient(120deg, rgba(34,197,94,0.18), rgba(56,189,248,0.18));
        border-radius: 18px;
        padding: 1.2rem 1.6rem;
        border: 1px solid rgba(52,211,153,0.55);
        box-shadow: 0 16px 38px rgba(22,163,74,0.8);
        backdrop-filter: blur(14px);
    }
    
    /* Prediction Text Styles */
    .big-prediction {
        font-size: 1.9rem;
        font-weight: 700;
        color: #bbf7d0; /* Light Green */
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .confidence-label {
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #a5b4fc; /* Light Blue/Purple */
    }
    
    /* Badge and Pill Styles */
    .disease-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: rgba(22,163,74,0.18);
        border: 1px solid rgba(52,211,153,0.7);
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bbf7d0;
    }
    
    /* Risk Pills */
    .risk-pill-high {
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(248, 113, 113, 0.15); /* Red Background */
        border: 1px solid rgba(248, 113, 113, 0.7); /* Red Border */
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #fecaca; /* Light Red Text */
    }
    .risk-pill-medium {
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(251, 191, 36, 0.12); /* Yellow Background */
        border: 1px solid rgba(251, 191, 36, 0.7); /* Yellow Border */
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #fef9c3; /* Light Yellow Text */
    }
    .risk-pill-low {
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(22, 163, 74, 0.15); /* Green Background */
        border: 1px solid rgba(52, 211, 153, 0.7); /* Green Border */
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bbf7d0; /* Light Green Text */
    }
    
    /* Protocol Section Title */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #bfdbfe; /* Light Blue */
        margin-bottom: 0.35rem;
    }
    
    /* Footer Note */
    .footer-note {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 1rem;
        text-align: right;
    }
    
    /* Custom Streamlit Progress Bar Styling for better contrast */
    .stProgress > div > div > div > div {
        background-color: #10b981; /* Custom Green color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar: Branding
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding-bottom: 0.8rem;">
            <div style="
                font-size: 2.2rem;
                line-height: 1;
                margin-bottom: 0.25rem;
            ">
                🌱
            </div>
            <div style="font-weight: 800; font-size: 1rem; letter-spacing: 0.16em; text-transform: uppercase;">
                Plant Disease Classifier
            </div>
            <div style="font-size: 0.8rem; color: #9ca3af; letter-spacing: 0.16em; text-transform: uppercase; margin-top: 0.25rem;">
                Vision + AI + Agronomy
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### 👥 Project Info")
    st.markdown("- **Model:** CNN (Keras/TensorFlow)")
    st.markdown("- **Dataset:** PlantVillage / MobileNetV2")

    st.markdown("---")
    st.caption("Prototype. Not a substitute for expert agronomic advice.")

# ============================================================
# Disease protocols (Expanded Dictionary)
# ============================================================
DISEASE_PROTOCOLS = {
    # --- Apple Diseases ---
    "Apple___Apple_scab": {
        "symptoms": [
            "Olive-green to brown velvety spots on young leaves and fruit.",
            "Scab lesions on fruit become black, rough, and corky.",
            "Can lead to premature leaf drop and misshapen fruit.",
        ],
        "causes": [
            "Fungal pathogen *Venturia inaequalis*.",
            "Favored by cool, wet conditions (rain and moisture on leaves).",
            "Overwinters in fallen leaves.",
        ],
        "solutions": [
            "Rake and destroy fallen leaves in autumn.",
            "Use fungicides preventatively, starting at bud break, as per local spray schedule.",
            "Plant resistant apple varieties (e.g., Liberty, Prima).",
            "Prune trees to improve air circulation and speed drying time.",
        ],
    },
    "Apple___Black_rot": {
        "symptoms": [
            "Small, purplish spots on leaves that enlarge to bull's-eye shaped lesions (Frog-eye leaf spot).",
            "Mummy fruit (shriveled, black fruit) may remain on the tree.",
            "Rotting fruit starts as a brown spot around a wound, turning dark and firm.",
        ],
        "causes": [
            "Fungal pathogen *Botryosphaeria obtusa*.",
            "Enters through wounds, pruning cuts, or dead wood.",
            "Common in neglected orchards.",
        ],
        "solutions": [
            "Remove all dead wood, cankers, and mummy fruit during dormant pruning.",
            "Protectants like Captan can be used during the growing season.",
            "Avoid wounding trees; sanitize pruning tools.",
        ],
    },
    "Apple___Cedar_apple_rust": {
        "symptoms": [
            "Bright yellow-orange spots on apple leaves, which turn red/orange later.",
            "Small tube-like structures (aeciospores) may form on the underside of leaves.",
            "Infection requires proximity to Cedar or Juniper trees (alternate host).",
        ],
        "causes": [
            "Fungal pathogen *Gymnosporangium juniperi-virginianae* (requires two hosts).",
            "Spores move from Juniper galls to apple trees in spring rain.",
        ],
        "solutions": [
            "Remove nearby Juniper trees (alternate host).",
            "Apply fungicides (e.g., myclobutanil) beginning when Juniper galls swell.",
            "Choose resistant apple cultivars.",
        ],
    },
    
    # --- Blueberry Disease ---
    "Blueberry___healthy": {
        "symptoms": [
            "Vibrant green leaves without discoloration or lesions.",
            "Strong, upright stem growth and abundant, healthy-looking berries.",
            "No sign of wilting, premature drop, or insect damage.",
        ],
        "causes": [
            "Optimal growing conditions, proper soil pH (4.5-5.5), and balanced nutrition.",
            "Good air circulation, minimal stress, and effective pest management.",
        ],
        "solutions": [
            "Maintain soil acidity (e.g., sulfur treatments).",
            "Use mulching to conserve moisture and regulate soil temperature.",
            "Continue regular monitoring for early signs of disease.",
        ],
    },
    
    # --- Cherry Disease ---
    "Cherry___Powdery_mildew": {
        "symptoms": [
            "White, powdery growth on the surface of young leaves and shoots, often causing leaves to curl upward.",
            "Affected tissue may turn brown or purple later in the season.",
        ],
        "causes": [
            "Fungal pathogen *Podosphaera clandestina*.",
            "Favored by warm temperatures (68-80°F) and high humidity, especially shaded areas.",
        ],
        "solutions": [
            "Prune out infected shoots during the dormant season.",
            "Use fungicides (sulfur, biological controls, or conventional products) upon first sign of infection.",
            "Improve air movement through canopy management.",
        ],
    },
    
    # --- Corn Diseases ---
    "Corn___Common_rust": {
        "symptoms": [
            "Small, cinnamon-brown, powdery pustules (uredinia) on both surfaces of leaves.",
            "Pustules often erupt through the leaf surface.",
            "Severe infection can cause leaves to yellow and die prematurely.",
        ],
        "causes": [
            "Fungal pathogen *Puccinia sorghi*.",
            "Spores are blown in from the South; favored by cool, moist weather (60-75°F).",
        ],
        "solutions": [
            "Plant rust-resistant hybrids.",
            "Apply foliar fungicides early in the season if susceptible varieties are used and risk is high.",
            "Avoid excessive nitrogen, which promotes lush growth.",
        ],
    },
    "Corn___Northern_Leaf_Blight": {
        "symptoms": [
            "Long (1-6 inches), elliptical, grayish-green to tan lesions on leaves.",
            "Lesions start on lower leaves and spread up the plant.",
            "Appearance is often described as 'cigar-shaped' or 'boat-shaped'.",
        ],
        "causes": [
            "Fungal pathogen *Exserohilum turcicum*.",
            "Favored by moderate temperatures (65-80°F) and high humidity or prolonged dew periods.",
        ],
        "solutions": [
            "Select resistant hybrids.",
            "Tillage to bury corn residue can reduce initial inoculum.",
            "Foliar fungicide application is warranted for susceptible hybrids under high disease pressure.",
        ],
    },

    # --- Grape Diseases ---
    "Grape___Black_rot": {
        "symptoms": [
            "Reddish-brown circular spots on leaves, becoming black and visible on the underside.",
            "Infected fruit turn into black, shriveled 'mummies'.",
            "Small, black, oval lesions may appear on shoots and tendrils.",
        ],
        "causes": [
            "Fungal pathogen *Guignardia bidwellii*.",
            "Infection occurs under warm, wet conditions (temperatures above 50°F and rain).",
        ],
        "solutions": [
            "Remove and destroy all mummified fruit from vines and ground before spring.",
            "Use protectant fungicides (e.g., Captan, Mancozeb) from early shoot growth through bloom and post-bloom.",
            "Ensure good spray coverage and air circulation via proper pruning and training.",
        ],
    },
    "Grape___Esca_(Black_Measles)": {
        "symptoms": [
            "Tiger-striped pattern of reddish-brown spots between veins on leaves (red varieties) or yellow patterns (white varieties).",
            "Grapes of affected shoots may develop dark spots and shrivel.",
            "In older wood, a characteristic white, punky rot of the inner trunk or cordon occurs.",
        ],
        "causes": [
            "Complex of several wood-rotting fungi (e.g., *Phaeomoniella chlamydospora*).",
            "Infection occurs primarily through large pruning wounds.",
        ],
        "solutions": [
            "Prune late in the dormant season to minimize wound exposure.",
            "Treat large pruning wounds with a wood protectant or fungicidal paste.",
            "Retrain young, healthy shoots to replace symptomatic cordons/trunks (renovation).",
        ],
    },
    
    # --- Citrus Disease (Example for simplicity, as HLB is complex) ---
    "Orange___Haunglongbing_(Citrus_greening)": {
        "symptoms": [
            "Asymmetrical blotchy mottle (blotchy mottling) on leaves, resembling nutrient deficiency but irregular across the leaf vein.",
            "Small, misshapen, hard fruit that remain partially green (hence 'greening').",
            "Premature defoliation, dieback, and eventual tree decline.",
        ],
        "causes": [
            "Bacterial pathogen *Candidatus Liberibacter asiaticus*.",
            "Spread by the Asian citrus psyllid (vector).",
        ],
        "solutions": [
            "Immediate removal and destruction of infected trees (essential for disease control).",
            "Aggressive management of the Asian citrus psyllid vector using systemic insecticides.",
            "Use of certified, disease-free nursery stock.",
        ],
    },
    
    # --- Peach Disease ---
    "Peach___Bacterial_spot": {
        "symptoms": [
            "Small, water-soaked spots on young leaves that enlarge and fall out, creating a 'shot-hole' appearance.",
            "Small, dark, greasy-looking spots on fruit that can become sunken and cracked.",
            "Cankers may develop on twigs.",
        ],
        "causes": [
            "Bacterial pathogen *Xanthomonas arboricola pv. pruni*.",
            "Favored by wet, windy conditions; rain spreads bacteria.",
        ],
        "solutions": [
            "Plant resistant varieties where available.",
            "Prune trees to improve air circulation.",
            "Dormant copper sprays and in-season applications of copper or oxytetracycline may offer suppression.",
        ],
    },
    
    # --- Pepper Disease ---
    "Pepper_bell___Bacterial_spot": {
        "symptoms": [
            "Small, brown, water-soaked spots on leaves that turn dark brown and scabby.",
            "Yellowing and defoliation of lower leaves.",
            "Raised, wart-like lesions on the fruit, reducing marketability.",
        ],
        "causes": [
            "Bacterial pathogens of the *Xanthomonas* species.",
            "Spread via splashing water, wind, and contaminated seeds/transplants.",
            "Thrives in warm, humid conditions.",
        ],
        "solutions": [
            "Use certified disease-free seed and transplants.",
            "Avoid overhead irrigation, especially in the evening.",
            "Apply copper bactericides and/or mancozeb preventatively, but rotation is key.",
            "Sanitize tools and remove infected plant debris.",
        ],
    },

    # --- Potato Diseases ---
    "Potato___Late_blight": {
        "symptoms": [
            "Irregular, dark, water-soaked spots on leaves that rapidly enlarge.",
            "White, downy fungal growth may be visible on the undersides of leaves, especially in humid conditions.",
            "Tuber rot appears as reddish-brown, dry, granular decay.",
        ],
        "causes": [
            "Oomycete pathogen *Phytophthora infestans*.",
            "Spreads rapidly under cool (60-70°F) and very wet conditions.",
        ],
        "solutions": [
            "Use resistant potato cultivars.",
            "Apply fungicides preventatively, often starting early in the season and repeating under favorable weather.",
            "Destroy volunteer potatoes and cull piles.",
            "Hill plants deeply to prevent spores from washing down to tubers.",
        ],
    },
    "Potato___Early_blight": {
        "symptoms": [
            "Concentric brown rings on older leaves, often with yellow halos.",
            "Lesions usually start on lower leaves and move upward.",
            "Causes defoliation and reduces yield.",
        ],
        "causes": [
            "Fungal pathogen *Alternaria solani* under warm, humid conditions.",
            "Spread by infected debris and splashing rain or irrigation.",
        ],
        "solutions": [
            "Remove and destroy heavily infected leaves and plant residues.",
            "Use protectant fungicides as per local recommendations.",
            "Adopt crop rotation and improve airflow.",
            "Ensure plants have adequate fertility, especially nitrogen.",
        ],
    },
    
    # --- Strawberry Disease ---
    "Strawberry___Leaf_scorch": {
        "symptoms": [
            "Small, dark purple spots on the upper leaf surface.",
            "Spots merge, and the entire leaf surface may appear scorched, turning reddish-purple or brown, starting from the margins.",
            "Severe infection can weaken the plant and reduce yield.",
        ],
        "causes": [
            "Fungal pathogen *Diplocarpon earlianum*.",
            "Favored by moderate temperatures and extended periods of leaf wetness.",
            "Overwinters on infected leaves.",
        ],
        "solutions": [
            "Remove old, infected leaves (mow and remove post-harvest).",
            "Apply fungicides (e.g., Captan, Switch) when new growth begins and as disease pressure increases.",
            "Use cultivars known for better resistance.",
            "Mulch properly to reduce splash dispersal of spores.",
        ],
    },

    # --- Tomato Diseases ---
    "Tomato___Bacterial_spot": {
        "symptoms": [
            "Small, angular, water-soaked spots on leaves that turn black and slightly greasy.",
            "Severe infection causes leaves to yellow and drop prematurely.",
            "Small, raised, scabby spots on green fruit.",
        ],
        "causes": [
            "Several species of *Xanthomonas* bacteria.",
            "High humidity and rainfall are critical for disease development and spread.",
            "Can be seed-borne.",
        ],
        "solutions": [
            "Use certified, disease-free seed and transplants.",
            "Avoid working in fields when foliage is wet.",
            "Apply copper-based bactericides in rotation with other compounds.",
            "Manage weeds and promote good airflow.",
        ],
    },
    "Tomato___Late_blight": {
        "symptoms": [
            "Large, greasy-looking, black or brown lesions on leaves and stems.",
            "White, fuzzy growth visible on the underside of leaves during cool, moist periods.",
            "Fruit turn brown, firm, and develop irregular, greasy-looking lesions.",
        ],
        "causes": [
            "Oomycete pathogen *Phytophthora infestans* (same as Potato Late Blight).",
            "Highly aggressive and rapidly destructive under cool, wet conditions.",
        ],
        "solutions": [
            "Scout regularly and destroy infected plants immediately.",
            "Apply aggressive fungicide spray program using targeted products (e.g., mefenoxam, chlorothalonil).",
            "Isolate tomato fields from potato fields if possible.",
        ],
    },
    "Tomato___Yellow_Leaf_Curl_Virus": {
        "symptoms": [
            "Severe upward curling and cupping of young leaves.",
            "Stunting of the entire plant; leaves are small and pale green/yellow (chlorotic).",
            "Flowers drop and few or no normal-sized fruit are produced.",
        ],
        "causes": [
            "Tomato Yellow Leaf Curl Virus (TYLCV).",
            "Transmitted exclusively by the silverleaf whitefly (*Bemisia tabaci*).",
        ],
        "solutions": [
            "Use resistant tomato cultivars (e.g., 'Bella Rosa', 'Celebrity Plus').",
            "Manage whiteflies aggressively with insecticides and biological controls.",
            "Use reflective mulch or row covers to deter whiteflies early in the season.",
            "Remove and destroy infected plants immediately.",
        ],
    },
    "Tomato___healthy": {
        "symptoms": [
            "Deep green, flat, and full leaves with no visible spots or discoloration.",
            "Strong, upright stem with healthy fruit set.",
            "Absence of wilting, premature yellowing, or signs of insect damage.",
        ],
        "causes": [
            "Adequate fertilization, consistent water management, and ideal growing temperatures.",
            "Use of resistant varieties and good cultural practices.",
        ],
        "solutions": [
            "Maintain balanced soil nutrition and monitor pH.",
            "Ensure proper pruning and staking for good air circulation.",
            "Continue regular scouting for pests and diseases, especially during high-risk weather.",
        ],
    },
    
    # Default protocol for unlisted or healthy classes not explicitly defined
    "Apple___healthy": { **DISEASE_PROTOCOLS.get("Tomato___healthy", {}) },
    "Cherry___healthy": { **DISEASE_PROTOCOLS.get("Tomato___healthy", {}) },
    "Corn___Cercospora_leaf_spot": { **DISEASE_PROTOCOLS.get("Corn___Northern_Leaf_Blight", {}) },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": { **DISEASE_PROTOCOLS.get("Grape___Black_rot", {}) },
    "Peach___healthy": { **DISEASE_PROTOCOLS.get("Tomato___healthy", {}) },
    "Pepper_bell___healthy": { **DISEASE_PROTOCOLS.get("Tomato___healthy", {}) },
    "Potato___healthy": { **DISEASE_PROTOCOLS.get("Tomato___healthy", {}) },
    "Raspberry___healthy": { **DISEASE_PROTOCOLS.get("Tomato___healthy", {}) },
    "Soybean___healthy": { **DISEASE_PROTOCOLS.get("Tomato___healthy", {}) },
    "Squash___Powdery_mildew": { **DISEASE_PROTOCOLS.get("Cherry___Powdery_mildew", {}) },
    "Strawberry___healthy": { **DISEASE_PROTOCOLS.get("Tomato___healthy", {}) },
    "Tomato___Leaf_Mold": { 
        "symptoms": [
            "Yellowing patches on upper leaf surface, corresponding to olive-green or grayish-purple fungal growth on the underside.",
            "Leaves curl, turn brown, and die.",
            "Fruit infection is rare but causes black, leathery spots on the stem end.",
        ],
        "causes": [
            "Fungal pathogen *Passalora fulva*.",
            "Favored by high humidity (90-100%) and moderate temperatures (68-77°F); most common in greenhouses or dense plantings.",
        ],
        "solutions": [
            "Increase ventilation and reduce humidity in enclosed spaces.",
            "Remove infected lower leaves to improve airflow.",
            "Use resistant varieties.",
            "Fungicides are effective if applied early and regularly.",
        ],
    },
    "Tomato___Septoria_leaf_spot": { 
        "symptoms": [
            "Numerous small, circular spots (1/16 to 1/4 inch) on older leaves.",
            "Spots have gray or tan centers with dark borders.",
            "Tiny black specks (pycnidia) are visible in the center of the spots.",
        ],
        "causes": [
            "Fungal pathogen *Septoria lycopersici*.",
            "Spread by splashing water from infected plant debris.",
            "Favored by prolonged periods of rain or overhead watering.",
        ],
        "solutions": [
            "Stake plants and use mulches to keep foliage off the ground.",
            "Water at the base of the plant (drip irrigation preferred).",
            "Fungicide sprays (e.g., chlorothalonil) are highly effective if started early.",
            "Rotate crops away from tomatoes for at least three years.",
        ],
    },
    "Tomato___Spider_mites": {
        "symptoms": [
            "Tiny yellow or white speckles (stippling) on leaves.",
            "Fine webbing visible on the undersides of leaves or between leaves and stems.",
            "Heavy infestations cause leaves to bronze, yellow, and drop.",
        ],
        "causes": [
            "Pest: Two-spotted spider mites (*Tetranychus urticae*).",
            "Favored by hot, dry weather; population explosions occur quickly.",
        ],
        "solutions": [
            "Release beneficial insects (e.g., predatory mites).",
            "Spray plants with strong jets of water to dislodge mites (especially undersides of leaves).",
            "Use miticides or insecticidal soaps/oils as directed.",
            "Keep dust down, as it favors mites.",
        ],
    },
    "Tomato___Target_Spot": {
        "symptoms": [
            "Small, dark, water-soaked spots on leaves that develop tan centers and characteristic concentric rings (target-like).",
            "Severe infection leads to leaf drop and sunscald on fruit.",
        ],
        "causes": [
            "Fungal pathogen *Corynespora cassiicola*.",
            "Spread by wind and rain, thriving in warm, humid conditions.",
        ],
        "solutions": [
            "Use resistant cultivars.",
            "Remove infected debris and volunteer plants.",
            "Fungicides used for Early Blight often provide control for Target Spot.",
        ],
    },
    "Tomato___Mosaic_virus": {
        "symptoms": [
            "Mottled appearance of light and dark green on the leaves (mosaic pattern).",
            "Leaves may be distorted, crinkled, or fern-like.",
            "Stunting of the plant and poor fruit development.",
        ],
        "causes": [
            "Tobacco Mosaic Virus (TMV) or Tomato Mosaic Virus (ToMV).",
            "Highly infectious; spread mechanically by handling plants, contaminated tools, or tobacco use.",
        ],
        "solutions": [
            "Use resistant varieties.",
            "Remove and destroy infected plants immediately.",
            "Sanitize hands and tools frequently (use milk or bleach solution).",
            "Avoid handling plants after smoking or using tobacco products.",
        ],
    },
}

DEFAULT_PROTOCOL = {
    "symptoms": [
        "Visual symptoms vary with crop and pathogen.",
        "Look for spots, lesions, discoloration, wilting, or abnormal growth.",
    ],
    "causes": [
        "Can be caused by fungal, bacterial, viral, or nutrient-related problems.",
        "Weather conditions, irrigation practices, and soil health strongly influence disease risk.",
    ],
    "solutions": [
        "Scout fields regularly and compare symptoms with trusted references.",
        "Consult local agricultural extension or agronomists for precise diagnosis.",
        "Use integrated management: resistant varieties, crop rotation, sanitation, and labeled crop-protection products.",
    ],
}

# ============================================================
# Utility: preprocessing, prediction, farm data
# ============================================================
def preprocess_image(uploaded_file, target_size=(224, 224)):
    """Preprocesses the uploaded image for model input."""
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32") / 255.0
    # Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    return image, img_batch

def predict(model, img_batch, class_names):
    """Runs prediction and extracts top 3 results."""
    # Ensure model is not None before prediction
    if model is None:
        raise ValueError("Model is not loaded. Check model download status.")
        
    preds = model.predict(img_batch)
    preds = np.squeeze(preds)
    
    # Handle case where preds might be a scalar (e.g., binary classification)
    if preds.ndim == 0:
        # Assuming multi-class output is desired for consistency
        # For a true binary model, this logic might need adjustment
        preds = np.array([1 - preds, preds])
        
    # Apply softmax if needed (if model output is logits)
    # Since the original code assumes non-normalized output, we normalize here:
    probs = np.exp(preds) / np.sum(np.exp(preds)) 

    top_indices = np.argsort(probs)[::-1][:3]
    top_probs = probs[top_indices]
    top_classes = [class_names[i] for i in top_indices]

    primary_class = top_classes[0]
    primary_confidence = float(top_probs[0] * 100)

    differential = list(
        zip(
            top_classes,
            [float(p * 100) for p in top_probs],
        )
    )

    return primary_class, primary_confidence, differential

def generate_farm_health_data(days=7):
    """Generates synthetic farm environment data for the dashboard."""
    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(days)][::-1]

    # Simulated fluctuations
    np.random.seed(42) # For reproducibility of simulation
    humidity = np.clip(np.random.normal(loc=78, scale=8, size=days), 40, 100)
    temperature = np.clip(np.random.normal(loc=27, scale=3, size=days), 15, 40)
    soil_ph = np.clip(np.random.normal(loc=6.4, scale=0.3, size=days), 4.5, 8.5)

    df = pd.DataFrame(
        {
            "Date": [d.strftime("%d-%b") for d in dates],
            "Humidity (%)": humidity,
            "Temperature (°C)": temperature,
            "Soil pH": soil_ph,
        }
    )
    df.set_index("Date", inplace=True)
    return df

def assess_risk(humidity_series, temperature_series, ph_series):
    """Assesses general disease risk based on simulated environmental averages."""
    avg_h = float(np.mean(humidity_series))
    avg_t = float(np.mean(temperature_series))
    avg_p = float(np.mean(ph_series))

    if avg_h >= 80 and 20 <= avg_t <= 30:
        level = "HIGH"
        reason = "sustained high humidity and moderate temperatures, which favour foliar fungal diseases (e.g., Early Blight, Late Blight)."
    elif avg_h >= 70 and 18 <= avg_t <= 32:
        level = "MEDIUM"
        reason = "elevated humidity with temperatures suitable for several leaf diseases. Monitor closely."
    else:
        level = "LOW"
        reason = "conditions are relatively less favourable for major foliar fungal pathogens. Focus on cultural control."

    ph_comment = ""
    if avg_p < 5.5:
        ph_comment = " Soil tends to be acidic; consider liming based on soil-test recommendations for optimal nutrient uptake."
    elif avg_p > 7.5:
        ph_comment = " Soil tends to be slightly alkaline; monitor micronutrient availability (e.g., Iron, Zinc)."

    return level, reason + ph_comment

# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="glass-header">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
            <div>
                <div style="font-size:0.8rem;letter-spacing:0.24em;text-transform:uppercase;color:#d1fae5;">
                    Plant Disease Detection
                </div>
                <div style="font-size:1.45rem;font-weight:800;margin-top:0.15rem;color:#f9fafb;">
                    Image-based Crop Disease Classifier
                </div>
                <div style="font-size:0.9rem;margin-top:0.4rem;color:#e5e7eb;max-width:540px;">
                    Upload a plant or leaf image, obtain AI predictions with confidence scores, 
                    explore treatment protocols, and view a sample farm health dashboard.
                </div>
            </div>
            <div style="text-align:right;min-width:180px;margin-top:0.8rem;">
                <span class="disease-badge">
                    🔬 CNN Classifier
                </span>
                <br/>
                <span class="disease-badge" style="margin-top:0.45rem;">
                    🌿 Integrated Management
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

tabs = st.tabs(
    [
        "🔎 AI Diagnosis & Classifier",
        "🌿 Treatment Protocol",
        "📊 Farm Health Monitor",
    ]
)

# Load model and classes once
model_obj, class_names = load_model_and_classes()

# ============================================================
# Tab 1: AI Diagnosis & Classifier
# ============================================================
with tabs[0]:
    st.markdown("### 🔎 AI Diagnosis & Classifier")

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.markdown("#### Upload Leaf / Plant Image")
        uploaded_file = st.file_uploader(
            "Drag & drop or browse (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            help="Use clear, focused images with the symptomatic area visible.",
        )

    with col_right:
        st.markdown("#### Model Summary")
        
        # Display model stats
        if model_obj:
            st.metric(
                label="Number of Classes",
                value=len(class_names),
                help="Total plant disease and healthy classes supported.",
            )
            st.metric(
                label="Input Size",
                value="224 × 224 × 3",
                help="Standard RGB size used during preprocessing.",
            )
        else:
            st.error("Model failed to load. Please check file access.")


    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if uploaded_file is not None and model_obj is not None:
        try:
            with st.spinner("Processing image and running classifier..."):
                time.sleep(0.6)
                image, img_batch = preprocess_image(uploaded_file)
                prediction, confidence, diff_list = predict(model_obj, img_batch, class_names)

            img_col, result_col = st.columns([1.2, 1])
            with img_col:
                st.markdown("##### Input Image")
                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_column_width=True,
                )

            with result_col:
                st.markdown("##### Prediction")
                st.markdown(
                    f"""
                    <div style="
                        border-radius:16px;
                        padding:1.0rem 1.2rem;
                        background:linear-gradient(135deg, rgba(34,197,94,0.16), rgba(22,163,74,0.75));
                        border:1px solid rgba(34,197,94,0.85);
                        box-shadow:0 18px 40px rgba(22,163,74,0.85);
                        text-align:center;
                    ">
                        <div style="font-size:0.85rem;letter-spacing:0.18em;text-transform:uppercase;color:#dcfce7;">
                            Primary Class
                        </div>
                        <div class="big-prediction" style="margin-top:0.35rem;">
                            {prediction}
                        </div>
                        <div style="margin-top:0.8rem;">
                            <span class="confidence-label">Confidence:</span>
                            <div style="font-size:1.6rem;font-weight:700;margin-top:0.1rem;color:#fefce8;">
                                {confidence:.2f}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(
                        label="Prediction Confidence",
                        value=f"{confidence:.1f} %",
                    )
                with c2:
                    st.metric(
                        label="Top‑3 Scores",
                        value=f"{diff_list[0][1]:.1f}% / {diff_list[1][1]:.1f}% / {diff_list[2][1]:.1f}%",
                        help="Confidence of three most likely classes.",
                    )

            st.write("")
            st.markdown("##### Differential Diagnosis (Top 3)")

            for idx, (cls_name, score) in enumerate(diff_list, start=1):
                # Clamp score between 0 and 100 before normalizing
                safe_score = min(max(score, 0.0), 100.0) 
                progress_val = safe_score / 100.0
                st.progress(
                    progress_val,
                    text=f"{idx}. {cls_name} — {safe_score:.2f}%",
                )
                time.sleep(0.03)

            st.markdown(
                """
                <div style="font-size:0.8rem;color:#9ca3af;margin-top:0.4rem;">
                    Use the top‑3 suggestions along with field history and expert consultation for reliable decisions.
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please ensure the uploaded file is a valid image and the model loaded correctly.")
    elif model_obj is None:
        st.error("Model is unavailable. Please check the log for download errors.")
    else:
        st.info(
            "Upload a plant or leaf image on the left to generate predictions and confidence scores."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Tab 2: Treatment Protocol (simplified)
# ============================================================
with tabs[1]:
    st.markdown("### 🌿 Integrated Treatment Protocol")

    st.markdown(
        """
        Select any predicted class to view typical symptoms, causes, and integrated management options.
        """
    )
    
    # Determine default index based on loading results
    default_index = 0
    if "Tomato___Early_blight" in class_names:
        try:
            default_index = class_names.index("Tomato___Early_blight")
        except ValueError:
            pass
            
    selected_disease = st.selectbox(
        "Choose a class / disease",
        options=sorted(class_names),
        index=default_index,
    )

    protocol = DISEASE_PROTOCOLS.get(selected_disease, DEFAULT_PROTOCOL)

    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1.2])

    with col1:
        st.markdown(
            f'<div class="section-title">🩺 {selected_disease.replace("_", " ").replace("___", " - ")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Symptoms (Field/Visual)**")
        st.markdown("\n".join([f"- {item}" for item in protocol["symptoms"]]))
        st.markdown("")
        st.markdown("**Causes & Favouring Conditions**")
        st.markdown("\n".join([f"- {item}" for item in protocol["causes"]]))

    with col2:
        st.markdown(
            '<div class="section-title">🧪 Management Recommendations</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Suggested Actions (Integrated Pest Management)**")
        st.markdown("\n".join([f"- {item}" for item in protocol["solutions"]]))

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer-note">
            Always verify product labels, safety intervals, and local recommendations when using pesticides or fungicides.
            This information is for educational purposes only.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Tab 3: Farm Health Monitor
# ============================================================
with tabs[2]:
    st.markdown("### 📊 Farm Health Monitor (Simulated)")

    st.markdown(
        """
        Example of a 7‑day IoT style feed for humidity, temperature, and soil pH.
        This provides context for disease risk.
        """
    )

    df_env = generate_farm_health_data(days=7)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Plot the environmental data
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Humidity (%)**")
        st.line_chart(df_env["Humidity (%)"], color="#22c55e")
    with c2:
        st.markdown("**Temperature (°C)**")
        st.line_chart(df_env["Temperature (°C)"], color="#f59e0b")
    with c3:
        st.markdown("**Soil pH**")
        st.line_chart(df_env["Soil pH"], color="#6366f1")

    # Assess overall risk
    level, reason = assess_risk(
        df_env["Humidity (%)"], df_env["Temperature (°C)"], df_env["Soil pH"]
    )

    # Set styling based on risk level
    if level == "HIGH":
        risk_class = "risk-pill-high"
        warn_type = st.error
        title = "HIGH FUNGAL RISK"
        icon = "⚠️"
    elif level == "MEDIUM":
        risk_class = "risk-pill-medium"
        warn_type = st.warning
        title = "MODERATE DISEASE RISK"
        icon = "⚠️"
    else:
        risk_class = "risk-pill-low"
        warn_type = st.info
        title = "LOW DISEASE RISK"
        icon = "✅"

    st.write("")
    
    # Use the native Streamlit warning/error for accessibility and standard look
    warn_type(
        f"**Risk Summary:** {reason}",
        icon=icon,
    )
    
    # Display the styled risk pill
    st.markdown(
        f"""
        <div style="margin-top:0.5rem;display:flex;align-items:center;">
            <span class="{risk_class}">
                {title}
            </span>
            <span style="font-size:0.82rem;color:#e5e7eb;margin-left:0.6rem;">
                Use such data to plan irrigation, fungicide sprays, and canopy management.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "Values shown are synthetic examples, intended only to demonstrate dashboard functionality."
    )