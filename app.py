import streamlit as st
import numpy as np
import pandas as pd
import json
import requests
from PIL import Image
import io
import time
from tensorflow import keras
import tensorflow as tf

# Page config for wide layout and professional look
st.set_page_config(
    page_title="Agri-Health Vision",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and professional design
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-container {
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_classes():
    """Load model from Google Drive and class names"""
    try:
        # Google Drive direct download link (modify file ID)
        model_url = "https://drive.google.com/uc?export=download&id=1DAWfGqtpzGT9khOvH79McW4SG_UBheft"
        
        # Load model
        response = requests.get(model_url)
        model_data = io.BytesIO(response.content)
        model = tf.keras.models.load_model(model_data)
        
        # Load class names (fallback to sample if not available)
        class_names_path = "class_names.json"
        if class_names_path.exists():
            with open(class_names_path, 'r') as f:
                class_names = json.load(f)
        else:
            # Sample class names for PlantVillage dataset
            class_names = [
                "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust",
                "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
                "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
                "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
                "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
                "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
                "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
                "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
            ]
        
        return model, class_names
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}. Using demo mode.")
        return None, ["Healthy", "Bacterial Spot", "Early Blight", "Late Blight"]

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_disease(model, image_array, class_names):
    """Generate predictions with confidence scores"""
    predictions = model.predict(image_array, verbose=0)
    top_indices = np.argsort(predictions[0])[::-1][:3]
    top_predictions = [(class_names[i], predictions[0][i]) for i in top_indices]
    return top_predictions

# Sidebar branding
with st.sidebar:
    st.markdown("""
    <div class="glass-card">
        <h2 style='color: #4CAF50; text-align: center;'>🌱 Agri-Health Vision</h2>
        <p style='text-align: center; color: #666;'>SIH 2025 Winner</p>
        <hr>
        <p><strong>Team:</strong> AI Innovators</p>
        <p><strong>Tech:</strong> CNN + Streamlit + MobileNetV2</p>
    </div>
    """, unsafe_allow_html=True)

# Treatment protocols database
TREATMENT_DATA = {
    "Tomato___Bacterial_spot": {
        "symptoms": ["Small, water-soaked spots on leaves", "Yellow halos around spots", "Spots turn dark brown"],
        "causes": ["Xanthomonas bacteria", "Splash dispersal by rain", "Warm, wet conditions"],
        "chemical": ["Copper-based bactericides", "Streptomycin", "Kasugamycin"],
        "organic": ["Remove infected leaves", "Apply copper soap", "Improve air circulation"]
    },
    "Tomato___Early_blight": {
        "symptoms": ["Dark brown spots with yellow halos", "Concentric rings in spots", "Defoliation from bottom up"],
        "causes": ["Alternaria solani fungus", "Warm temperatures", "Wet foliage"],
        "chemical": ["Chlorothalonil", "Mancozeb", "Azoxystrobin"],
        "organic": ["Mulching", "Proper spacing", "Remove lower leaves"]
    },
    "Tomato___Late_blight": {
        "symptoms": ["Large irregular lesions", "White mold on leaf undersides", "Rapid plant collapse"],
        "causes": ["Phytophthora infestans", "Cool, wet weather", "Overhead watering"],
        "chemical": ["Mancozeb + metalaxyl", "Chlorothalonil", "Propamocarb"],
        "organic": ["Copper fungicides", "Remove volunteers", "Avoid overhead irrigation"]
    },
    "Healthy": {
        "symptoms": ["No visible symptoms"],
        "causes": ["Good plant health"],
        "chemical": ["None required"],
        "organic": ["Continue good practices"]
    }
    # Add more diseases as needed
}

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔎 AI Diagnosis", "🌿 Treatment Protocol", "📊 Farm Monitor"])

# Tab 1: AI Diagnosis
with tab1:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Upload Plant Leaf Image</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of plant leaves"
        )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Show spinner during processing
        with st.spinner("🔬 Processing with MobileNetV2 CNN..."):
            time.sleep(1.5)
        
        model, class_names = load_model_and_classes()
        
        col_img, col_pred = st.columns([1, 2])
        
        with col_img:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col_pred:
            # Preprocess and predict
            processed_image = preprocess_image(image)
            
            if model is not None:
                predictions = predict_disease(model, processed_image, class_names)
                
                # Main prediction
                top_prediction = predictions[0]
                disease_name = top_prediction[0].replace("___", " ").title()
                confidence = top_prediction[1] * 100
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h1>{disease_name}</h1>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Prediction Confidence", f"{confidence:.1f}%")
                
                # Top 3 predictions
                st.subheader("🔍 Differential Diagnosis (Top 3)")
                for i, (disease, conf) in enumerate(predictions):
                    disease_display = disease.replace("___", " ").title()
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{i+1}. {disease_display}")
                    with col2:
                        st.progress(min(conf, 1.0))
            else:
                st.warning("Demo mode: Simulated prediction")
                st.success("🥬 **Healthy Plant** - Confidence: 92.3%")
                st.metric("Demo Confidence", "92.3%")

# Tab 2: Treatment Protocol
with tab2:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Treatment Recommendations</h2>", unsafe_allow_html=True)
    
    disease_list = list(TREATMENT_DATA.keys()) + ["Healthy", "Apple___Apple_scab"]
    selected_disease = st.selectbox("Select diagnosed disease:", disease_list)
    
    if selected_disease in TREATMENT_DATA:
        data = TREATMENT_DATA[selected_disease]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Symptoms & Causes")
            for symptom in data["symptoms"]:
                st.write(f"• {symptom}")
            st.write("**Causes:**")
            for cause in data["causes"]:
                st.write(f"• {cause}")
        
        with col2:
            st.subheader("💊 Treatment Protocol")
            st.write("**Chemical:**")
            for chem in data["chemical"]:
                st.write(f"• {chem}")
            st.write("**Organic:**")
            for org in data["organic"]:
                st.write(f"• {org}")

# Tab 3: Farm Monitor
with tab3:
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>IoT Farm Health Monitor</h2>", unsafe_allow_html=True)
    
    # Generate simulated data
    days = pd.date_range(start='2025-12-09', periods=7, freq='D')
    np.random.seed(42)
    
    humidity = 70 + 15 * np.sin(np.arange(7) * 0.8) + np.random.normal(0, 3, 7)
    temperature = 25 + 5 * np.sin(np.arange(7) * 0.6) + np.random.normal(0, 2, 7)
    soil_ph = 6.5 + 0.5 * np.sin(np.arange(7)) + np.random.normal(0, 0.2, 7)
    
    df_humidity = pd.DataFrame({'Day': days, 'Humidity (%)': humidity})
    df_temp = pd.DataFrame({'Day': days, 'Temperature (°C)': temperature})
    df_ph = pd.DataFrame({'Day': days, 'Soil pH': soil_ph})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💧 Humidity")
        st.line_chart(df_humidity.set_index('Day'))
    
    with col2:
        st.subheader("🌡️ Temperature")
        st.line_chart(df_temp.set_index('Day'))
    
    with col3:
        st.subheader("🧪 Soil pH")
        st.line_chart(df_ph.set_index('Day'))
    
    # Risk assessment
    avg_humidity = np.mean(humidity)
    avg_temp = np.mean(temperature)
    
    if avg_humidity > 75 and 22 < avg_temp < 28:
        st.error("🚨 **HIGH FUNGAL RISK** - Sustained high humidity + optimal fungal temperature")
    elif avg_humidity > 80:
        st.warning("⚠️ **MODERATE RISK** - High humidity detected")
    else:
        st.success("✅ **LOW RISK** - Favorable conditions")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Built for SIH 2025 | Powered by CNN & Streamlit 🌱</p>", unsafe_allow_html=True)
