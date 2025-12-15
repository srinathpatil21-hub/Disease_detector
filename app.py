import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import gdown
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix
import time

# =============================================================================
# CONFIGURATION - CUSTOMIZE HERE
# =============================================================================
MODEL_URL = "https://drive.google.com/uc?id=1DAWfGqtpzGT9khOvH79McW4SG_UBheft"
MODEL_PATH = "cnn_model.h5"
IMG_SIZE = (224, 224)  # Change if your model uses different size
CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]  # UPDATE YOUR CLASSES HERE
CONFIDENCE_THRESHOLD = 70.0

# =============================================================================
# MODEL LOADING & CACHING
# =============================================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🚀 Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# =============================================================================
# IMAGE PROCESSING
# =============================================================================
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def enhance_image(image, contrast=1.2, sharpness=1.1):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(sharpness)

# =============================================================================
# PREDICTION ENGINE
# =============================================================================
def predict_image(model, image):
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)[0]
    confidence = np.max(prediction) * 100
    predicted_class = np.argmax(prediction)
    return predicted_class, confidence, prediction

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(
    page_title="🧠 AI Image Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 AI Image Classifier")
st.markdown("Upload image → Get instant predictions → Analyze results")

# Load model
try:
    model = load_model()
    st.success(f"✅ Model loaded! Ready for {len(CLASS_NAMES)} classes")
except Exception as e:
    st.error(f"❌ Model error: {str(e)}")
    st.stop()

# =============================================================================
# SIDEBAR - CONTROLS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Image upload
    uploaded_file = st.file_uploader("📁 Upload Image", type=['png','jpg','jpeg'])
    
    # Enhancement options
    use_enhancement = st.checkbox("✨ Auto-enhance image", value=True)
    contrast = st.slider("Contrast", 0.8, 1.5, 1.2, 0.1)
    sharpness = st.slider("Sharpness", 0.8, 1.5, 1.1, 0.1)
    
    # Prediction settings
    st.header("🎯 Prediction")
    show_top_k = st.slider("Show top predictions", 1, 5, 3)
    min_confidence = st.slider("Min confidence %", 50, 95, CONFIDENCE_THRESHOLD)

# =============================================================================
# MAIN CONTENT
# =============================================================================
if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Original Image")
        if use_enhancement:
            enhanced_image = enhance_image(image, contrast, sharpness)
            st.image(enhanced_image, caption="Enhanced", use_column_width=True)
        else:
            st.image(image, caption="Original", use_column_width=True)
    
    # Prediction
    with col2:
        with st.spinner("🔮 Predicting..."):
            pred_class, confidence, probs = predict_image(model, image)
            top_classes = np.argsort(probs)[-show_top_k:][::-1]
        
        # Main prediction
        st.metric(
            label="🎯 Prediction", 
            value=f"{CLASS_NAMES[pred_class]}",
            delta=f"{confidence:.1f}%"
        )
        
        color = "🟢" if confidence >= min_confidence else "🟡"
        st.markdown(f"**Confidence:** {color} {confidence:.1f}%")
        
        if confidence < min_confidence:
            st.warning("⚠️ Low confidence - try another image!")
    
    # Results table
    st.subheader("📊 Top Predictions")
    results = []
    for i, cls_idx in enumerate(top_classes):
        prob = probs[cls_idx] * 100
        results.append({
            "Rank": i+1,
            "Class": CLASS_NAMES[cls_idx],
            "Probability": f"{prob:.1f}%",
            "Confidence": prob
        })
    
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    with col1:
        # Confidence bar
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[CLASS_NAMES[pred_class]], y=[confidence],
            marker_color=['#10B981'], text=f"{confidence:.1f}%",
            textposition='outside'
        ))
        fig_bar.update_layout(height=250, showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Probability pie
        fig_pie = px.pie(
            df_results.head(5), values='Confidence', names='Class',
            color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Model info
    with st.expander("ℹ️ Model Information"):
        st.info(f"""
        **Model Details:**
        - Classes: {', '.join(CLASS_NAMES)}
        - Input Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}
        - Top Prediction: {CLASS_NAMES[pred_class]} ({confidence:.1f}%)
        - Status: {'✅ High Confidence' if confidence >= min_confidence else '⚠️ Review Required'}
        """)

else:
    # Welcome screen
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("🚀 How to use")
        st.markdown("""
        1. **Upload** image in sidebar
        2. **Adjust** enhancement settings
        3. **Get** instant predictions
        4. **Analyze** confidence scores
        """)
    
    with col2:
        st.header("✅ Features")
        st.markdown("""
        - Auto model download
        - Image enhancement
        - Confidence analysis
        - Top-K predictions
        - Interactive charts
        - Mobile friendly
        """)

# Footer
st.markdown("---")
st.markdown("**🤖 Made with Streamlit | Deploy: GitHub Codespaces/Streamlit Cloud**")

# Auto-refresh button
if st.sidebar.button("🔄 Reload Model"):
    st.cache_resource.clear()
    st.rerun()
