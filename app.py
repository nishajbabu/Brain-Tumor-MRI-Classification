import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os

# ==============================================================================
# Page Configuration & UI Theme
# ==============================================================================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clinical Dashboard
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #F3F4F6;
        border-left: 6px solid #2563EB;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Load Model & Class Labels (Cached for Speed)
# ==============================================================================
@st.cache_resource
def load_classification_model():
    model_path = 'best_brain_tumor_model.h5'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found in current directory!")
        return None
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_data
def load_class_labels():
    pkl_path = 'label_mapping.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            labels = pickle.load(f)
        return labels
    # Fallback to standard classes if pickle not found
    return {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary'}

model = load_classification_model()
class_labels = load_class_labels()

# Clean display names and descriptions
label_descriptions = {
    'glioma': "Glioma Tumor — An intracranial tumor originating from glial cells of the central nervous system.",
    'meningioma': "Meningioma Tumor — Typically benign mass arising from the meningeal membranes surrounding brain tissue.",
    'pituitary': "Pituitary Tumor — An abnormal lesion in the pituitary gland affecting endocrine hormonal regulation.",
    'no_tumor': "Healthy Control (No Tumor) — No pathological cranial lesion or neoplastic mass detected."
}

# ==============================================================================
# Sidebar Information
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=70)
    st.title("About This System")
    st.info(
        "**Diagnostic AI Engine:** MobileNetV2 Transfer Learning\n\n"
        "**Trained On:** 2,443 Labeled Multi-Class MRI Scans\n\n"
        "**Overall Test Accuracy:** 87.40%\n\n"
        "**Target Pathologies:**\n"
        "- Glioma\n"
        "- Meningioma\n"
        "- Pituitary\n"
        "- Healthy / No Tumor"
    )
    st.warning("⚠️ **Medical Disclaimer:** This system is an AI diagnostic support tool intended for screening and second opinions. All outputs must be validated by a licensed radiologist.")

# ==============================================================================
# Main Dashboard UI
# ==============================================================================
st.markdown('<div class="main-title">🧠 Brain Tumor MRI Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated AI-assisted diagnostic classification of brain MRI scans</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Brain MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an MRI image file (JPEG, JPG, or PNG)",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Patient MRI Scan", use_container_width=True)

with col2:
    st.subheader("🔍 Diagnostic Inference")
    
    if uploaded_file is None:
        st.write("👈 *Please upload an MRI image on the left panel to begin analysis.*")
    else:
        if model is None:
            st.error("Model could not be loaded. Please ensure 'best_brain_tumor_model.h5' is placed in this folder.")
        else:
            with st.spinner("Processing MRI scan through Deep Neural Network..."):
                # Preprocess: Resize to (224, 224) and normalize to [0, 1]
                processed_img = image.resize((224, 224))
                img_array = np.array(processed_img, dtype=np.float32) / 255.0
                img_tensor = np.expand_dims(img_array, axis=0)

                # Inference
                predictions = model.predict(img_tensor, verbose=0)[0]
                pred_idx = int(np.argmax(predictions))
                confidence = float(predictions[pred_idx]) * 100
                pred_label = class_labels.get(pred_idx, 'Unknown')

            # Display Diagnosis Card
            badge_color = "#10B981" if pred_label == 'no_tumor' else "#EF4444"
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="margin-top:0; color: {badge_color};">
                        Predicted Class: {pred_label.replace('_', ' ').upper()}
                    </h3>
                    <h4>Confidence Score: <b>{confidence:.2f}%</b></h4>
                    <p>{label_descriptions.get(pred_label, '')}</p>
                </div>
            """, unsafe_allow_html=True)

            # Class Probability Breakdown
            st.write("### 📊 Probability Distribution")
            for idx, prob in enumerate(predictions):
                class_name = class_labels.get(idx, f"Class {idx}").replace('_', ' ').capitalize()
                st.write(f"**{class_name}**: `{prob * 100:.2f}%`")
                st.progress(float(prob))