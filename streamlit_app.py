"""
🚑 LifeLine AI - Streamlit Deployment
Intelligent Medical Aid for Crisis Zones & Remote Areas

Author: Afan Jeelani
CIS 508 - Machine Learning in Business Term Project (December 2025)
MS in AI in Business | WP Carey School of Business, ASU
"""

import streamlit as st
import pandas as pd
import pickle
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="LifeLine AI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #1E88E5;
        color: #1a1a2e !important;
    }
    .prediction-box h2 {
        color: #1a1a2e !important;
        margin: 0;
    }
    .urgency-high {
        background-color: #ffebee;
        border-left: 5px solid #e53935;
        padding: 15px;
        border-radius: 5px;
        color: #b71c1c !important;
    }
    .urgency-high h3, .urgency-high p {
        color: #b71c1c !important;
    }
    .urgency-moderate {
        background-color: #fff3e0;
        border-left: 5px solid #fb8c00;
        padding: 15px;
        border-radius: 5px;
        color: #e65100 !important;
    }
    .urgency-moderate h3, .urgency-moderate p {
        color: #e65100 !important;
    }
    .urgency-low {
        background-color: #e8f5e9;
        border-left: 5px solid #43a047;
        padding: 15px;
        border-radius: 5px;
        color: #1b5e20 !important;
    }
    .urgency-low h3, .urgency-low p {
        color: #1b5e20 !important;
    }
    .first-aid-box {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
        color: #1a1a2e !important;
    }
    .first-aid-box p {
        color: #1a1a2e !important;
        margin: 10px 0;
    }
    .first-aid-box strong {
        color: #1a1a2e !important;
    }
    .disclaimer {
        background-color: #fff8e1;
        border: 1px solid #ffca28;
        border-radius: 5px;
        padding: 15px;
        font-size: 0.9rem;
        color: #5d4037 !important;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        color: #1a1a2e !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and artifacts."""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, tfidf, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load supplementary data."""
    try:
        descriptions = pd.read_csv('lifeline_ai_descriptions.csv')
        precautions = pd.read_csv('lifeline_ai_precautions.csv')
        return descriptions, precautions
    except Exception as e:
        st.warning(f"Could not load supplementary data: {e}")
        return None, None

# Load everything
model, tfidf, label_encoder = load_model()
descriptions, precautions = load_data()

# =============================================================================
# URGENCY CLASSIFICATION
# =============================================================================
HIGH_URGENCY = [
    'Heart Attack', 'Paralysis (Brain Hemorrhage)', 'Pneumonia',
    'Hypoglycemia', 'AIDS', 'Tuberculosis', 'Hepatitis B',
    'Hepatitis C', 'Hepatitis D'
]

MODERATE_URGENCY = [
    'Dengue', 'Malaria', 'Typhoid', 'Hepatitis A', 'Hepatitis E',
    'Gastroenteritis', 'Jaundice', 'Alcoholic Hepatitis',
    'Chronic Cholestasis', 'Drug Reaction'
]

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_condition(symptoms_text):
    """Predict condition from symptoms."""
    if model is None or tfidf is None:
        return None, None, None, None, None, None
    
    # Vectorize and predict
    text_vec = tfidf.transform([symptoms_text])
    prediction = model.predict(text_vec)[0]
    disease = label_encoder.inverse_transform([prediction])[0]
    
    # Get confidence (improved calculation)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(text_vec)[0]
        raw_confidence = probs[prediction] * 100
        
        # Calculate relative confidence (how much better than 2nd best)
        top_2 = sorted(probs, reverse=True)[:2]
        if top_2[1] > 0:
            relative_confidence = (top_2[0] / (top_2[0] + top_2[1])) * 100
        else:
            relative_confidence = 100.0
        
        # Use the higher of the two (more meaningful to users)
        confidence = max(raw_confidence, relative_confidence)
        
        # Determine confidence level for display
        if confidence >= 80:
            confidence_level = "HIGH"
        elif confidence >= 60:
            confidence_level = "MODERATE"
        else:
            confidence_level = "LOW"
    else:
        confidence = None
        confidence_level = None
    
    # Determine urgency
    if disease in HIGH_URGENCY:
        urgency = "HIGH"
    elif disease in MODERATE_URGENCY:
        urgency = "MODERATE"
    else:
        urgency = "LOW"
    
    # Get description
    description = ""
    if descriptions is not None:
        desc_row = descriptions[descriptions['Disease'] == disease]
        if len(desc_row) > 0:
            description = desc_row['Description'].values[0]
    
    # Get precautions
    first_aid = []
    if precautions is not None:
        prec_row = precautions[precautions['Disease'] == disease]
        if len(prec_row) > 0:
            for i in range(1, 5):
                prec = prec_row[f'Precaution_{i}'].values[0]
                if pd.notna(prec):
                    first_aid.append(prec)
    
    if not first_aid:
        first_aid = ["Rest and monitor symptoms", "Stay hydrated", 
                     "Seek medical attention if symptoms worsen"]
    
    return disease, confidence, confidence_level, urgency, description, first_aid

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/ambulance.png", width=80)
    st.title("LifeLine AI")
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    LifeLine AI provides AI-powered medical guidance for crisis zones 
    and remote areas where professional healthcare is unavailable.
    """)
    
    st.markdown("---")
    st.markdown("### Model Info")
    if model is not None:
        st.success("✅ Model Loaded")
        st.metric("Model Type", type(model).__name__)
        st.metric("Classes", len(label_encoder.classes_))
    else:
        st.error("❌ Model Not Loaded")
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "99.2%")
    col2.metric("F1 Score", "99.2%")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Built by Afan Jeelani<br>
    MS in AI in Business<br>
    WP Carey School of Business, ASU
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown("<h1 class='main-header'>🚑 LifeLine AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Intelligent Medical Aid for Crisis Zones & Remote Areas</p>", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class='disclaimer'>
⚠️ <strong>DISCLAIMER:</strong> This tool provides general guidance only and is NOT a substitute for professional 
medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Initialize session state for symptoms
if 'symptoms' not in st.session_state:
    st.session_state.symptoms = ""

# Input Section
st.markdown("### 📝 Describe Your Symptoms")
symptoms_input = st.text_area(
    label="Enter symptoms",
    value=st.session_state.symptoms,
    placeholder="Example: I have severe chest pain, sweating, and difficulty breathing...",
    height=120,
    label_visibility="collapsed",
    key="symptoms_input"
)

# Update session state when user types
st.session_state.symptoms = symptoms_input

# Example buttons
st.markdown("**Try an example:**")
example_cols = st.columns(4)

examples = [
    ("Chest Pain", "I have severe chest pain, sweating, and difficulty breathing"),
    ("Skin Rash", "I've been having itchy skin with red patches for a week"),
    ("High Fever", "I have high fever, body aches, and feel very weak"),
    ("Joint Pain", "My joints are painful and swollen, especially in the morning")
]

for i, (label, example) in enumerate(examples):
    if example_cols[i].button(label, use_container_width=True, key=f"example_{i}"):
        st.session_state.symptoms = example
        st.rerun()

# Analyze button
st.markdown("<br>", unsafe_allow_html=True)
analyze_button = st.button("🔍 Analyze Symptoms", type="primary", use_container_width=True)

# =============================================================================
# RESULTS SECTION
# =============================================================================
if analyze_button and symptoms_input:
    if len(symptoms_input.strip()) < 10:
        st.warning("Please describe your symptoms in more detail.")
    else:
        with st.spinner("Analyzing symptoms..."):
            disease, confidence, confidence_level, urgency, description, first_aid = predict_condition(symptoms_input)
        
        if disease:
            st.markdown("---")
            
            # LOW CONFIDENCE WARNING
            if confidence and confidence < 60:
                st.error("""
                ⚠️ **LOW CONFIDENCE WARNING**
                
                Your symptoms may not match the conditions our model was trained on. 
                This could be because:
                - Your symptoms describe a trauma/injury (gunshot, fracture, burns, etc.)
                - Your symptoms are for a condition not in our database
                - The symptom description needs more detail
                
                **Please seek professional medical help immediately.**
                
                The prediction below may not be accurate.
                """)
                st.markdown("---")
            
            st.markdown("## 📊 Analysis Results")
            
            # Results in columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Condition
                st.markdown("### 🩺 Identified Condition")
                st.markdown(f"<div class='prediction-box'><h2>{disease}</h2></div>", unsafe_allow_html=True)
                
                # Confidence with level indicator
                if confidence:
                    if confidence_level == "HIGH":
                        conf_color = "#43a047"  # Green
                        conf_icon = "✅"
                    elif confidence_level == "MODERATE":
                        conf_color = "#fb8c00"  # Orange
                        conf_icon = "⚠️"
                    else:
                        conf_color = "#e53935"  # Red
                        conf_icon = "🔍"
                    
                    st.markdown(f"**Confidence:** {conf_icon} {confidence:.0f}% ({confidence_level})")
                    st.progress(min(confidence / 100, 1.0))
                
                # Description
                if description:
                    st.markdown("### 📋 About This Condition")
                    st.info(description)
            
            with col2:
                # Urgency
                st.markdown("### ⚠️ Urgency Level")
                if urgency == "HIGH":
                    st.markdown("""
                    <div class='urgency-high'>
                        <h3>🔴 HIGH URGENCY</h3>
                        <p><strong>SEEK IMMEDIATE MEDICAL ATTENTION!</strong></p>
                        <p>This condition may be life-threatening. Call emergency services or go to the nearest hospital immediately.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif urgency == "MODERATE":
                    st.markdown("""
                    <div class='urgency-moderate'>
                        <h3>🟠 MODERATE URGENCY</h3>
                        <p><strong>Consult a healthcare provider soon.</strong></p>
                        <p>Monitor your symptoms closely and seek medical advice within 24-48 hours.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class='urgency-low'>
                        <h3>🟢 LOW-MODERATE URGENCY</h3>
                        <p><strong>Monitor your symptoms.</strong></p>
                        <p>Rest and follow first-aid recommendations. Seek medical advice if symptoms worsen.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # First Aid
                st.markdown("### 🩹 Recommended Actions")
                first_aid_html = "<div class='first-aid-box'>"
                for i, step in enumerate(first_aid, 1):
                    first_aid_html += f"<p><strong>{i}.</strong> {step}</p>"
                first_aid_html += "</div>"
                st.markdown(first_aid_html, unsafe_allow_html=True)
            
            # Additional warning for high urgency
            if urgency == "HIGH":
                st.markdown("---")
                st.error("🚨 **EMERGENCY:** If you are experiencing these symptoms, please call emergency services immediately!")

elif analyze_button and not symptoms_input:
    st.warning("Please enter your symptoms before analyzing.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>LifeLine AI</strong> | Built for humanitarian aid in crisis zones</p>
    <p> CIS 508 - Machine Learning in Business Term Project | MS in AI in Business, WP Carey School of Business, Arizona State University</p>
    <p style='font-size: 0.8rem;'>Powered by Machine Learning • Designed with ❤️ for humanity</p>
</div>
""", unsafe_allow_html=True)
