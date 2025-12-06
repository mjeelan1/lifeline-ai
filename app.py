"""
LifeLine AI - Medical Triage System for Crisis Zones
Streamlit Application - Direct Model Loading

Author: Afan Jeelani
Course: CIS 508 - Machine Learning in Business, MS-AIB
Model: XGBoost (92.06% accuracy)
"""

import streamlit as st
import pickle
import json
import numpy as np

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="LifeLine AI - Medical Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================
@st.cache_resource
def load_model():
    """Load XGBoost model and TF-IDF vectorizer."""
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('xgboost_tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, tfidf, label_encoder, True
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None, False

@st.cache_data
def load_condition_data():
    """Load condition info and supply chain from JSON file."""
    try:
        with open('lifeline_streamlit_data.json', 'r') as f:
            data = json.load(f)
        return data.get('condition_info', {}), data.get('supply_chain', {})
    except FileNotFoundError:
        return {}, {}

# Load everything
MODEL, TFIDF, LABEL_ENCODER, MODEL_LOADED = load_model()
CONDITION_INFO, SUPPLY_CHAIN = load_condition_data()

# =============================================================================
# DEFAULT DATA (Fallback)
# =============================================================================
DEFAULT_CONDITION_INFO = {
    "description": "Medical condition requiring professional evaluation.",
    "precautions": ["Consult healthcare provider", "Monitor symptoms", "Rest", "Stay hydrated"]
}

DEFAULT_SUPPLY_CHAIN = {
    "priority": "MEDIUM",
    "triage_color": "YELLOW",
    "immediate_supplies": ["First aid kit", "Pain medication", "Bandages", "Antiseptic"],
    "equipment": ["Vital signs monitor", "Basic diagnostic tools"],
    "notes": "Assess patient condition. Provide supportive care. Monitor vital signs."
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_triage_color_css(color_name):
    """Get CSS color for triage."""
    colors = {"RED": "#FF4444", "YELLOW": "#FFD700", "GREEN": "#00AA00"}
    return colors.get(color_name, "#808080")

def get_priority_emoji(priority):
    """Get emoji for priority level."""
    emojis = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    return emojis.get(priority, "⚪")

def predict(text):
    """Predict condition using XGBoost model."""
    if not MODEL_LOADED:
        return None, None
    
    # Transform text
    text_tfidf = TFIDF.transform([text])
    
    # Get prediction probabilities
    probs = MODEL.predict_proba(text_tfidf)[0]
    
    # Primary prediction
    pred_idx = np.argmax(probs)
    condition = LABEL_ENCODER.classes_[pred_idx]
    top1_prob = probs[pred_idx]
    
    # Get second highest for confidence calculation
    sorted_probs = np.sort(probs)[::-1]
    top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0
    
    # Calculate confidence tier
    confidence_ratio = top1_prob / top2_prob if top2_prob > 0 else 10
    
    if confidence_ratio >= 2.0 and top1_prob >= 0.15:
        confidence_tier = "HIGH"
    elif confidence_ratio >= 1.3 and top1_prob >= 0.10:
        confidence_tier = "MEDIUM"
    else:
        confidence_tier = "LOW"
    
    return condition, confidence_tier

def get_condition_info(condition):
    """Get description and precautions for a condition."""
    return CONDITION_INFO.get(condition, DEFAULT_CONDITION_INFO)

def get_supply_chain(condition):
    """Get supply chain information for a condition."""
    return SUPPLY_CHAIN.get(condition, DEFAULT_SUPPLY_CHAIN)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Centered Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 3em; margin-bottom: 0;">🏥 LifeLine AI</h1>
        <p style="font-size: 1.3em; color: #666;">Medical Triage System for Crisis Zones</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown("""
        **LifeLine AI** is an ML-powered medical triage system 
        designed for crisis zones and resource-limited settings.
        
        **Capabilities:**
        - 🔍 Symptom-based diagnosis
        - 🎯 65 medical conditions
        - 📋 Disease descriptions
        - ⚠️ Precautions & first aid
        - 📦 Supply chain recommendations
        - ⚡ Triage prioritization
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("**Model:** XGBoost")
        st.info("**Accuracy:** 92.06%")
        st.info("**Conditions:** 65")
        
        st.markdown("---")
        st.markdown("### 🔌 Status")
        if MODEL_LOADED:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Found")
        
        st.markdown("---")
        st.caption("**Developer:** Afan Jeelani")
        st.caption("**Course:** CIS 508 - Machine Learning in Business, MS-AIB")
    
    # Check if model is loaded
    if not MODEL_LOADED:
        st.error("Model files not found. Please ensure these files are in the app directory:")
        st.code("xgboost_model.pkl\nxgboost_tfidf.pkl\nlabel_encoder.pkl")
        return
    
    # Main input
    st.markdown("### 📝 Enter Patient Symptoms")
    
    symptoms_input = st.text_area(
        "Describe the patient's symptoms or condition:",
        height=120,
        placeholder="Example: Patient presents with severe leg pain, swelling, inability to walk after a fall. Visible deformity in lower leg."
    )
    
    st.markdown("---")
    
    # Analyze button
    if st.button("🔍 **Analyze Symptoms**", type="primary", use_container_width=True):
        if symptoms_input:
            with st.spinner("🔄 Analyzing symptoms..."):
                condition, confidence_tier = predict(symptoms_input)
            
            if condition:
                # Get all info
                condition_info = get_condition_info(condition)
                supply_chain = get_supply_chain(condition)
                
                priority = supply_chain.get('priority', 'MEDIUM')
                triage_color = supply_chain.get('triage_color', 'YELLOW')
                
                st.markdown("---")
                st.markdown("## 🎯 Diagnosis Results")
                
                # Main result card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                            padding: 25px; border-radius: 15px; 
                            border-left: 6px solid {get_triage_color_css(triage_color)};
                            margin-bottom: 20px;">
                    <h2 style="color: white; margin: 0 0 10px 0;">🏥 {condition}</h2>
                    <p style="color: #ddd; font-size: 1.1em; margin: 0;">
                        {get_priority_emoji(priority)} <strong>Priority:</strong> {priority} &nbsp;|&nbsp;
                        <strong>Triage:</strong> <span style="color: {get_triage_color_css(triage_color)}; font-weight: bold;">{triage_color}</span> &nbsp;|&nbsp;
                        <strong>Confidence:</strong> {confidence_tier}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tabs for organized information
                tab1, tab2, tab3 = st.tabs(["📋 Description & Precautions", "📦 Supply Chain", "🏥 Clinical Notes"])
                
                with tab1:
                    # Description
                    st.markdown("### 📖 About This Condition")
                    description = condition_info.get('description', 'No description available.')
                    st.info(description)
                    
                    # Precautions
                    st.markdown("### ⚠️ Recommended Precautions")
                    precautions = condition_info.get('precautions', [])
                    if precautions:
                        for i, precaution in enumerate(precautions, 1):
                            st.markdown(f"**{i}.** {precaution.title()}")
                    else:
                        st.write("No specific precautions listed.")
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🚨 Immediate Supplies")
                        immediate = supply_chain.get('immediate_supplies', [])
                        for item in immediate:
                            st.markdown(f"- {item}")
                    
                    with col2:
                        st.markdown("### 🔧 Equipment Needed")
                        equipment = supply_chain.get('equipment', [])
                        for item in equipment:
                            st.markdown(f"- {item}")
                
                with tab3:
                    # Clinical Notes
                    st.markdown("### 📝 First Aid Notes")
                    notes = supply_chain.get('notes', 'Provide supportive care.')
                    st.warning(notes)
                    
                    # Priority explanation
                    st.markdown("### 🚦 Priority Level")
                    if priority == "CRITICAL":
                        st.error("🔴 **CRITICAL:** Immediate life-threatening. Treat FIRST!")
                    elif priority == "HIGH":
                        st.warning("🟠 **HIGH:** Urgent. Treat within 10-15 minutes.")
                    elif priority == "MEDIUM":
                        st.info("🟡 **MEDIUM:** Semi-urgent. Treat within 30-60 minutes.")
                    else:
                        st.success("🟢 **LOW:** Non-urgent. Can wait if needed.")
                
                # Disclaimer
                st.markdown("---")
                st.caption("""
                ⚠️ **Disclaimer:** LifeLine AI is a decision-support tool designed for crisis zones. 
                It should not replace professional medical diagnosis. Always consult qualified healthcare providers when available.
                """)
        else:
            st.warning("⚠️ Please enter symptoms to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        <p>🏥 LifeLine AI | Built for Humanitarian Aid in Crisis Zones</p>
        <p>CIS 508 - Machine Learning in Business | MS-AIB, W.P. Carey School of Business, ASU</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()
