"""
LifeLine AI - Medical Triage System for Crisis Zones
Streamlit Application with Databricks Model Serving

Author: Afan Jeelani
Course: MS in AI in Business, ASU
Model: XGBoost (92.06% accuracy)
"""

import streamlit as st
import requests
import json
import os

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
# LOAD DATA FROM JSON FILE
# =============================================================================
@st.cache_data
def load_condition_data():
    """Load condition info and supply chain from JSON file."""
    try:
        with open('lifeline_streamlit_data.json', 'r') as f:
            data = json.load(f)
        return data.get('condition_info', {}), data.get('supply_chain', {})
    except FileNotFoundError:
        st.error("❌ Data file not found. Please ensure lifeline_streamlit_data.json is in the app directory.")
        return {}, {}

CONDITION_INFO, SUPPLY_CHAIN = load_condition_data()

# =============================================================================
# DATABRICKS CONFIGURATION
# =============================================================================
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
MODEL_ENDPOINT = os.getenv("MODEL_ENDPOINT", "")

# Load from Streamlit secrets if available
try:
    DATABRICKS_HOST = st.secrets.get("DATABRICKS_HOST", DATABRICKS_HOST)
    DATABRICKS_TOKEN = st.secrets.get("DATABRICKS_TOKEN", DATABRICKS_TOKEN)
    MODEL_ENDPOINT = st.secrets.get("MODEL_ENDPOINT", MODEL_ENDPOINT)
except:
    pass

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

def predict_with_databricks(text):
    """Call Databricks Model Serving endpoint for prediction."""
    if not all([DATABRICKS_HOST, DATABRICKS_TOKEN, MODEL_ENDPOINT]):
        st.error("❌ Databricks credentials not configured. Please set secrets.")
        return None
    
    url = f"{DATABRICKS_HOST}/serving-endpoints/{MODEL_ENDPOINT}/invocations"
    
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {"inputs": [text]}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        prediction = result.get("predictions", [None])[0]
        return prediction
    except Exception as e:
        st.error(f"Error calling Databricks: {str(e)}")
        return None

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
    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# 🏥")
    with col2:
        st.title("LifeLine AI")
        st.markdown("**Medical Triage System for Crisis Zones**")
    
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
        st.markdown("### 🔌 Connection Status")
        if all([DATABRICKS_HOST, DATABRICKS_TOKEN, MODEL_ENDPOINT]):
            st.success("✅ Databricks Connected")
        else:
            st.error("❌ Configure Databricks Secrets")
        
        st.markdown("---")
        st.markdown("### 📚 Data Sources")
        st.caption("Disease descriptions from Kaggle Medical Dataset")
        st.caption("Supply chain mapped for crisis zones")
        
        st.markdown("---")
        st.caption("**Developer:** Afan Jeelani")
        st.caption("**Course:** MS in AI in Business, ASU")
    
    # Main input
    st.markdown("### 📝 Enter Patient Symptoms")
    
    symptoms_input = st.text_area(
        "Describe the patient's symptoms or condition:",
        height=120,
        placeholder="Example: Patient presents with severe leg pain, swelling, inability to walk after a fall. Visible deformity in lower leg."
    )
    
    # Example buttons
    st.markdown("**Quick Examples:**")
    col1, col2, col3, col4 = st.columns(4)
    
    example_selected = None
    with col1:
        if st.button("🦴 Fracture"):
            example_selected = "Patient fell from height, severe pain in arm, visible swelling, cannot move the limb"
    with col2:
        if st.button("🔥 Burn"):
            example_selected = "Patient has burn injury with blistering, severe pain, red and moist skin on hand"
    with col3:
        if st.button("🐕 Animal Bite"):
            example_selected = "Patient presents with bite wound, puncture marks, bleeding, swelling from dog attack"
    with col4:
        if st.button("🤒 Malaria"):
            example_selected = "Patient has high fever, chills, sweating, headache, body aches, nausea"
    
    # Use example if selected
    if example_selected:
        symptoms_input = example_selected
    
    st.markdown("---")
    
    # Analyze button
    if st.button("🔍 **Analyze Symptoms**", type="primary", use_container_width=True):
        if symptoms_input:
            with st.spinner("🔄 Analyzing symptoms..."):
                condition = predict_with_databricks(symptoms_input)
            
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
                        <strong>Triage:</strong> <span style="color: {get_triage_color_css(triage_color)}; font-weight: bold;">{triage_color}</span>
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
                    st.markdown("### 🚦 Priority Level Meaning")
                    if priority == "CRITICAL":
                        st.error("🔴 **CRITICAL:** Immediate life-threatening condition. Treat FIRST before all others!")
                    elif priority == "HIGH":
                        st.warning("🟠 **HIGH:** Urgent condition. Treat within 10-15 minutes.")
                    elif priority == "MEDIUM":
                        st.info("🟡 **MEDIUM:** Semi-urgent. Treat within 30-60 minutes.")
                    else:
                        st.success("🟢 **LOW:** Non-urgent. Can wait if resources are limited.")
                
                # Disclaimer
                st.markdown("---")
                st.caption("""
                ⚠️ **Disclaimer:** LifeLine AI is a decision-support tool designed for crisis zones and 
                resource-limited settings. It should not replace professional medical diagnosis. 
                Always consult qualified healthcare providers when available.
                """)
        else:
            st.warning("⚠️ Please enter symptoms to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9em;">
        <p>🏥 LifeLine AI | Built for Humanitarian Aid in Crisis Zones</p>
        <p>MS in AI in Business | WP Carey School of Business, ASU</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()
