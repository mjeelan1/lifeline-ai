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
import re

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
# SMART SYMPTOM EXPANSION FUNCTION
# =============================================================================

# Injury-related keywords (OSHA data uses third person)
INJURY_KEYWORDS = [
    'fall', 'fell', 'falling', 'dropped', 'slip', 'slipped', 'trip', 'tripped',
    'cut', 'cutting', 'struck', 'hit', 'hitting', 'crush', 'crushed', 'caught',
    'burn', 'burned', 'burning', 'fire', 'flame', 'hot', 'chemical', 'acid',
    'electric', 'shock', 'electrocuted', 'wire', 'current',
    'machine', 'equipment', 'tool', 'saw', 'drill', 'press', 'conveyor',
    'ladder', 'scaffold', 'roof', 'height', 'stairs',
    'vehicle', 'truck', 'forklift', 'car', 'collision', 'crash',
    'explosion', 'exploded', 'blast',
    'amputation', 'amputated', 'severed', 'laceration', 'lacerated',
    'fracture', 'fractured', 'broken', 'break', 'broke',
    'sprain', 'strain', 'twisted', 'dislocated', 'dislocation',
    'puncture', 'punctured', 'stabbed', 'pierced',
    'bite', 'bitten', 'bit', 'sting', 'stung', 'attack', 'attacked',
    'gunshot', 'shot', 'bullet', 'wound', 'wounded',
    'smoke', 'inhaled', 'inhalation', 'fumes', 'gas',
    'drown', 'drowning', 'suffocate', 'choking', 'choked',
    'accident', 'incident', 'injury', 'injured', 'trauma',
    'working', 'worker', 'employee', 'operator', 'technician'
]

# Medical symptom keywords (use first person)
MEDICAL_KEYWORDS = [
    'fever', 'temperature', 'chills', 'sweating', 'sweat',
    'headache', 'migraine', 'dizzy', 'dizziness', 'vertigo',
    'nausea', 'vomiting', 'vomit', 'diarrhea', 'constipation',
    'cough', 'coughing', 'sneeze', 'cold', 'flu', 'throat',
    'rash', 'itching', 'itchy', 'skin', 'acne', 'patches', 'spots',
    'fatigue', 'tired', 'weakness', 'weak', 'exhausted',
    'stomach', 'abdominal', 'belly', 'indigestion', 'bloating',
    'urination', 'urinating', 'urine', 'bladder', 'kidney',
    'breathing', 'breathless', 'breath', 'asthma', 'wheezing',
    'heart', 'chest', 'palpitation', 'blood pressure',
    'diabetes', 'sugar', 'glucose', 'insulin',
    'joint', 'arthritis', 'stiff', 'stiffness', 'swollen',
    'infection', 'infected', 'inflammation', 'inflamed',
    'allergy', 'allergic', 'reaction', 'hives',
    'anxiety', 'stress', 'depression', 'mental',
    'vision', 'eye', 'blurred', 'blind', 'seeing',
    'hearing', 'ear', 'deaf', 'ringing',
    'sleeping', 'insomnia', 'sleep', 'restless'
]

def detect_input_style(text):
    """
    Detect if input is first person, third person, or clinical style.
    Returns: 'first', 'third', 'clinical', or 'unknown'
    """
    text_lower = text.lower().strip()
    
    # First person indicators
    first_person = ['i have', 'i am', 'i feel', "i've", 'i got', 'my ', 'me ', 
                    'i experienced', 'i noticed', "i'm", 'i was', 'myself']
    
    # Third person indicators
    third_person = ['a person', 'the person', 'a worker', 'the worker', 'a patient',
                    'an employee', 'the employee', 'he was', 'she was', 'they were',
                    'a man', 'a woman', 'the man', 'the woman', 'someone', 'victim']
    
    # Clinical indicators
    clinical = ['patient presents', 'patient has', 'patient is', 'patient was',
                'presents with', 'complains of', 'symptoms include', 'diagnosed',
                'examination', 'vital signs', 'bp:', 'temp:', 'hr:']
    
    for phrase in first_person:
        if phrase in text_lower:
            return 'first'
    
    for phrase in third_person:
        if phrase in text_lower:
            return 'third'
    
    for phrase in clinical:
        if phrase in text_lower:
            return 'clinical'
    
    return 'unknown'

def detect_condition_type(text):
    """
    Detect if the input is likely an injury or medical condition.
    Returns: 'injury' or 'medical'
    """
    text_lower = text.lower()
    
    injury_score = sum(1 for keyword in INJURY_KEYWORDS if keyword in text_lower)
    medical_score = sum(1 for keyword in MEDICAL_KEYWORDS if keyword in text_lower)
    
    # If mentions work/accident/machine context, likely injury
    work_context = any(word in text_lower for word in ['work', 'job', 'site', 'factory', 'construction', 'machine'])
    if work_context:
        injury_score += 3
    
    return 'injury' if injury_score > medical_score else 'medical'

def expand_symptoms(user_input):
    """
    ALWAYS convert input to training style for best accuracy.
    - Injuries → 3rd person OSHA style
    - Medical → 1st person narrative style
    """
    text = user_input.strip()
    text_lower = text.lower()
    
    # Detect if injury or medical
    condition_type = detect_condition_type(text)
    
    # Common symptom expansions
    symptom_expansions = {
        'headache': 'severe headache',
        'fever': 'high fever',
        'cough': 'persistent cough',
        'pain': 'severe pain',
        'vomiting': 'nausea and vomiting',
        'nausea': 'feeling nauseous',
        'fatigue': 'extreme fatigue and weakness',
        'tired': 'feeling very tired and weak',
        'dizzy': 'feeling dizzy and lightheaded',
        'dizziness': 'experiencing dizziness',
        'rash': 'skin rash',
        'itching': 'severe itching',
        'swelling': 'visible swelling',
        'bleeding': 'bleeding from the affected area',
        'breathless': 'shortness of breath',
        'chest pain': 'chest pain and tightness',
        'stomach pain': 'abdominal pain and discomfort',
        'back pain': 'severe back pain',
        'burning': 'burning sensation',
    }
    
    # Expand known symptoms in the text
    expanded_text = text_lower
    for short, long in symptom_expansions.items():
        if short in expanded_text and long not in expanded_text:
            expanded_text = expanded_text.replace(short, long)
    
    # Extract symptom phrases
    symptoms = re.split(r'[,;.\n]+', expanded_text)
    symptoms = [s.strip() for s in symptoms if s.strip() and len(s.strip()) > 2]
    
    if len(symptoms) == 0:
        symptoms = [expanded_text]
    
    # Build narrative based on condition type and detected/target style
    if condition_type == 'injury':
        # Use OSHA-style third person narrative for injuries
        narrative = build_injury_narrative(symptoms, text_lower)
    else:
        # Use first person narrative for medical conditions
        narrative = build_medical_narrative(symptoms, text_lower)
    
    return narrative

def build_injury_narrative(symptoms, original_text):
    """Build OSHA-style third person narrative for injuries."""
    
    # Detect specific injury context
    context = ""
    
    if any(word in original_text for word in ['fall', 'fell', 'ladder', 'height', 'stairs', 'roof']):
        context = "a person fell"
    elif any(word in original_text for word in ['cut', 'laceration', 'saw', 'blade', 'sharp']):
        context = "a person was cut"
    elif any(word in original_text for word in ['burn', 'fire', 'hot', 'chemical', 'acid']):
        context = "a person was burned"
    elif any(word in original_text for word in ['crush', 'caught', 'machine', 'press', 'conveyor']):
        context = "a person was caught in machinery"
    elif any(word in original_text for word in ['struck', 'hit', 'object', 'falling object']):
        context = "a person was struck by an object"
    elif any(word in original_text for word in ['electric', 'shock', 'wire', 'current']):
        context = "a person received an electric shock"
    elif any(word in original_text for word in ['bite', 'bitten', 'dog', 'animal', 'snake', 'spider']):
        context = "a person was bitten"
    elif any(word in original_text for word in ['gun', 'shot', 'bullet']):
        context = "a person sustained a gunshot wound"
    elif any(word in original_text for word in ['vehicle', 'car', 'truck', 'crash', 'collision']):
        context = "a person was involved in a vehicle accident"
    else:
        context = "a person was injured"
    
    # Build the narrative
    symptom_text = ", ".join(symptoms[:4])  # Limit to first 4 symptoms
    
    narrative = f"{context}. The person is experiencing {symptom_text}."
    
    # Add injury-specific details
    if any(word in original_text for word in ['fracture', 'broken', 'break', 'bone']):
        narrative += " There appears to be a possible fracture."
    if any(word in original_text for word in ['bleeding', 'blood']):
        narrative += " There is visible bleeding."
    if any(word in original_text for word in ['unconscious', 'unresponsive', 'passed out']):
        narrative += " The person lost consciousness."
    if any(word in original_text for word in ['swelling', 'swollen']):
        narrative += " There is visible swelling in the affected area."
    if any(word in original_text for word in ['cannot move', 'unable to move', 'immobile']):
        narrative += " The person is unable to move the affected limb."
    
    return narrative

def build_medical_narrative(symptoms, original_text):
    """Build first person narrative for medical conditions."""
    
    if len(symptoms) == 1:
        narrative = f"I have been experiencing {symptoms[0]}. This has been bothering me significantly."
    elif len(symptoms) == 2:
        narrative = f"I have been experiencing {symptoms[0]}. I also have {symptoms[1]}. These symptoms are affecting my daily life."
    else:
        first_symptoms = ", ".join(symptoms[:-1])
        last_symptom = symptoms[-1]
        narrative = f"I have been experiencing {first_symptoms}, and {last_symptom}. These symptoms have been persistent."
    
    # Add context based on detected symptoms
    if any(word in original_text for word in ['fever', 'temperature', 'chills']):
        narrative += " My body temperature feels elevated and I have chills."
    if any(word in original_text for word in ['skin', 'rash', 'itch', 'patches']):
        narrative += " My skin has been affected with visible changes."
    if any(word in original_text for word in ['stomach', 'abdominal', 'digest', 'eat']):
        narrative += " I am having digestive issues."
    if any(word in original_text for word in ['breath', 'breathing', 'chest', 'lung']):
        narrative += " I am having difficulty breathing."
    if any(word in original_text for word in ['urin', 'bladder', 'kidney']):
        narrative += " I am experiencing urinary symptoms."
    if any(word in original_text for word in ['joint', 'muscle', 'ache', 'stiff']):
        narrative += " My joints and muscles are affected."
    if any(word in original_text for word in ['head', 'migraine', 'vision']):
        narrative += " I am also having issues with my head and vision."
    
    return narrative

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
    
    # Expand short inputs to match training style
    expanded_text = expand_symptoms(text)
    
    # Transform text
    text_tfidf = TFIDF.transform([expanded_text])
    
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
        placeholder="Examples:\n• fever, chills, headache, body aches\n• patient fell from ladder with leg pain and swelling\n• I have burning sensation during urination"
    )
    
    # Helpful tips
    with st.expander("💡 Tips for best results"):
        st.markdown("""
        **The system understands multiple input styles:**
        
        🩺 **Medical symptoms (any format):**
        - "fever, chills, headache"
        - "I have been experiencing stomach pain and nausea"
        - "Patient presents with chest pain and breathlessness"
        
        🩹 **Injuries (any format):**
        - "fell from height, leg pain, cannot walk"
        - "A worker was cut by a machine"
        - "Patient has burn injuries on arm"
        
        **Short or long - both work!**
        """)
    
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
        <p>CIS 508 - Machine Learning in Business | MS-AIB, ASU</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    main()
