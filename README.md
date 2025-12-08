# 🏥 LifeLine AI

## Medical Triage System for Crisis Zones

LifeLine AI is a machine learning-powered medical triage system designed to assist healthcare workers and first responders in crisis zones and resource-limited settings. The system analyzes patient symptoms and provides diagnostic predictions, triage prioritization, and supply chain recommendations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-92.06%25-brightgreen.svg)

---

## 🎯 Features

- **Symptom-Based Diagnosis** - Enter patient symptoms in natural language for AI-powered condition prediction
- **65 Medical Conditions** - Covers both medical conditions and injury types
- **Triage Prioritization** - Color-coded priority levels (RED/YELLOW/GREEN) for emergency response
- **Disease Descriptions** - Detailed information about each predicted condition
- **Precautions and First Aid** - Recommended actions for immediate care
- **Supply Chain Mapping** - Lists required medical supplies and equipment for each condition

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Model** | XGBoost |
| **Accuracy** | 92.06% |
| **F1 Score** | 0.9215 |
| **Conditions** | 65 |
| **Dataset Size** | 7,803 records |

---

## 📁 Project Structure

```
lifeline-ai/
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── lifeline_streamlit_data.json  # Disease info, precautions, supply chain
├── xgboost_model.pkl             # Trained XGBoost model
├── xgboost_tfidf.pkl             # TF-IDF vectorizer
├── label_encoder.pkl             # Label encoder for conditions
└── README.md                     # This file
```

---

## 🛠️ Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lifeline-ai.git
   cd lifeline-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   ```
   http://localhost:8501
   ```

### Deploy on Streamlit Cloud

1. Push all files to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Click Deploy

---

## 💡 Usage

1. Enter patient symptoms in the text area
   - Example: *"Patient presents with severe leg pain, swelling, inability to walk after a fall"*

2. Click **"Analyze Symptoms"**

3. View results:
   - **Diagnosis** - Predicted condition with confidence level
   - **Priority** - Triage color and urgency level
   - **Description** - Information about the condition
   - **Precautions** - Recommended immediate actions
   - **Supply Chain** - Required medical supplies and equipment

---

## 🏥 Supported Conditions

### Medical Conditions (40)
Malaria, Typhoid, Dengue, Pneumonia, Tuberculosis, Diabetes, Hypertension, Heart Attack, Migraine, Asthma, and more...

### Injury Conditions (25)
Fracture, Burns (Thermal, Chemical, Electrical), Concussion, Traumatic Brain Injury, Gunshot Wound, Animal Bite, Laceration, and more...

---

## 🔬 Technical Details

### Machine Learning Pipeline
- **Text Vectorization:** TF-IDF (8000 features, bigrams)
- **Model:** XGBoost Classifier with class weights for imbalanced data
- **Training:** 9 models evaluated, XGBoost selected as best performer

### Data Sources
- Symptom2Disease - Kaggle
- OSHA Injury Dataset - OSHA, U.S. Department of Labor
- Disease Symptom Prediction - Kaggle

---

## ⚠️ Disclaimer

LifeLine AI is a **decision-support tool** designed for crisis zones and resource-limited settings. It should **not replace professional medical diagnosis**. Always consult qualified healthcare providers when available.

---

## 👨‍💻 Author

**Afan Jeelani**

- Course: CIS 508 - Machine Learning in Business
- Program: MS in AI in Business (MS-AIB)
- University: W.P. Carey School of Business, Arizona State University

---

## 📄 License

This project is for educational purposes as part of the CIS 508 coursework at Arizona State University.

---

## 🙏 Acknowledgments

- W.P. Carey School of Business, ASU
- Kaggle for medical datasets
- Doctors Without Borders for inspiration on crisis zone healthcare challenges
