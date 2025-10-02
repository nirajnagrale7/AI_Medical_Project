import streamlit as st
import pandas as pd
import pickle
from analyzer import process_uploaded_pdf, extract_text_from_image, analyze_text, detect_gender

# Load the Symptom Checker Model and Encoder
model = pickle.load(open('disease_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
data = pd.read_csv('Training.csv')
all_symptoms = data.columns[:-1].tolist()

# Streamlit App Interface
st.set_page_config(page_title="AI Medical Assistant", layout="wide")

st.title("🩺 AI Medical Report Analyzer & Symptom Checker")
st.markdown("This tool helps you understand your medical reports and check symptoms. **Not a substitute for professional medical advice.**")

# Define tabs
tab1, tab2 = st.tabs(["Symptom Checker", "Medical Report Analyzer"])

# Symptom Checker Tab
with tab1:
    st.header("Symptom Checker")
    selected_symptoms = st.multiselect(
        "What are your symptoms?",
        options=all_symptoms,
        key="symptoms"
    )

    if st.button("Predict Condition"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            input_vector = [0] * len(all_symptoms)
            for symptom in selected_symptoms:
                if symptom in all_symptoms:
                    index = all_symptoms.index(symptom)
                    input_vector[index] = 1
            
            prediction_code = model.predict([input_vector])[0]
            predicted_disease = le.inverse_transform([prediction_code])[0]
            
            st.success(f"The AI suggests the possible condition is: **{predicted_disease}**")
            st.info("It is highly recommended to consult a doctor for an accurate diagnosis.")

# Medical Report Analyzer Tab
with tab2:
    st.header("Medical Report Analyzer")

    uploaded_file = st.file_uploader(
        "Upload your medical report (PDF or Image)",
        type=['pdf', 'png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        st.info("File uploaded successfully. Processing...")
        
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            extracted_text = process_uploaded_pdf(uploaded_file)
        else:
            extracted_text = extract_text_from_image(uploaded_file)

        # Show extracted text for debugging (optional)
        with st.expander("🔍 Show Extracted Text (Debug Mode)", expanded=False):
            st.text_area("Extracted Text Content", extracted_text, height=300)
        
        # Check if extraction failed
        if "Error" in extracted_text or "Could not" in extracted_text:
            st.error(extracted_text)
        else:
            # Detect gender from text
            detected_gender = detect_gender(extracted_text)
            
            # Show detected gender with override option
            col1, col2 = st.columns([2, 3])
            with col1:
                if detected_gender:
                    st.success(f"✅ Gender detected: **{detected_gender.capitalize()}**")
                else:
                    st.warning("⚠️ Gender not detected, defaulting to Male")
            
            with col2:
                # Allow manual override
                use_manual = st.checkbox("Override gender selection")
                if use_manual:
                    gender = st.radio(
                        "Select Gender",
                        ["Male", "Female"],
                        horizontal=True,
                        index=0 if detected_gender != 'female' else 1
                    ).lower()
                else:
                    gender = detected_gender if detected_gender else 'male'
            
            # Analyze with the determined gender
            analysis_results = analyze_text(extracted_text, gender=gender)
            st.subheader("Analysis Results")
            
            # Get the gender used for analysis (could be detected or overridden)
            gender_used = analysis_results.get('detected_gender', gender)
            
            # Remove detected_gender from display results
            display_results = {k: v for k, v in analysis_results.items() if k != 'detected_gender'}

            if not display_results:
                st.warning("Could not find any standard medical parameters to analyze.")
            else:
                # Display gender used for analysis
                st.info(f"Analysis performed using **{gender_used.capitalize()}** reference ranges")
                
                for key, result in display_results.items():
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label=key.replace('_', ' ').title(), value=f"{result['value']} {result['unit']}")
                    col2.metric(label="Status", value=result['status'])
                    col3.metric(label="Normal Range", value=result['normal_range'])
                    
                    if result['status'] == "Abnormal":
                        st.error(f"**Warning:** Your **{key.replace('_', ' ').title()}** level is outside the normal range for {gender_used}.")
st.sidebar.header("About")
st.sidebar.info(
    "This AI assistant combines a symptom checker with an intelligent medical report analyzer. "
    "It automatically detects gender from reports and uses gender-specific reference ranges for accurate analysis. "
    "All results should be confirmed with a healthcare professional."
)