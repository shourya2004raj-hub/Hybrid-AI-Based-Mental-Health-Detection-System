import os

# ====================================
# HIDE TENSORFLOW WARNINGS
# ====================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

# ====================================
# IMPORT LIBRARIES
# ====================================

import streamlit as st
import re
import pickle
import numpy as np

from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]


# ====================================
# PAGE CONFIGURATION
# ====================================

st.set_page_config(
    page_title="Hybrid Mental Health Detection System",
    page_icon="🧠",
    layout="centered"
)

# ====================================
# CUSTOM CSS
# ====================================

st.markdown("""
<style>

.stApp{
    background-color:#F4F7FC;
}

html, body, [class*="css"]{
    font-family:'Segoe UI';
}

.main-container{

    background:white;

    padding:40px;

    border-radius:22px;

    box-shadow:0px 6px 18px rgba(0,0,0,0.08);

    margin-bottom:30px;
}

.result-container{

    background:linear-gradient(
        135deg,
        #EEF4FF,
        #FFFFFF
    );

    padding:35px;

    border-radius:22px;

    box-shadow:0px 6px 18px rgba(0,0,0,0.08);

    margin-top:30px;
}

.final-box{

    background:linear-gradient(
        135deg,
        #DCEBFF,
        #FFFFFF
    );

    padding:30px;

    border-radius:22px;

    border-left:8px solid #4A90E2;

    margin-top:30px;
}

.big-title{

    font-size:48px;

    font-weight:700;

    color:#1E3A5F;

    text-align:center;
}

.sub-title{

    font-size:20px;

    text-align:center;

    color:#4B6584;

    margin-top:10px;
}

.section-title{

    font-size:32px;

    font-weight:700;

    color:#1E3A5F;

    margin-bottom:20px;
}

.final-title{

    font-size:36px;

    font-weight:700;

    color:#1E3A5F;

    margin-bottom:15px;
}

.final-text{

    font-size:24px;

    line-height:1.8;

    color:#1E3A5F;
}

.stButton > button{

    width:100%;

    background:linear-gradient(
        90deg,
        #4A90E2,
        #50C9C3
    );

    color:white;

    border:none;

    border-radius:14px;

    padding:16px;

    font-size:20px;

    font-weight:bold;
}

.stButton > button:hover{

    transform:scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# ====================================
# HEADER
# ====================================

st.markdown(
    '<div class="main-container">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="big-title">🧠 Hybrid Mental Health Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-powered emotional assessment using Questionnaire Analysis and BiLSTM NLP Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ====================================
# LOAD MODEL + TOKENIZER
# ====================================

@st.cache_resource
def load_resources():

    model = load_model(
        r"C:\Final year project\models\bilstm_model.h5"
    )

    with open(
        r"C:\Final year project\models\tokenizer.pkl",
        "rb"
    ) as file:

        tokenizer = pickle.load(file)

    return model, tokenizer

model, tokenizer = load_resources()

# ====================================
# TEXT CLEANING
# ====================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ====================================
# EXPLANATION ENGINE
# ====================================

def generate_explanation(text, confidence):

    text = text.lower()

    indicators = []

    stress_keywords = [
        "anxious",
        "stress",
        "worried",
        "pressure",
        "panic",
        "overthinking",
        "restless",
        "exhausted"
    ]

    depression_keywords = [
        "sad",
        "lonely",
        "hopeless",
        "isolated",
        "empty",
        "worthless",
        "crying",
        "lost interest"
    ]

    normal_keywords = [
        "happy",
        "peaceful",
        "motivated",
        "calm",
        "better",
        "relaxed"
    ]

    for word in stress_keywords:

        if word in text:

            indicators.append(
                f"{word} (Stress Indicator)"
            )

    for word in depression_keywords:

        if word in text:

            indicators.append(
                f"{word} (Depression Indicator)"
            )

    for word in normal_keywords:

        if word in text:

            indicators.append(
                f"{word} (Normal Indicator)"
            )

    if confidence >= 85:

        confidence_text = (
            "The model showed high confidence in this assessment."
        )

    elif confidence >= 60:

        confidence_text = (
            "The model detected moderate emotional patterns."
        )

    else:

        confidence_text = (
            "Some emotional overlap was observed during analysis."
        )

    if indicators:

        return (
            "Detected emotional indicators: "
            + ", ".join(indicators)
            + ". "
            + confidence_text
        )

    else:

        return (
            "The prediction was based on overall sentence-level emotional patterns. "
            + confidence_text
        )

# ====================================
# SUGGESTION ENGINE
# ====================================

def generate_suggestion(prediction):

    if prediction == "Normal":

        return (
            "Maintain healthy routines, proper sleep, physical activity, and positive social interaction."
        )

    elif prediction == "Stress":

        return (
            "Consider stress-management techniques, work-life balance, and relaxation exercises."
        )

    else:

        return (
            "Consider emotional support, healthy routines, and talking to trusted individuals if emotional distress continues."
        )

# ====================================
# FINAL CONCLUSION ENGINE
# ====================================

def generate_final_conclusion(
    prediction,
    confidence,
    questionnaire_level
):

    if prediction == "Normal":

        conclusion = (
            "The emotional analysis indicates emotionally stable patterns."
        )

    elif prediction == "Stress":

        conclusion = (
            "The emotional analysis detected noticeable stress-related emotional patterns."
        )

    else:

        conclusion = (
            "The emotional analysis detected elevated emotional concern along with depression-related emotional patterns."
        )

    if confidence >= 85:

        confidence_text = (
            "The model showed high confidence in this assessment."
        )

    elif confidence >= 60:

        confidence_text = (
            "The model showed moderate confidence in this assessment."
        )

    else:

        confidence_text = (
            "Some emotional overlap was observed during analysis."
        )

    questionnaire_text = (
        f"Questionnaire responses also suggested {questionnaire_level.lower()}."
    )

    return (
        conclusion
        + " "
        + confidence_text
        + " "
        + questionnaire_text
    )

# ====================================
# SIDEBAR
# ====================================

with st.sidebar:

    st.header("📌 Project Information")

    st.success("Model Accuracy: 94%")

    st.subheader("⚙ Technologies Used")

    st.write("""
- Python
- TensorFlow
- BiLSTM
- NLP
- Streamlit
- NLTK
""")

# ====================================
# QUESTIONNAIRE SECTION
# ====================================

st.markdown(
    '<div class="main-container">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">📋 Emotional Questionnaire</div>',
    unsafe_allow_html=True
)

normal_score_map = {
    "Never": 0,
    "Sometimes": 1,
    "Often": 2
}

reverse_score_map = {
    "Never": 2,
    "Sometimes": 1,
    "Often": 0
}

q1 = st.radio(
    "1. Do you often feel emotionally exhausted?",
    ["Never", "Sometimes", "Often"],
    horizontal=True
)

q2 = st.radio(
    "2. Have you lost interest in activities recently?",
    ["Never", "Sometimes", "Often"],
    horizontal=True
)

q3 = st.radio(
    "3. Do you frequently overthink situations?",
    ["Never", "Sometimes", "Often"],
    horizontal=True
)

q4 = st.radio(
    "4. Do you feel lonely or isolated?",
    ["Never", "Sometimes", "Often"],
    horizontal=True
)

q5 = st.radio(
    "5. Are you able to enjoy daily activities?",
    ["Never", "Sometimes", "Often"],
    horizontal=True
)

q6 = st.radio(
    "6. Do you feel motivated during the day?",
    ["Never", "Sometimes", "Often"],
    horizontal=True
)

q7 = st.radio(
    "7. Do you experience sleep-related issues?",
    ["Never", "Sometimes", "Often"],
    horizontal=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ====================================
# TEXT ANALYSIS SECTION
# ====================================

st.markdown(
    '<div class="main-container">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">✍ Emotional Text Analysis</div>',
    unsafe_allow_html=True
)

user_text = st.text_area(
    "Describe your emotional state:",
    height=200
)

predict_button = st.button(
    "🔍 Predict Emotional State"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ====================================
# PREDICTION SECTION
# ====================================

if predict_button:

    with st.spinner(
        "Analyzing emotional patterns..."
    ):

        # ====================================
        # QUESTIONNAIRE SCORE
        # ====================================

        total_score = (

            normal_score_map[q1]
            + normal_score_map[q2]
            + normal_score_map[q3]
            + normal_score_map[q4]

            + reverse_score_map[q5]
            + reverse_score_map[q6]

            + normal_score_map[q7]
        )

        # ====================================
        # QUESTIONNAIRE INTERPRETATION
        # ====================================

        if total_score <= 3:

            questionnaire_level = (
                "emotionally stable patterns"
            )

        elif total_score <= 7:

            questionnaire_level = (
                "mild emotional stress"
            )

        elif total_score <= 11:

            questionnaire_level = (
                "moderate emotional concern"
            )

        else:

            questionnaire_level = (
                "high emotional concern"
            )

        # ====================================
        # NLP PREDICTION
        # ====================================

        cleaned = clean_text(user_text)

        sequence = tokenizer.texts_to_sequences(
            [cleaned]
        )

        if len(sequence[0]) < 2:

            st.error(
                "Invalid or insufficient emotional text."
            )

        else:

            padded = pad_sequences(
                sequence,
                maxlen=100
            )

            prediction = model.predict(
                padded,
                verbose=0
            )

            predicted_class = np.argmax(
                prediction
            )

            confidence = np.max(
                prediction
            ) * 100

            labels = {
                0: "Normal",
                1: "Stress",
                2: "Depression"
            }

            predicted_label = labels[
                predicted_class
            ]

            explanation = generate_explanation(
                cleaned,
                confidence
            )

            suggestion = generate_suggestion(
                predicted_label
            )

            final_conclusion = generate_final_conclusion(
                predicted_label,
                confidence,
                questionnaire_level
            )

            # ====================================
            # RESULT DISPLAY
            # ====================================

            st.markdown(
                '<div class="result-container">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">📊 Analysis Result</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Questionnaire Score",
                f"{total_score}/14"
            )

            col2.metric(
                "NLP Prediction",
                predicted_label
            )

            col3.metric(
                "Confidence",
                f"{round(confidence,2)}%"
            )

            st.subheader(
                "📋 Questionnaire Assessment"
            )

            st.info(
                questionnaire_level.title()
            )

            st.subheader(
                "📖 Explanation"
            )

            st.info(
                explanation
            )

            st.subheader(
                "💡 Suggestion"
            )

            st.success(
                suggestion
            )

            # ====================================
            # FINAL CONCLUSION
            # ====================================

            st.markdown(
                '<div class="final-box">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="final-title">🧠 Final Conclusion</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="final-text">{final_conclusion}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )