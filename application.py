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

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    background-color: var(--secondary-background-color);
    padding:40px;
    border-radius:22px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
    margin-bottom:30px;
}

.result-container{
    background-color: var(--secondary-background-color);
    padding:35px;
    border-radius:22px;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
    margin-top:30px;
}

.final-box{
    background-color: var(--secondary-background-color);
    padding:30px;
    border-radius:22px;
    border-left:8px solid #4A90E2;
    margin-top:30px;
}

.big-title{
    font-size:48px;
    font-weight:700;
    text-align:center;
}

.sub-title{
    font-size:20px;
    text-align:center;
    margin-top:10px;
}

.section-title{
    font-size:32px;
    font-weight:700;
    margin-bottom:20px;
}

.final-title{
    font-size:36px;
    font-weight:700;
    margin-bottom:15px;
}

.final-text{
    font-size:24px;
    line-height:1.8;
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
        "models/bilstm_model.h5"
    )

    with open(
        "models/tokenizer.pkl",
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

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text

# ====================================
# ADVANCED CONTEXT DETECTION
# ====================================

def detect_context(text):

    text = text.lower()

    context_map = {

        "career": [
            "job",
            "offer",
            "career",
            "company",
            "layoff",
            "placement",
            "internship",
            "future",
            "promotion"
        ],

        "academic": [
            "exam",
            "study",
            "marks",
            "college",
            "assignment",
            "semester",
            "result",
            "cgpa"
        ],

        "relationship": [
            "breakup",
            "relationship",
            "partner",
            "love",
            "ignored",
            "alone",
            "friend"
        ],

        "financial": [
            "money",
            "financial",
            "loan",
            "debt",
            "salary",
            "expenses"
        ],

        "burnout": [
            "tired",
            "burnout",
            "exhausted",
            "drained",
            "overworked"
        ],

        "health": [
            "health",
            "disease",
            "pain",
            "illness",
            "hospital"
        ],

        "family": [
            "family",
            "parents",
            "mother",
            "father",
            "home"
        ]
    }

    positive_words = [
        "confident",
        "hopeful",
        "motivated",
        "optimistic",
        "determined",
        "calm",
        "fine",
        "okay",
        "stable",
        "positive",
        "happy",
        "excited"
    ]

    negative_words = [
        "worried",
        "afraid",
        "scared",
        "hopeless",
        "lonely",
        "sad",
        "depressed",
        "anxious",
        "stressed"
    ]

    detected_contexts = []

    for context, keywords in context_map.items():

        if any(word in text for word in keywords):

            detected_contexts.append(context)

    positive_detected = any(
        word in text for word in positive_words
    )

    negative_detected = any(
        word in text for word in negative_words
    )

    return (
        detected_contexts,
        positive_detected,
        negative_detected
    )

# ====================================
# SMART EXPLANATION ENGINE
# ====================================

def generate_explanation(
    text,
    prediction,
    confidence
):

    (
        contexts,
        positive_detected,
        negative_detected
    ) = detect_context(text)

    explanation_parts = []

    # ====================================
    # PRIMARY EMOTIONAL ANALYSIS
    # ====================================

    if prediction == "Stress":

        explanation_parts.append(
            "The emotional analysis detected stress-related emotional patterns."
        )

    elif prediction == "Depression":

        explanation_parts.append(
            "The emotional analysis detected emotionally concerning depressive indicators."
        )

    else:

        explanation_parts.append(
            "The emotional analysis detected emotionally stable patterns."
        )

    # ====================================
    # CONTRADICTION HANDLING
    # ====================================

    if positive_detected and negative_detected:

        explanation_parts.append(
            "The response contains both emotional concern and positive coping indicators, suggesting emotional resilience during challenging situations."
        )

    elif positive_detected:

        explanation_parts.append(
            "Positive emotional indicators such as confidence, optimism, or emotional stability were observed."
        )

    elif negative_detected:

        explanation_parts.append(
            "The response reflected emotional concern and psychological pressure indicators."
        )

    # ====================================
    # CONTEXTUAL INTERPRETATION
    # ====================================

    if "career" in contexts:

        explanation_parts.append(
            "Career-related thoughts and future uncertainty patterns were identified."
        )

    if "academic" in contexts:

        explanation_parts.append(
            "Academic pressure and performance-related emotional indicators were observed."
        )

    if "relationship" in contexts:

        explanation_parts.append(
            "Relationship-related emotional sensitivity patterns were identified."
        )

    if "financial" in contexts:

        explanation_parts.append(
            "Financial uncertainty and future-security concerns were reflected in the response."
        )

    if "burnout" in contexts:

        explanation_parts.append(
            "The response indicates emotional exhaustion and mental fatigue patterns."
        )

    if "health" in contexts:

        explanation_parts.append(
            "Health-related emotional concern and psychological stress indicators were observed."
        )

    if "family" in contexts:

        explanation_parts.append(
            "Family-related emotional attachment and concern patterns were identified."
        )

    # ====================================
    # CONFIDENCE INTERPRETATION
    # ====================================

    if confidence >= 90:

        explanation_parts.append(
            "The model showed high confidence during contextual emotional interpretation."
        )

    elif confidence >= 70:

        explanation_parts.append(
            "Moderate contextual emotional confidence was observed."
        )

    else:

        explanation_parts.append(
            "Some emotional overlap was observed during prediction."
        )

    return " ".join(explanation_parts)

# ====================================
# SMART SUGGESTION ENGINE
# ====================================

def generate_suggestion(
    prediction,
    text
):

    (
        contexts,
        positive_detected,
        negative_detected
    ) = detect_context(text)

    suggestions = []

    if prediction == "Normal":

        suggestions.append(
            "Continue maintaining healthy routines, emotional balance, and positive social interaction."
        )

    elif prediction == "Stress":

        suggestions.append(
            "Consider stress-management strategies, emotional breaks, and maintaining work-life balance."
        )

    else:

        suggestions.append(
            "Consider emotional support, healthy routines, and discussing concerns with trusted individuals."
        )

    if positive_detected:

        suggestions.append(
            "Your response also reflects positive coping ability and emotional resilience."
        )

    if "career" in contexts:

        suggestions.append(
            "Career uncertainty can create emotional pressure. Focusing on controllable goals and professional planning may help reduce anxiety."
        )

    if "academic" in contexts:

        suggestions.append(
            "Academic stress may improve through structured scheduling and realistic study planning."
        )

    if "relationship" in contexts:

        suggestions.append(
            "Healthy communication and emotional support from trusted individuals may improve emotional stability."
        )

    if "financial" in contexts:

        suggestions.append(
            "Financial concerns may feel overwhelming during uncertainty. Practical planning and support systems may help reduce stress."
        )

    if "burnout" in contexts:

        suggestions.append(
            "Mental exhaustion may improve with proper rest, sleep, reduced workload, and relaxation activities."
        )

    if "health" in contexts:

        suggestions.append(
            "Maintaining physical health routines and proper consultation may help reduce emotional stress."
        )

    if "family" in contexts:

        suggestions.append(
            "Open communication with trusted family members may help improve emotional comfort and support."
        )

    return " ".join(suggestions)

# ====================================
# SMART FINAL CONCLUSION
# ====================================

def generate_final_conclusion(
    prediction,
    confidence,
    questionnaire_level,
    text
):

    (
        contexts,
        positive_detected,
        negative_detected
    ) = detect_context(text)

    if prediction == "Normal":

        conclusion = (
            "The overall emotional analysis indicates emotionally stable behavioral patterns."
        )

    elif prediction == "Stress":

        conclusion = (
            "The overall emotional analysis detected noticeable stress-related emotional patterns."
        )

    else:

        conclusion = (
            "The overall emotional analysis detected elevated emotional concern with depression-related indicators."
        )

    if positive_detected and negative_detected:

        conclusion += (
            " The response also reflected emotional resilience despite ongoing concerns."
        )

    elif positive_detected:

        conclusion += (
            " Positive emotional coping indicators were observed during analysis."
        )

    elif negative_detected:

        conclusion += (
            " Emotional concern indicators were consistently observed."
        )

    if contexts:

        conclusion += (
            " Contextual emotional signals related to "
            + ", ".join(contexts)
            + " were identified."
        )

    if confidence >= 90:

        conclusion += (
            " The NLP model showed high confidence during prediction."
        )

    elif confidence >= 70:

        conclusion += (
            " Moderate emotional confidence was observed during analysis."
        )

    else:

        conclusion += (
            " Some emotional overlap was detected during prediction."
        )

    conclusion += (
        f" Questionnaire assessment suggested {questionnaire_level.lower()}."
    )

    return conclusion

# ====================================
# SIDEBAR
# ====================================

with st.sidebar:

    st.header("📌 Project Information")

    st.success("Model Accuracy: 93.6%")

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

        total_score = (

            normal_score_map[q1]
            + normal_score_map[q2]
            + normal_score_map[q3]
            + normal_score_map[q4]

            + reverse_score_map[q5]
            + reverse_score_map[q6]

            + normal_score_map[q7]
        )

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
                predicted_label,
                confidence
            )

            suggestion = generate_suggestion(
                predicted_label,
                cleaned
            )

            final_conclusion = generate_final_conclusion(
                predicted_label,
                confidence,
                questionnaire_level,
                cleaned
            )

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
