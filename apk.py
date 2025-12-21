import streamlit as st
import joblib as jb
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences   
import re

# --- Page Configuration ---
st.set_page_config(page_title="Emotion AI", page_icon="🎭", layout="centered")

# --- Custom CSS for a Polished Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .emotion-text {
        font-size: 200px;
        font-weight: bold;
        color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Resource Loading (Cached for performance) ---
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('emotion_model.h5')
    tokenizer = jb.load('tokenizer.jb')
    label_encoder = jb.load('label_encoder.jb')
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_resources()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+\#','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()

def predict_emotion(text):
    processed_text = preprocess_text(text)
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
    prediction = model.predict(padded_sequences)
    predicted_label = label_encoder.inverse_transform([tf.argmax(prediction, axis=1).numpy()[0]])[0]
    return predicted_label

# --- UI Header ---
st.title("🎭 Emotion Analysis AI")
st.markdown("Analyze the emotional tone of your text instantly using Deep Learning.")
st.divider()

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("Input Text", placeholder="How are you feeling today?", height=150)

with col2:
    st.info("**Instructions:** \n1. Paste your text. \n2. Click Predict. \n3. View the detected emotion.")

if st.button("Predict Emotion"):
    if user_input.strip():
        with st.spinner('Analyzing sentiment...'):
            prediction = predict_emotion(user_input)
            
            # --- Results Display ---
            st.success("Analysis Complete!")
            st.markdown(f"""
                <div class="prediction-card">
                    <p style="margin-bottom:0;">The detected emotion is:</p>
                    <p class="emotion-text">{prediction.upper()}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Visual flair based on emotion
            if prediction.lower() in ['joy', 'happy', 'love']:
                st.balloons()
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# --- Footer ---
st.markdown("---")
st.caption("Powered by TensorFlow & Streamlit")