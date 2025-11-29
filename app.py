import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# === Optimization: Hide only internal recorder audio tag ===
st.markdown("""
<style>
audio[src*="data:audio"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- Load model once ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# Warm-up
dummy = np.zeros((1, 128, 128, 1))
_ = model.predict(dummy, verbose=0)

# --- Preprocess ---
def preprocess_audio(filename, max_len=48000):
    wav, sr = librosa.load(filename, sr=16000, mono=True)
    if len(wav) > max_len:
        wav = wav[:max_len]
    else:
        wav = np.pad(wav, (0, max_len - len(wav)))

    spec = np.abs(librosa.stft(wav, n_fft=512, hop_length=256))
    spec = np.expand_dims(spec, -1)
    spec = tf.image.resize(spec, [128, 128])
    spec = np.expand_dims(spec, 0)

    return spec, wav, sr

# Predict
def predict_gender(path):
    features, _, _ = preprocess_audio(path)
    pred = model.predict(features, verbose=0)
    return "👨‍🦱 Male" if pred[0][0] > 0.5 else "👩‍🦰 Female"

# Session state
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result", "record_pending"]:
    st.session_state.setdefault(key, None)

st.title("🎤 Voice Gender Recognition")
st.markdown("Detect gender from voice using a deep learning model.")

# ============================================================
# 📂 Upload Section
# ============================================================
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.uploaded_path = tmp.name
    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

if st.session_state.uploaded_result:
    st.success(f"Prediction (Uploaded): {st.session_state.uploaded_result}")

# ============================================================
# 🎤 Record Section
# ============================================================
st.subheader("🎤 Record Your Voice")
audio_bytes = audio_recorder()

# STEP 1 — FIRST PASS → GET AUDIO AND TRIGGER RERUN
if audio_bytes and st.session_state.record_pending is None:
    st.session_state.record_pending = audio_bytes
    st.rerun()

# STEP 2 — SECOND PASS → SHOW LOADING FIRST THEN PROCESS
if st.session_state.record_pending is not None:
    with st.spinner("🔄 Analyzing your voice... Please wait"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(st.session_state.record_pending)
            st.session_state.recorded_path = tmp.name

        st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)

    st.session_state.record_pending = None

# DISPLAY RESULT
if st.session_state.recorded_result:
    st.success(f"Prediction (Recorded): {st.session_state.recorded_result}")
    st.audio(st.session_state.recorded_path)