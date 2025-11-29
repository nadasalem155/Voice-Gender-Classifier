import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

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

# === SPEED UP PREDICTION (Warm-up model once) ===
dummy = np.zeros((1, 128, 128, 1))
_ = model.predict(dummy, verbose=0)

# --- Efficient Preprocessing ---
def preprocess_audio(filename, max_len=48000):
    try:
        wav, sr = librosa.load(filename, sr=16000, mono=True)

        # Pad/Trim
        if len(wav) > max_len:
            wav = wav[:max_len]
        else:
            wav = np.pad(wav, (0, max_len - len(wav)))

        # Mel spectrogram
        spec = np.abs(librosa.stft(wav, n_fft=512, hop_length=256))
        spec = np.expand_dims(spec, -1)
        spec = tf.image.resize(spec, [128, 128])
        spec = np.expand_dims(spec, 0)

        return spec, wav, sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# --- Predict gender ---
def predict_gender(path):
    features, _, _ = preprocess_audio(path)
    if features is None:
        return None
    pred = model.predict(features, verbose=0)
    return "👨‍🦱 Male" if pred[0][0] > 0.5 else "👩‍🦰 Female"

# --- Session state ---
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    st.session_state.setdefault(key, None)

# --- UI ---
st.title("🎤 Voice Gender Recognition")
st.markdown("Detect gender from voice using a deep learning model.")

# ============================================================
# 📂 Upload Section
# ============================================================
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"], key="file_uploader")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.uploaded_path = tmp.name
    
    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

# ---- Display uploaded results ----
if st.session_state.uploaded_path and st.session_state.uploaded_result:
    st.success(f"Prediction (Uploaded): {st.session_state.uploaded_result}")

    try:
        spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(wav)
        ax.set_title("📈 Waveform (Uploaded)")
        st.pyplot(fig)

        st.audio(st.session_state.uploaded_path)

    except Exception as e:
        st.error(f"Error displaying uploaded audio: {e}")

# ---- Remove uploaded ----
if st.session_state.uploaded_path:
    if st.button("🗑️ Remove Uploaded File"):
        try:
            if os.path.exists(st.session_state.uploaded_path):
                os.remove(st.session_state.uploaded_path)
            st.session_state.uploaded_path = None
            st.session_state.uploaded_result = None
            del st.session_state["file_uploader"]
            st.rerun()
        except Exception as e:
            st.error(f"Error removing file: {e}")

# ============================================================
# 🎤 Record Section
# ============================================================
st.subheader("🎤 Record Your Voice")
st.markdown("Click the microphone to start recording.")

audio_bytes = audio_recorder(key="audio_recorder")

if audio_bytes:
    # === SHOW LOADING IMMEDIATELY ===
    with st.spinner("🔄 Analyzing your voice... Please wait"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            st.session_state.recorded_path = tmp.name

        st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)

# ---- Display recorded results ----
if st.session_state.recorded_path and st.session_state.recorded_result:
    st.success(f"Prediction (Recorded): {st.session_state.recorded_result}")

    try:
        spec, wav, sr = preprocess_audio(st.session_state.recorded_path)

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(wav)
        ax.set_title("📈 Waveform (Recorded)")
        st.pyplot(fig)

        st.audio(st.session_state.recorded_path)

    except Exception as e:
        st.error(f"Error displaying recorded audio: {e}")

# ---- Remove recording ----
if st.session_state.recorded_path:
    if st.button("🗑️ Remove Recorded Audio"):
        try:
            if os.path.exists(st.session_state.recorded_path):
                os.remove(st.session_state.recorded_path)
            st.session_state.recorded_path = None
            st.session_state.recorded_result = None
            del st.session_state["audio_recorder"]
            st.rerun()
        except Exception as e:
            st.error(f"Error removing recorded audio: {e}")