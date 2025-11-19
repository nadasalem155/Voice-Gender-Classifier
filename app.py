import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# === Optimization 1: Hide only the internal recorder audio tag (NOT all audio tags) ===
st.markdown("""
<style>
audio[src*="data:audio"] { display: none; }
</style>
""", unsafe_allow_html=True)

# --- Load Keras model once ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# --- Preprocess audio file ---
def preprocess_audio(filename, max_len=48000):
    try:
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
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# --- Predict gender from audio ---
def predict_gender(file_path):
    features, _, _ = preprocess_audio(file_path)
    if features is None:
        return None
    pred = model.predict(features)
    return "👨‍🦱 Male" if pred[0][0] > 0.5 else "👩‍🦰 Female"

# --- Initialize session state ---
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Streamlit UI ---
st.title("🎤 Voice Gender Recognition")
st.markdown("Detect whether a voice belongs to a **Male 👨‍🦱** or **Female 👩‍🦰** using a CNN model.")

# --- Upload audio file section ---
st.subheader("📂 Upload an Audio File")
uploaded_file = st.file_uploader("Choose a file (wav, mp3, ogg) 🎧", type=["wav", "mp3", "ogg"], key="file_uploader")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        st.session_state.uploaded_path = tmp_file.name
    
    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

# --- Display uploaded file result with waveform and audio ---
if st.session_state.uploaded_path and st.session_state.uploaded_result:
    st.success(f"Prediction (Uploaded): {st.session_state.uploaded_result}")
    
    try:
        spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)
        if wav is not None:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.plot(wav, color="#1f77b4")
            ax.set_title("📈 Waveform (Uploaded)")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
        st.audio(st.session_state.uploaded_path, format="audio/wav")
    except Exception as e:
        st.error(f"Error displaying uploaded audio: {e}")

# --- Remove button for uploaded file ---
if st.session_state.uploaded_path:
    if st.button("🗑️ Remove Uploaded File", key="remove_uploaded"):
        try:
            if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
                os.remove(st.session_state.uploaded_path)
            st.session_state.uploaded_path = None
            st.session_state.uploaded_result = None
            if "file_uploader" in st.session_state:
                del st.session_state["file_uploader"]
            st.success("Uploaded file removed successfully!")
            time.sleep(0.1)
            st.rerun()
        except Exception as e:
            st.error(f"Error removing file: {e}")

# --- Record audio section ---
st.subheader("🎤 Record Your Voice")
st.markdown("Click the microphone button to record your voice from the browser.")
audio_bytes = audio_recorder(key="audio_recorder")
if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        st.session_state.recorded_path = tmp_file.name
    
    st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)

# --- Display recorded audio result with waveform and audio ---
if st.session_state.recorded_path and st.session_state.recorded_result:
    st.success(f"Prediction (Recorded): {st.session_state.recorded_result}")
    
    try:
        spec, wav, sr = preprocess_audio(st.session_state.recorded_path)
        if wav is not None:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.plot(wav, color="#ff7f0e")
            ax.set_title("📈 Waveform (Recorded)")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
        st.audio(st.session_state.recorded_path, format="audio/wav")
    except Exception as e:
        st.error(f"Error displaying recorded audio: {e}")

# --- Remove button for recorded audio ---
if st.session_state.recorded_path:
    if st.button("🗑️ Remove Recorded Audio", key="remove_recorded"):
        try:
            if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
                os.remove(st.session_state.recorded_path)
            st.session_state.recorded_path = None
            st.session_state.recorded_result = None
            if "audio_recorder" in st.session_state:
                del st.session_state["audio_recorder"]
            st.success("Recorded audio removed successfully!")
            time.sleep(0.1)
            st.rerun()
        except Exception as e:
            st.error(f"Error removing recorded audio: {e}")