import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# Hide default audio tag for faster UI
st.markdown("<style>audio{display:none;}</style>", unsafe_allow_html=True)

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# Preprocess audio
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
        return spec.astype(np.float32), wav, sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# Predict gender with confidence
def predict_gender(file_path):
    features, _, _ = preprocess_audio(file_path)
    if features is None:
        return None, None
    pred = model.predict(features, verbose=0)[0][0]
    if pred > 0.5:
        return "Man", pred
    else:
        return "Woman", 1 - pred

# Initialize session state
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# Header
# =========================
st.title("Voice Gender Recognition")
st.markdown("### Detect whether a voice is **Male** or **Female** using AI")

# =========================
# Upload Section
# =========================
st.markdown("---")
st.subheader("Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose a file (wav, mp3, ogg)",
    type=["wav", "mp3", "ogg"],
    key="uploader"
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded_file.read())
        st.session_state.uploaded_path = f.name
    
    label, conf = predict_gender(st.session_state.uploaded_path)
    st.session_state.uploaded_result = (label, conf)

if st.session_state.uploaded_path and st.session_state.uploaded_result:
    label, conf = st.session_state.uploaded_result
    
    st.success(f"### Result: {label}")
    st.caption(f"Confidence: {conf:.1%}")

    _, wav, _ = preprocess_audio(st.session_state.uploaded_path)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(wav, color="#2E86AB", linewidth=1)
        ax.set_title("Waveform", fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    with col2:
        st.audio(st.session_state.uploaded_path)

    if st.button("Remove Uploaded File", key="del_up"):
        os.unlink(st.session_state.uploaded_path)
        st.session_state.uploaded_path = None
        st.session_state.uploaded_result = None
        st.success("File removed!")
        time.sleep(0.1)
        st.rerun()

# =========================
# Record Section
# =========================
st.markdown("---")
st.subheader("Record Your Voice")
st.write("Click the microphone, speak, then stop – instant analysis!")

audio_bytes = audio_recorder(
    text="Click to record",
    recording_color="#e74c3c",
    neutral_color="#34495e",
    icon_size="3x",
    key="recorder"   # ← Fixed line
)

if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        st.session_state.recorded_path = f.name
    
    label, conf = predict_gender(st.session_state.recorded_path)
    st.session_state.recorded_result = (label, conf)

if st.session_state.recorded_path and st.session_state.recorded_result:
    label, conf = st.session_state.recorded_result
    
    st.success(f"### Result: {label}")
    st.caption(f"Confidence: {conf:.1%}")

    _, wav, _ = preprocess_audio(st.session_state.recorded_path)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(wav, color="#E67E22", linewidth=1)
        ax.set_title("Recorded Waveform", fontsize=14, pad=15)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    with col2:
        st.audio(st.session_state.recorded_path)

    if st.button("Remove Recording", key="del_rec"):
        os.unlink(st.session_state.recorded_path)
        st.session_state.recorded_path = None
        st.session_state.recorded_result = None
        st.success("Recording removed!")
        time.sleep(0.1)
        st.rerun()

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Made with Streamlit • Powered by a CNN trained on voice spectrograms")