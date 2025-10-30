import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# --- Load Keras model once (cached) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# --- Preprocess audio: convert to spectrogram (128x128) ---
def preprocess_audio(filename, max_len=48000):
    try:
        wav, sr = librosa.load(filename, sr=16000, mono=True)
        # Trim or pad to fixed length
        if len(wav) > max_len:
            wav = wav[:max_len]
        else:
            wav = np.pad(wav, (0, max_len - len(wav)))

        # Create spectrogram
        spec = np.abs(librosa.stft(wav, n_fft=512, hop_length=256))
        spec = np.expand_dims(spec, -1)
        spec = tf.image.resize(spec, [128, 128])
        spec = np.expand_dims(spec, 0)  # Add batch dimension
        return spec, wav, sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# --- Predict gender from preprocessed audio ---
def predict_gender(file_path):
    with st.spinner("Analyzing voice..."):
        features, _, _ = preprocess_audio(file_path)
        if features is None:
            return None
        pred = model.predict(features, verbose=0)
        time.sleep(0.3)  # Tiny UX delay
    return "Male" if pred[0][0] > 0.5 else "Female"

# --- Initialize session state ---
if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None
if "recorded_path" not in st.session_state:
    st.session_state.recorded_path = None
if "uploaded_result" not in st.session_state:
    st.session_state.uploaded_result = None
if "recorded_result" not in st.session_state:
    st.session_state.recorded_result = None

# --- Streamlit UI ---
st.title("Voice Gender Detector")
st.markdown("Upload or record your voice to detect **Male** or **Female** using a CNN model.")

# ==================================================
# === 1. Upload Audio File ===
# ==================================================
st.subheader("Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose .wav, .mp3, or .ogg",
    type=["wav", "mp3", "ogg"],
    key="uploader_widget"  # Unique key
)

if uploaded_file is not None:
    # Clear recorded audio if uploading
    if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
        try:
            os.unlink(st.session_state.recorded_path)
        except:
            pass
    st.session_state.recorded_path = None
    st.session_state.recorded_result = None

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded_file.read())
        st.session_state.uploaded_path = f.name

    # Predict
    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

# --- Display uploaded result ---
if st.session_state.uploaded_path and st.session_state.uploaded_result:
    st.success(f"**Prediction (Uploaded):** {st.session_state.uploaded_result}")

    spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)
    if wav is not None:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(wav, color="#1f77b4")
        ax.set_title("Waveform (Uploaded)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        st.audio(st.session_state.uploaded_path, format="audio/wav")

    # --- Remove uploaded file ---
    if st.button("Remove Uploaded File", key="remove_upload"):
        if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
            os.unlink(st.session_state.uploaded_path)
        st.session_state.uploaded_path = None
        st.session_state.uploaded_result = None
        st.success("Uploaded file removed!")
        st.rerun()

# ==================================================
# === 2. Record Audio ===
# ==================================================
st.subheader("Record Your Voice")
st.markdown("Click the mic and speak clearly for 2–5 seconds.")

audio_bytes = audio_recorder(
    text="Start Recording",
    recording_color="#e74c3c",
    neutral_color="#95a5a6",
    icon_name="microphone",
    icon_size="2x",
    key="recorder_widget"
)

if audio_bytes:
    # Clear uploaded file if recording
    if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
        try:
            os.unlink(st.session_state.uploaded_path)
        except:
            pass
    st.session_state.uploaded_path = None
    st.session_state.uploaded_result = None

    # Save recorded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        st.session_state.recorded_path = f.name

    # Predict
    st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)

# --- Display recorded result ---
if st.session_state.recorded_path and st.session_state.recorded_result:
    st.success(f"**Prediction (Recorded):** {st.session_state.recorded_result}")

    spec, wav, sr = preprocess_audio(st.session_state.recorded_path)
    if wav is not None:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(wav, color="#ff7f0e")
        ax.set_title("Waveform (Recorded)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

        st.audio(st.session_state.recorded_path, format="audio/wav")

    # --- Remove recorded audio ---
    if st.button("Remove Recorded Audio", key="remove_record"):
        if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
            os.unlink(st.session_state.recorded_path)
        st.session_state.recorded_path = None
        st.session_state.recorded_result = None
        st.success("Recorded audio removed!")
        st.rerun()

# --- Footer ---
st.markdown("---")
