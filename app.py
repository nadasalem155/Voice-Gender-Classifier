import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# --- Load Keras model once (cached for performance) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# --- Preprocess audio: pad/trim to 3 sec, convert to 128x128 spectrogram ---
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
        spec = np.expand_dims(spec, 0)  # Add batch dim
        return spec, wav, sr
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None

# --- Predict gender with confidence threshold ---
def predict_gender(file_path):
    with st.spinner("Analyzing voice..."):
        features, _, _ = preprocess_audio(file_path)
        if features is None:
            return None
        pred = model.predict(features, verbose=0)[0][0]
    return "Male" if pred > 0.5 else "Female"

# --- Initialize session state (only once) ---
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Streamlit UI ---
st.title("Voice Gender Recognition")
st.markdown("Upload or record your voice to detect **Male** or **Female** using a CNN model.")

# ======================================================
# === 1. Upload Audio File ===
# ======================================================
st.subheader("Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose .wav, .mp3, or .ogg file",
    type=["wav", "mp3", "ogg"],
    key="upload_widget"  # Unique key to avoid conflicts
)

if uploaded_file is not None:
    # Clear any previous recording when uploading
    if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
        os.unlink(st.session_state.recorded_path)
    st.session_state.recorded_path = None
    st.session_state.recorded_result = None

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.uploaded_path = tmp.name

    # Predict
    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

# --- Show uploaded result ---
if st.session_state.uploaded_path and st.session_state.uploaded_result:
    st.success(f"**Prediction (Uploaded):** {st.session_state.uploaded_result}")

    spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)
    if wav is not None:
        plt.figure(figsize=(8, 2))
        plt.plot(wav, color="#1f77b4")
        plt.title("Waveform (Uploaded)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        st.pyplot(plt)
        st.audio(st.session_state.uploaded_path, format="audio/wav")

    # --- Remove uploaded file (FAST & CLEAN) ---
    if st.button("Remove Uploaded File", key="btn_remove_upload"):
        if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
            os.unlink(st.session_state.uploaded_path)
        st.session_state.uploaded_path = None
        st.session_state.uploaded_result = None
        st.success("Uploaded file removed!")
        st.rerun()  # Instant refresh

# ======================================================
# === 2. Record Audio ===
# ======================================================
st.subheader("Record Your Voice")
st.markdown("Click the mic and speak for 2–5 seconds.")

audio_bytes = audio_recorder(
    text="Start Recording",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_name="microphone",
    icon_size="2x",
    key="record_widget"  # Unique key
)

if audio_bytes:
    # Clear any uploaded file when recording
    if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
        os.unlink(st.session_state.uploaded_path)
    st.session_state.uploaded_path = None
    st.session_state.uploaded_result = None

    # Save recorded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        st.session_state.recorded_path = tmp.name

    # Predict
    st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)

# --- Show recorded result ---
if st.session_state.recorded_path and st.session_state.recorded_result:
    st.success(f"**Prediction (Recorded):** {st.session_state.recorded_result}")

    spec, wav, sr = preprocess_audio(st.session_state.recorded_path)
    if wav is not None:
        plt.figure(figsize=(8, 2))
        plt.plot(wav, color="#ff7f0e")
        plt.title("Waveform (Recorded)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        st.pyplot(plt)
        st.audio(st.session_state.recorded_path, format="audio/wav")

    # --- Remove recorded audio (FAST & FINAL) ---
    if st.button("Remove Recorded Audio", key="btn_remove_record"):
        if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
            os.unlink(st.session_state.recorded_path)
        st.session_state.recorded_path = None
        st.session_state.recorded_result = None
        st.success("Recording removed!")
        st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("Powered by Streamlit • Model: CNN on STFT Spectrograms")