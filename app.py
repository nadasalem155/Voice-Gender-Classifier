import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# --- Load Keras model once ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# --- Preprocess ---
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
        st.error(f"⚠ Error processing audio: {e}")
        return None, None, None

# --- Predict ---
def predict_gender(file_path):
    with st.spinner("🎧 Analyzing your voice... Please wait ⏳"):
        features, _, _ = preprocess_audio(file_path)
        if features is None:
            return None
        pred = model.predict(features, verbose=0)[0][0]
    return "👨 Male" if pred > 0.5 else "👩 Female"

# --- Session state ---
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- UI ---
st.title("🎙 Voice Gender Recognition")
st.markdown("Upload or record your voice to detect *Male 👨* or *Female 👩* using a CNN model.")

# ======================================================
# === 1. Upload ===
# ======================================================
st.subheader("📂 Upload Audio File")
uploaded_file = st.file_uploader(
    "🎵 Choose a .wav, .mp3, or .ogg file:",
    type=["wav", "mp3", "ogg"],
    key="upload_widget"
)

if uploaded_file is not None:
    if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
        os.unlink(st.session_state.recorded_path)
    st.session_state.recorded_path = None
    st.session_state.recorded_result = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.uploaded_path = tmp.name

    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

if st.session_state.uploaded_path and st.session_state.uploaded_result:
    st.success(f"✅ *Prediction (Uploaded):* {st.session_state.uploaded_result}")

    if os.path.exists(st.session_state.uploaded_path):
        spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)
        if wav is not None:
            plt.figure(figsize=(8, 2))
            plt.plot(wav, color="#1f77b4")
            plt.title("📈 Waveform (Uploaded)")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            st.pyplot(plt)
            st.audio(st.session_state.uploaded_path, format="audio/wav")

    # ✅ Fixed deletion logic
    if st.button("🗑 Remove Uploaded File", key="btn_remove_upload"):
        try:
            if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
                os.unlink(st.session_state.uploaded_path)
            st.session_state.uploaded_path = None
            st.session_state.uploaded_result = None
            st.success("✅ Uploaded file removed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"⚠ Failed to remove file: {e}")
        st.stop()

# ======================================================
# === 2. Record ===
# ======================================================
st.subheader("🎤 Record Your Voice")
st.markdown("Click the mic and speak for *2–5 seconds* 🕐")

audio_bytes = audio_recorder(
    text="🎙 Start Recording",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_name="microphone",
    icon_size="2x",
    key="record_widget"
)

if audio_bytes:
    if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
        os.unlink(st.session_state.uploaded_path)
    st.session_state.uploaded_path = None
    st.session_state.uploaded_result = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        st.session_state.recorded_path = tmp.name

    st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)

if st.session_state.recorded_path and st.session_state.recorded_result:
    st.success(f"✅ *Prediction (Recorded):* {st.session_state.recorded_result}")

    if os.path.exists(st.session_state.recorded_path):
        spec, wav, sr = preprocess_audio(st.session_state.recorded_path)
        if wav is not None:
            plt.figure(figsize=(8, 2))
            plt.plot(wav, color="#ff7f0e")
            plt.title("📊 Waveform (Recorded)")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            st.pyplot(plt)
            st.audio(st.session_state.recorded_path, format="audio/wav")

    # ✅ Fixed deletion logic
    if st.button("🗑 Remove Recorded Audio", key="btn_remove_record"):
        try:
            if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
                os.unlink(st.session_state.recorded_path)
            st.session_state.recorded_path = None
            st.session_state.recorded_result = None
            st.success("✅ Recording removed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"⚠ Failed to remove recording: {e}")
        st.stop()

# --- Footer ---
st.markdown("---")
st.caption("💡 Powered by Streamlit • 🧠 Model: CNN trained on STFT Spectrograms")