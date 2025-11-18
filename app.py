import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# === Hide default audio element for cleaner UI ===
st.markdown("<style>audio{display:none;}</style>", unsafe_allow_html=True)

# === Page config for better look ===
st.set_page_config(
    page_title="Voice Gender Classifier",
    page_icon="🎧",
    layout="centered",
)

# === Load model once ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# === Preprocess audio file ===
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

# === Predict gender ===
def predict_gender(file_path):
    features, _, _ = preprocess_audio(file_path)
    if features is None:
        return None
    pred = model.predict(features)
    return "👨 Male Voice" if pred[0][0] > 0.5 else "👩 Female Voice"

# === Initialize session state ===
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# === TITLE ===
st.markdown("<h1 style='text-align:center;'>🎧 Voice Gender Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Detect if the voice is Male or Female using AI 💡</p>", unsafe_allow_html=True)

# ===========================================================
# ================== UPLOAD AUDIO SECTION ===================
# ===========================================================
st.markdown("### 📂 Upload an Audio File")

with st.container():
    st.markdown(
        """
        <div style='border:1px solid #1f77b4; padding:15px; border-radius:10px;'>
        <p>Upload your audio file (wav, mp3, ogg) and get instant prediction!</p>
        </div>
        """, unsafe_allow_html=True
    )

uploaded_file = st.file_uploader("Choose an audio file:", type=["wav","mp3","ogg"], key="file_uploader")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.uploaded_path = tmp.name

    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

if st.session_state.uploaded_path and st.session_state.uploaded_result:
    with st.container():
        st.markdown(
            f"""
            <div style='border:2px solid #1f77b4; padding:15px; border-radius:15px; background-color:#e6f2ff;'>
            <h3>Prediction (Uploaded): {st.session_state.uploaded_result}</h3>
            </div>
            """, unsafe_allow_html=True
        )

        # waveform
        spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)
        if wav is not None:
            plt.figure(figsize=(10,2))
            plt.plot(wav, color="#1f77b4")
            plt.title("📈 Waveform")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            st.pyplot(plt)
            plt.close()
            st.audio(st.session_state.uploaded_path, format="audio/wav")

# Remove uploaded file
if st.session_state.uploaded_path:
    if st.button("🗑️ Remove Uploaded File"):
        try:
            if os.path.exists(st.session_state.uploaded_path):
                os.remove(st.session_state.uploaded_path)
            st.session_state.uploaded_path = None
            st.session_state.uploaded_result = None
            if "file_uploader" in st.session_state:
                del st.session_state["file_uploader"]
            st.success("Uploaded file removed!")
            time.sleep(0.1)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# ===========================================================
# ================== RECORD AUDIO SECTION ===================
# ===========================================================
st.markdown("### 🎤 Record Your Voice")
st.markdown("Click the microphone button to start recording:")

audio_bytes = audio_recorder(key="rec_button")

if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        st.cache_resource.clear()
        st.session_state.recorded_path = tmp.name

    st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)

if st.session_state.recorded_path and st.session_state.recorded_result:
    with st.container():
        st.markdown(
            f"""
            <div style='border:2px solid #ff7f0e; padding:15px; border-radius:15px; background-color:#fff2e6;'>
            <h3>Prediction (Recorded): {st.session_state.recorded_result}</h3>
            </div>
            """, unsafe_allow_html=True
        )

        spec, wav, sr = preprocess_audio(st.session_state.recorded_path)
        if wav is not None:
            plt.figure(figsize=(10,2))
            plt.plot(wav, color="#ff7f0e")
            plt.title("📈 Recorded Waveform")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            st.pyplot(plt)
            plt.close()
            st.audio(st.session_state.recorded_path, format="audio/wav")

# Remove recorded audio
if st.session_state.recorded_path:
    if st.button("🗑️ Remove Recording"):
        try:
            if os.path.exists(st.session_state.recorded_path):
                os.remove(st.session_state.recorded_path)
            st.session_state.recorded_path = None
            st.session_state.recorded_result = None
            if "rec_button" in st.session_state:
                del st.session_state["rec_button"]
            st.success("Recording removed.")
            time.sleep(0.1)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")