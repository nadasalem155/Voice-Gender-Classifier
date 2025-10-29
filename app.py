import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# --- Preprocess ---
def preprocess_audio(filepath):
    wav, sr = librosa.load(filepath, sr=16000, mono=True)
    max_len = 48000
    if len(wav) > max_len:
        wav = wav[:max_len]
    else:
        wav = np.pad(wav, (0, max_len - len(wav)))
    spec = np.abs(librosa.stft(wav, n_fft=512, hop_length=256))
    spec = tf.image.resize(np.expand_dims(spec, -1), [128, 128])
    return np.expand_dims(spec, 0), wav, sr

# --- Predict ---
def predict_gender(filepath):
    X, _, _ = preprocess_audio(filepath)
    pred = model.predict(X, verbose=0)[0][0]
    return "Male" if pred > 0.5 else "Female"

# --- Session State ---
for k in ["up_path", "rec_path", "up_res", "rec_res"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ========================================
# UI
# ========================================
st.title("Voice Gender Recognition")

# --- Upload ---
st.subheader("Upload Audio")
uploaded = st.file_uploader("WAV/MP3/OGG", type=["wav", "mp3", "ogg"], key="up")

if uploaded:
    # تجنب التكرار
    if st.session_state.up_path is None or not os.path.exists(st.session_state.up_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded.read())
            st.session_state.up_path = f.name
        st.session_state.up_res = predict_gender(st.session_state.up_path)
        st.rerun()

# --- Record ---
st.subheader("Record Voice")
audio_bytes = audio_recorder(
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_name="microphone",
    icon_size="2x",
    key="rec"
)

if audio_bytes and st.session_state.rec_path is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name
    st.session_state.rec_path = path
    st.session_state.rec_res = predict_gender(path)
    st.rerun()

# ========================================
# عرض + إزالة فورية
# ========================================

# --- Uploaded Audio ---
if st.session_state.up_path and os.path.exists(st.session_state.up_path):
    st.success(f"Uploaded: **{st.session_state.up_res}**")
    
    _, wav, _ = preprocess_audio(st.session_state.up_path)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(wav, color="#1f77b4")
    ax.set_title("Waveform")
    st.pyplot(fig)
    plt.close(fig)  # مهم: إغلاق الشكل
    
    st.audio(st.session_state.up_path)

    if st.button("Remove Uploaded File", key="rm_up"):
        try:
            os.unlink(st.session_state.up_path)  # حذف فوري
            st.session_state.up_path = None
            st.session_state.up_res = None
            st.success("Uploaded file removed!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# --- Recorded Audio ---
if st.session_state.rec_path and os.path.exists(st.session_state.rec_path):
    st.success(f"Recorded: **{st.session_state.rec_res}**")
    
    _, wav, _ = preprocess_audio(st.session_state.rec_path)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(wav, color="#ff7f0e")
    ax.set_title("Waveform")
    st.pyplot(fig)
    plt.close(fig)
    
    st.audio(st.session_state.rec_path)

    if st.button("Remove Recorded Audio", key="rm_rec"):
        try:
            os.unlink(st.session_state.rec_path)
            st.session_state.rec_path = None
            st.session_state.rec_res = None
            st.success("Recorded audio removed!")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")