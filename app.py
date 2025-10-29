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
keys = ["up_path", "rec_path", "up_res", "rec_res"]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ========================================
# UI
# ========================================
st.title("Voice Gender Recognition")

# --- Upload ---
st.subheader("Upload Audio")
uploaded = st.file_uploader("WAV/MP3/OGG", type=["wav", "mp3", "ogg"], key="up")

if uploaded and st.session_state.up_path != uploaded.name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded.read())
        st.session_state.up_path = f.name
    st.session_state.up_res = predict_gender(st.session_state.up_path)

# --- Record ---
st.subheader("Record Voice")
audio_bytes = audio_recorder(
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_name="microphone",
    icon_size="2x",
    key="rec"
)

# معالجة التسجيل فور الانتهاء (بدون rerun متكرر)
if audio_bytes and st.session_state.rec_path is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name
    # كل حاجة في rerun واحد
    st.session_state.rec_path = path
    st.session_state.rec_res = predict_gender(path)
    st.rerun()  # rerun واحد فقط

# --- Display Upload ---
if st.session_state.up_path:
    st.success(f"Uploaded: **{st.session_state.up_res}**")
    _, wav, _ = preprocess_audio(st.session_state.up_path)
    plt.figure(figsize=(8,2))
    plt.plot(wav, color="#1f77b4")
    plt.title("Waveform")
    st.pyplot(plt)
    st.audio(st.session_state.up_path)

    if st.button("Remove Upload", key="rm_up"):
        os.unlink(st.session_state.up_path)
        st.session_state.up_path = st.session_state.up_res = None
        st.rerun()

# --- Display Record ---
if st.session_state.rec_path:
    st.success(f"Recorded: **{st.session_state.rec_res}**")
    _, wav, _ = preprocess_audio(st.session_state.rec_path)
    plt.figure(figsize=(8,2))
    plt.plot(wav, color="#ff7f0e")
    plt.title("Waveform")
    st.pyplot(plt)
    st.audio(st.session_state.rec_path)

    if st.button("Remove Record", key="rm_rec"):
        os.unlink(st.session_state.rec_path)
        st.session_state.rec_path = st.session_state.rec_res = None
        st.rerun()