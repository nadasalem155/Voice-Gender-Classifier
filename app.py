import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# --- Load Model Once ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# --- Preprocess Audio ---
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

# --- Predict Gender ---
def predict_gender(file_path):
    features, _, _ = preprocess_audio(file_path)
    if features is None:
        return None
    pred = model.predict(features, verbose=0)
    return "Male" if pred[0][0] > 0.5 else "Female"

# --- Session State ---
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ========================================
# 🎨 UI
# ========================================
st.title("Voice Gender Recognition")
st.markdown("Detect **Male** or **Female** voice instantly.")

# ========================================
# 📂 UPLOAD SECTION (سريع أصلاً)
# ========================================
st.subheader("Upload Audio File")
uploaded_file = st.file_uploader("Choose WAV/MP3/OGG", type=["wav", "mp3", "ogg"], key="uploader")

if uploaded_file is not None and st.session_state.uploaded_path != uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded_file.read())
        st.session_state.uploaded_path = f.name

    st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)

# Display Upload Result
if st.session_state.uploaded_path and st.session_state.uploaded_result:
    st.success(f"**Uploaded:** {st.session_state.uploaded_result}")
    spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)
    if wav is not None:
        plt.figure(figsize=(8, 2))
        plt.plot(wav, color="#1f77b4")
        plt.title("Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        st.pyplot(plt)
        st.audio(st.session_state.uploaded_path)

    if st.button("Remove Uploaded", key="rm_up"):
        os.unlink(st.session_state.uploaded_path)
        st.session_state.uploaded_path = st.session_state.uploaded_result = None
        st.rerun()

# ========================================
# 🎤 RECORD SECTION (مُحسّن للسرعة)
# ========================================
st.subheader("Record Your Voice")
st.markdown("*Click mic → speak → stop → instant result!*")

# استخدم key فريد + force rerun فوري
audio_bytes = audio_recorder(
    text="",
    recording_color="#e74c3c",
    neutral_color="#95a5a6",
    icon_name="microphone",
    icon_size="2x",
    key="recorder"
)

# === معالجة فورية بعد التسجيل ===
if audio_bytes and st.session_state.recorded_path != "processing":
    # تجنب التكرار
    if not st.session_state.recorded_path or not os.path.exists(st.session_state.recorded_path):
        st.session_state.recorded_path = "processing"  # قفل مؤقت
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name

        # معالجة فورية
        result = predict_gender(temp_path)
        st.session_state.recorded_path = temp_path
        st.session_state.recorded_result = result
        st.rerun()  # تحديث فوري

# === عرض النتيجة بعد المعالجة ===
if st.session_state.recorded_path and st.session_state.recorded_path != "processing" and st.session_state.recorded_result:
    st.success(f"**Recorded:** {st.session_state.recorded_result}")
    
    spec, wav, sr = preprocess_audio(st.session_state.recorded_path)
    if wav is not None:
        plt.figure(figsize=(8, 2))
        plt.plot(wav, color="#ff7f0e")
        plt.title("Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        st.pyplot(plt)
        st.audio(st.session_state.recorded_path)

    if st.button("Remove Recording", key="rm_rec"):
        os.unlink(st.session_state.recorded_path)
        st.session_state.recorded_path = st.session_state.recorded_result = None
        st.rerun()