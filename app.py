import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# إعداد صفحة Streamlit لتحسين الأداء
st.set_page_config(
    page_title="Voice Gender Recognition",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Load Keras model once ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("gender_voice_model.keras", compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# --- Audio preprocessing ---
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

# --- Gender prediction ---
def predict_gender(file_path):
    if model is None:
        st.error("❌ Model not available. Please check if the model file exists.")
        return None
        
    with st.spinner("🎧 Analyzing your voice... Please wait ⏳"):
        features, _, _ = preprocess_audio(file_path)
        if features is None:
            return None
        pred = model.predict(features, verbose=0)[0][0]
    return "👨 Male" if pred > 0.5 else "👩 Female"

# --- Initialize session state ---
def init_session_state():
    keys = [
        "uploaded_path", "recorded_path", "uploaded_result", "recorded_result",
        "remove_uploaded", "remove_recorded", "clear_all"
    ]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

init_session_state()

# --- Cleanup temporary files ---
def cleanup_files():
    files_to_remove = []
    
    if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
        files_to_remove.append(st.session_state.uploaded_path)
    
    if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
        files_to_remove.append(st.session_state.recorded_path)
    
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except:
            pass

# --- Handle remove actions ---
def handle_remove_actions():
    # Handle uploaded file removal
    if st.session_state.remove_uploaded:
        if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
            try:
                os.remove(st.session_state.uploaded_path)
                st.success("✅ Uploaded file removed successfully!")
            except Exception as e:
                st.error(f"⚠ Failed to remove uploaded file: {e}")
        
        st.session_state.uploaded_path = None
        st.session_state.uploaded_result = None
        st.session_state.remove_uploaded = False
        time.sleep(0.5)
        st.rerun()
    
    # Handle recorded audio removal
    if st.session_state.remove_recorded:
        if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
            try:
                os.remove(st.session_state.recorded_path)
                st.success("✅ Recording removed successfully!")
            except Exception as e:
                st.error(f"⚠ Failed to remove recording: {e}")
        
        st.session_state.recorded_path = None
        st.session_state.recorded_result = None
        st.session_state.remove_recorded = False
        time.sleep(0.5)
        st.rerun()
    
    # Handle clear all
    if st.session_state.clear_all:
        cleanup_files()
        st.session_state.uploaded_path = None
        st.session_state.recorded_path = None
        st.session_state.uploaded_result = None
        st.session_state.recorded_result = None
        st.success("✅ All files cleared successfully!")
        st.session_state.clear_all = False
        time.sleep(0.5)
        st.rerun()

# --- UI Header ---
st.title("🎙️ Voice Gender Recognition")
st.markdown("Upload or record your voice to detect **Male 👨** or **Female 👩** using a CNN model.")

# Handle remove actions first
handle_remove_actions()

# استخدام علامات تبويب لتقسيم الواجهة
tab1, tab2 = st.tabs(["📂 Upload Audio", "🎤 Record Voice"])

with tab1:
    # ======================================================
    # === 1. Upload Section ===
    # ======================================================
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose a .wav, .mp3, or .ogg file:",
        type=["wav", "mp3", "ogg"],
        key="upload_widget"
    )

    if uploaded_file is not None:
        # Clear any previous recording
        if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
            try:
                os.remove(st.session_state.recorded_path)
            except:
                pass
            st.session_state.recorded_path = None
            st.session_state.recorded_result = None

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.uploaded_path = tmp.name

        # Predict gender
        st.session_state.uploaded_result = predict_gender(st.session_state.uploaded_path)
        st.rerun()

    # Display results for uploaded file
    if st.session_state.uploaded_path and st.session_state.uploaded_result:
        st.success(f"**Prediction (Uploaded):** {st.session_state.uploaded_result}")

        if os.path.exists(st.session_state.uploaded_path):
            spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)
            if wav is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.plot(wav, color="#1f77b4")
                    ax.set_title("Waveform (Uploaded)")
                    ax.set_xlabel("Samples")
                    ax.set_ylabel("Amplitude")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.audio(st.session_state.uploaded_path, format="audio/wav")

        # --- Remove uploaded file ---
        if st.button("🗑 Remove Uploaded File", key="btn_remove_upload"):
            st.session_state.remove_uploaded = True

with tab2:
    # ======================================================
    # === 2. Recording Section ===
    # ======================================================
    st.subheader("Record Your Voice")
    st.markdown("Click the mic and speak for **2-5 seconds** 🕐")

    audio_bytes = audio_recorder(
        text="🎙️ Start Recording",
        recording_color="#e74c3c",
        neutral_color="#2c3e50",
        icon_name="microphone",
        icon_size="2x",
        key="record_widget"
    )

    if audio_bytes:
        # Clear uploaded audio if it exists
        if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
            try:
                os.remove(st.session_state.uploaded_path)
            except:
                pass
            st.session_state.uploaded_path = None
            st.session_state.uploaded_result = None

        # Save recording temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            st.session_state.recorded_path = tmp.name

        # Predict gender
        st.session_state.recorded_result = predict_gender(st.session_state.recorded_path)
        st.rerun()

    # Display results for recorded audio
    if st.session_state.recorded_path and st.session_state.recorded_result:
        st.success(f"**Prediction (Recorded):** {st.session_state.recorded_result}")

        if os.path.exists(st.session_state.recorded_path):
            spec, wav, sr = preprocess_audio(st.session_state.recorded_path)
            if wav is not None:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.plot(wav, color="#ff7f0e")
                    ax.set_title("Waveform (Recorded)")
                    ax.set_xlabel("Samples")
                    ax.set_ylabel("Amplitude")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.audio(st.session_state.recorded_path, format="audio/wav")

        # --- Remove recorded audio ---
        if st.button("🗑 Remove Recorded Audio", key="btn_remove_record"):
            st.session_state.remove_recorded = True

# --- Clear All Button ---
st.markdown("---")
if st.button("🗑 Clear All Files", type="secondary", key="btn_clear_all"):
    st.session_state.clear_all = True

# --- Footer ---
st.markdown("---")
st.caption("💡 Powered by Streamlit • 🧠 Model: CNN trained on STFT Spectrograms")

# تنظيف الملفات المؤقتة عند إغلاق التطبيق
import atexit
atexit.register(cleanup_files)

# حل إضافي للمشاكل المستعصية
st.markdown("---")
with st.expander("🛠️ Troubleshooting Section"):
    st.write("إذا استمرت مشاكل المسح، جرب:")
    
    if st.button("🔄 Force Refresh App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ App refreshed successfully!")
        time.sleep(1)
        st.experimental_rerun()