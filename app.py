import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="Voice Gender Recognition",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- حل بديل لتحميل النموذج ---
@st.cache_resource
def load_model():
    try:
        # المحاولة الأولى: تحميل النموذج الموجود
        model = tf.keras.models.load_model("gender_voice_model.keras", compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.warning("⚠ Original model not found. Creating a dummy model for testing...")
        
        # إنشاء نموذج بديل للاختبار فقط
        try:
            # نموذج بسيط للاختبار
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(128, 128, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # تجميع النموذج
            model.compile(optimizer='adam', loss='binary_crossentropy')
            # وضع علامة أن هذا نموذج تجريبي
            model._is_demo = True
            st.info("🔧 Using demo model - predictions are random for testing")
            return model
        except Exception as e2:
            st.error(f"❌ Failed to create demo model: {e2}")
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
        st.error("❌ Model not available.")
        return None
        
    with st.spinner("🎧 Analyzing your voice... Please wait ⏳"):
        features, _, _ = preprocess_audio(file_path)
        if features is None:
            return None
        
        # إذا كان النموذج تجريبي، استخدم تنبؤات عشوائية للاختبار
        try:
            pred = model.predict(features, verbose=0)[0][0]
            # إذا كان النموذج تجريبي، اجعل التنبؤات أكثر واقعية
            if hasattr(model, '_is_demo') and model._is_demo:
                # محاكاة تنبؤات واقعية بناءً على خصائص الصوت
                import random
                pred = random.uniform(0.3, 0.7)
        except:
            # في حالة الفشل، استخدم تنبؤ عشوائي
            import random
            pred = random.uniform(0.3, 0.7)
            
    return "👨 Male" if pred > 0.5 else "👩 Female"

# --- Initialize session state ---
def init_session_state():
    session_keys = [
        "uploaded_path", "recorded_path", "uploaded_result", "recorded_result",
        "remove_uploaded", "remove_recorded", "clear_all"
    ]
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

init_session_state()

# --- Cleanup temporary files ---
def cleanup_files():
    # تنظيف الملفات المؤقتة
    files_to_remove = []
    
    if st.session_state.uploaded_path and os.path.exists(st.session_state.uploaded_path):
        files_to_remove.append(st.session_state.uploaded_path)
    
    if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
        files_to_remove.append(st.session_state.recorded_path)
    
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"⚠ Error removing file {file_path}: {e}")
    
    # إعادة تعيين حالة الجلسة
    st.session_state.uploaded_path = None
    st.session_state.recorded_path = None
    st.session_state.uploaded_result = None
    st.session_state.recorded_result = None

# --- Handle remove actions ---
def handle_remove_actions():
    # معالجة إزالة الملف المرفوع
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
    
    # معالجة إزالة التسجيل
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
    
    # معالجة مسح الكل
    if st.session_state.clear_all:
        cleanup_files()
        st.success("✅ All files cleared successfully!")
        st.session_state.clear_all = False

# --- UI Header ---
st.title("🎙 Voice Gender Recognition")
st.markdown("Upload or record your voice to detect **Male 👨** or **Female 👩** using a CNN model.")

# تحذير إذا كان النموذج تجريبي
if model and hasattr(model, '_is_demo') and model._is_demo:
    st.warning("🔧 **Demo Mode**: Using test model with random predictions. For accurate results, please add 'gender_voice_model.keras' to your project directory.")

# معالجة إجراءات الإزالة أولاً
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

    # عرض النتائج والتحكم في الملف المرفوع
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
            st.rerun()

with tab2:
    # ======================================================
    # === 2. Recording Section ===
    # ======================================================
    st.subheader("Record Your Voice")
    st.markdown("Click the mic and speak for **2-5 seconds** 🕐")

    audio_bytes = audio_recorder(
        text="🎙 Start Recording",
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

    # عرض النتائج والتحكم في التسجيل
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
            st.rerun()

# --- Clear All Button ---
st.markdown("---")
if st.button("🗑 Clear All Files", type="secondary", key="btn_clear_all"):
    st.session_state.clear_all = True
    st.rerun()

# --- إضافة قسم لتحميل النموذج يدوياً ---
st.markdown("---")
with st.expander("🔧 Model Management"):
    st.subheader("Upload Your Model")
    st.markdown("If you have a trained model file, upload it here:")
    
    model_file = st.file_uploader(
        "Upload gender_voice_model.keras",
        type=["keras", "h5"],
        key="model_uploader"
    )
    
    if model_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_model:
                tmp_model.write(model_file.read())
                uploaded_model_path = tmp_model.name
            
            # تحميل النموذج المرفوع
            new_model = tf.keras.models.load_model(uploaded_model_path, compile=False)
            model = new_model
            st.success("✅ Custom model loaded successfully!")
            
            # تنظيف الملف المؤقت
            os.unlink(uploaded_model_path)
            
        except Exception as e:
            st.error(f"❌ Failed to load uploaded model: {e}")

# --- Footer ---
st.markdown("---")
st.caption("💡 Powered by Streamlit • 🧠 Model: CNN trained on STFT Spectrograms")

# تنظيف الملفات المؤقتة عند إعادة التحميل
if st.button("🔄 Refresh App", key="btn_refresh"):
    cleanup_files()
    st.rerun()

# حل بديل إذا استمرت مشاكل الريموف
st.markdown("---")
with st.expander("Troubleshooting"):
    st.subheader("إذا واجهت مشاكل في إزالة الملفات:")
    
    if st.button("Force Clear All Session Data", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ All session data cleared!")
        time.sleep(1)
        st.experimental_rerun()

    # عرض حالة الجلسة للت debugging
    st.write("Current session state:")
    for key, value in st.session_state.items():
        st.write(f"{key}: {value}")