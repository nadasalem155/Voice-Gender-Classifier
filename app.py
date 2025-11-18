import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# --- تحميل الموديل مرة واحدة فقط ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# --- معالجة الصوت ---
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
        st.error(f"خطأ في معالجة الصوت: {e}")
        return None, None, None

# --- التنبؤ بالجنس (سريع جدًا دلوقتي) ---
def predict_gender(file_path):
    features, _, _ = preprocess_audio(file_path)
    if features is None:
        return None
    pred = model.predict(features, verbose=0)[0][0]  # verbose=0 عشان ما يطبعش لوج
    return "Male" if pred > 0.5 else "Female"

# --- تهيئة session state ---
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ==============================
#           العنوان
# ==============================
st.title("Voice Gender Recognition")
st.markdown("اكتشفي جنس الصوت فورًا – سواء رفع أو تسجيل!")

# ==============================
#        رفع الملف
# ==============================
st.subheader("رفع ملف صوتي")
uploaded_file = st.file_uploader("اختري ملف (wav, mp3, ogg)", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # حفظ الملف مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # لو الملف جديد (مش نفس اللي فات)
    if st.session_state.uploaded_path != temp_path:
        st.session_state.uploaded_path = temp_path
        st.session_state.uploaded_result = predict_gender(temp_path)

# عرض نتيجة الرفع
if st.session_state.uploaded_result:
    st.success(f"النتيجة (المرفوع): {st.session_state.uploaded_result}")
    
    spec, wav, sr = preprocess_audio(st.session_state.uploaded_path)
    if wav is not None:
        plt.figure(figsize=(10, 3))
        plt.plot(wav, color="#1f77b4")
        plt.title("شكل الموجة - الملف المرفوع")
        plt.xlabel("العينات")
        plt.ylabel("السعة")
        st.pyplot(plt)
        
        st.audio(st.session_state.uploaded_path)

# حذف الملف المرفوع
if st.session_state.uploaded_path:
    if st.button("حذف الملف المرفوع", key="remove_uploaded"):
        if os.path.exists(st.session_state.uploaded_path):
            os.remove(st.session_state.uploaded_path)
        st.session_state.uploaded_path = None
        st.session_state.uploaded_result = None
        st.success("تم حذف الملف بنجاح!")
        st.rerun()

# ==============================
#           التسجيل
# ==============================
st.subheader("سجلي صوتك الآن")
audio_bytes = audio_recorder(
    text="اضغطي هنا للتسجيل",
    recording_color="#e74c3c",
    neutral_color="#2ecc71",
    icon_name="microphone",
    icon_size="3x"
)

# لما يخلّص تسجيل
if audio_bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        new_recorded_path = tmp_file.name

    # لو ده تسجيل جديد (مش نفس اللي فات)
    if st.session_state.recorded_path != new_recorded_path:
        # نحذف التسجيل القديم لو موجود
        if st.session_state.recorded_path and os.path.exists(st.session_state.recorded_path):
            os.remove(st.session_state.recorded_path)
        
        st.session_state.recorded_path = new_recorded_path
        st.session_state.recorded_result = predict_gender(new_recorded_path)

# عرض نتيجة التسجيل فورًا
if st.session_state.recorded_result:
    st.success(f"النتيجة (المُسجل): {st.session_state.recorded_result}")
    
    spec, wav, sr = preprocess_audio(st.session_state.recorded_path)
    if wav is not None:
        plt.figure(figsize=(10, 3))
        plt.plot(wav, color="#ff7f0e")
        plt.title("شكل الموجة - التسجيل الصوتي")
        plt.xlabel("العينات")
        plt.ylabel("السعة")
        st.pyplot(plt)
        
        st.audio(st.session_state.recorded_path)

# حذف التسجيل
if st.session_state.recorded_path:
    if st.button("حذف التسجيل", key="remove_recorded"):
        if os.path.exists(st.session_state.recorded_path):
            os.remove(st.session_state.recorded_path)
        st.session_state.recorded_path = None
        st.session_state.recorded_result = None
        st.success("تم حذف التسجيل بنجاح!")
        st.rerun()

# تنظيف عند الخروج (اختياري)
import atexit
def cleanup_temp_files():
    for path in [st.session_state.uploaded_path, st.session_state.recorded_path]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass
atexit.register(cleanup_temp_files)