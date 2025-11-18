import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import time

# === تحميل الموديل مرة واحدة فقط ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("gender_voice_model.keras", compile=False)

model = load_model()

# === دالة المعالجة (مع كاشينج ذكي) ===
@st.cache_data(show_spinner=False)
def preprocess_and_predict(_model, audio_path):
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # تقسيم الصوت لـ 3 ثانية (أو نكمل بالطريقة القديمة لو حابة)
    max_len = 48000  # 3 ثواني عند 16kHz
    if len(wav) > max_len:
        wav = wav[:max_len]
    else:
        wav = np.pad(wav, (0, max_len - len(wav)))

    # تحويل لـ Spectrogram
    spec = np.abs(librosa.stft(wav, n_fft=512, hop_length=256))
    spec = np.expand_dims(spec, -1)
    spec = tf.image.resize(spec, [128, 128])
    spec = np.expand_dims(spec, 0)

    pred = _model.predict(spec, verbose=0)[0][0]
    gender = "Male" if pred > 0.5 else "Female"
    
    return gender, wav, sr

# === تهيئة session state
for key in ["uploaded_path", "recorded_path", "uploaded_result", "recorded_result", "uploaded_wav", "recorded_wav"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("Voice Gender Recognition")
st.markdown("اكتشف جنس الصوت بضغطة زر – سريع ودقيق!")

# ==============================
#       Upload Section
# ==============================
st.subheader("Upload an Audio File")
uploaded_file = st.file_uploader("اختر ملف صوتي (wav, mp3, ogg)", type=["wav", "mp3", "ogg"], key="uploader")

if uploaded_file is not None and (st.session_state.uploaded_path != uploaded_file.name or st.session_state.uploaded_result is None):
    # حفظ الملف مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    # معالجة وتوقع سريع باستخدام الكاش
    gender, wav, sr = preprocess_and_predict(model, temp_path)
    
    # حفظ في session state
    st.session_state.uploaded_path = temp_path
    st.session_state.uploaded_result = gender
    st.session_state.uploaded_wav = wav

    # حذف الملف القديم لو موجود
    if os.path.exists(getattr(st.session_state, "old_uploaded", "")):
        try: os.unlink(st.session_state.old_uploaded)
        except: pass
    st.session_state.old_uploaded = temp_path

# عرض نتيجة الـ Upload
if st.session_state.uploaded_result:
    st.success(f"Prediction (Uploaded): {st.session_state.uploaded_result}")
    st.audio(st.session_state.uploaded_path)
    
    plt.figure(figsize=(10, 3))
    plt.plot(st.session_state.uploaded_wav, color="#1f77b4")
    plt.title("Waveform - Uploaded Audio")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

# زر حذف الـ Upload
if st.session_state.uploaded_path:
    if st.button("Remove Uploaded File", key="del_up"):
        for key in ["uploaded_path", "uploaded_result", "uploaded_wav", "old_uploaded"]:
            if key in st.session_state:
                if key.endswith("path") and st.session_state[key] and os.path.exists(st.session_state[key]):
                    os.unlink(st.session_state[key])
                st.session[key] = None
        st.success("Uploaded file deleted!")
        st.rerun()

# ==============================
#       Record Section
# ==============================
st.subheader("Record Your Voice")

audio_bytes = audio_recorder(
    text="اضغط للتسجيل",
    recording_color="#e74c3c",
    neutral_color="#34495e",
    icon_name="microphone",
    icon_size="3x",
    key="recorder"
)

if audio_bytes and (st.session_state.recorded_path is None or st.session_state.recorded_result is None):
    # حفظ التسجيل مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        temp_path = f.name

    # معالجة فورية وسريعة جدًا بفضل الكاش
    gender, wav, sr = preprocess_and_predict(model, temp_path)

    # حفظ في session state
    st.session_state.recorded_path = temp_path
    st.session_state.recorded_result = gender
    st.session_state.recorded_wav = wav

# عرض نتيجة التسجيل مباشرة وبدون تأخير
if st.session_state.recorded_result:
    st.success(f"Prediction (Recorded): {st.session_state.recorded_result}")
    st.audio(st.session_state.recorded_path)

    plt.figure(figsize=(10, 3))
    plt.plot(st.session_state.recorded_wav, color="#ff7f0e")
    plt.title("Waveform - Recorded Audio")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

# زر حذف التسجيل – سريع جدًا
if st.session_state.recorded_path:
    if st.button("Remove Recorded Audio", key="del_rec"):
        for key in ["recorded_path", "recorded_result", "recorded_wav"]:
            if key in st.session_state and st.session_state[key] is not None:
                if key.endswith("path") and os.path.exists(st.session_state[key]):
                    os.unlink(st.session_state[key])
                st.session_state[key] = None
        st.success("Recording deleted!")
        st.rerun()

# تنظيف تلقائي للملفات المؤقتة عند إغلاق التطبيق (اختياري)
def cleanup():
    for path in [st.session_state.uploaded_path, st.session_state.recorded_path]:
        if path and os.path.exists(path):
            try: os.unlink(path)
            except: pass

import atexit
atexit.register(cleanup)