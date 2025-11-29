# 🎤 Voice Gender Recognition

This project is a **deep learning-based voice classifier** that predicts whether a recorded audio sample belongs to a **male** or **female** speaker.  
The model processes audio files, converts them into **spectrograms**, and uses a **CNN** to perform the classification.

---

## 🔗 Try the Web App
**Streamlit App:** [Click here](https://gender-voice-classifier.streamlit.app/) 🎧

**Dataset:** [Click here](https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal) 📂

---

## 🧰 Streamlit App Features

- **Upload or record audio** directly in the browser 🎙️  
- **Real-time gender prediction** 👨‍🦱👩‍🦰  
- **Waveform visualization** for each audio input 📈  
- **Remove uploaded or recorded audio safely** with a single button 🗑️  
- **Enhanced UI**: hidden internal recorder tag and larger buttons for easier use  

---

## 📚 Notebook Explanation

The notebook demonstrates the full workflow:

1. **Loading and preprocessing audio**:
    - Load audio at 16 kHz, convert to mono  
    - Trim or pad to **48,000 samples**  
    - Convert to **spectrogram** with `librosa.stft`  
    - Resize to `128x128` and add channel & batch dimensions

2. **Preparing the dataset**:
    - Load **female and male audio files**  
    - Randomly select **half of each class** for training  
    - Convert all audio to spectrograms  
    - Split into **train (70%)** and **test (30%)** sets  
    - Batch and prefetch using TensorFlow Dataset API

3. **Model definition**:
    - Simple **CNN** with 2 Conv2D + MaxPooling layers  
    - Flatten → Dense(64) → Dense(1) with **sigmoid activation**  
    - Compile with **binary cross-entropy**, track **Precision & Recall**

4. **Training**:
    - **EarlyStopping** and **ReduceLROnPlateau** callbacks  
    - Train for 6 epochs (adjustable)  
    - Save model as `gender_voice_model.keras`

5. **Prediction**:
    - Preprocess new audio  
    - Predict gender with the trained CNN  
    - Output **👨 Male** or **👩 Female**

---


## 📌 Notes

The Streamlit app loads the model once for efficiency using @st.cache_resource.

Audio files are temporarily stored for processing and can be removed safely with the 🗑️ button.

Waveform colors:

Uploaded audio: #FF6F61

Recorded audio: #4CAF50


Quiet environment and a few seconds of speech improve prediction accuracy.

Spectrogram resizing ensures uniform input shape (128x128x1) for CNN.


---

ℹ️ How to Use (Sidebar)

1️⃣ Upload a file:

Click on 'Browse files' 📁

Supported formats: wav, mp3, ogg

Wait a few seconds to get the prediction ✅


2️⃣ Record your voice:

Click the microphone 🎙️

Speak clearly for better results

Wait for analysis ⏳


3️⃣ Remove audio:

Use the 🗑️ button to delete uploaded or recorded audio

This will reset the interface


Tips:

Quiet environment = more accurate prediction

Speak a few seconds, not just 1 word

Male voice → 👨, Female voice → 👩