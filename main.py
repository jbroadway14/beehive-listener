import streamlit as st
import numpy as np
import scipy.fft
import sounddevice as sd
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="BeeHive Listener v2", layout="centered")
st.title("🐝 BeeHive Listener v2 - Learning Edition")

DATA_FILE = "hive_data.csv"
MODEL_FILE = "beehive_model.joblib"
DURATION = 5
SAMPLE_RATE = 44100

LABELS = ["Calm", "Agitated", "Queen_Piping", "Low_Rumble", "Unsure"]

# --- Audio + Features ---
def record_audio(duration, fs):
    st.info(f"Recording for {duration}s... keep it steady near entrance")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten().astype(np.float32)

def extract_features(audio, fs):
    N = len(audio)
    yf = np.abs(scipy.fft.rfft(audio))
    xf = scipy.fft.rfftfreq(N, 1 / fs)

    # Ignore <80Hz wind noise
    mask = xf >= 80
    xf, yf = xf[mask], yf[mask]

    if np.sum(yf) == 0:
        return None # too quiet

    peak_freq = xf[np.argmax(yf)]
    centroid = np.sum(xf * yf) / np.sum(yf) # "brightness"
    bandwidth = np.sqrt(np.sum(((xf - centroid)**2) * yf) / np.sum(yf)) # "spread"
    total_energy = np.sum(yf)

    return [peak_freq, centroid, bandwidth, total_energy]

# --- ML Model ---
def train_model():
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)
    df = df[df['label']!= 'Unsure'] # don't train on unsure
    if len(df) < 5: # need some data first
        return None

    X = df[['peak_freq', 'centroid', 'bandwidth', 'energy']].values
    y = df['label'].values
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_model()

# --- UI ---
col1, col2 = st.columns(2)
with col1:
    if st.button("🎙️ Record + Analyze"):
        st.session_state['audio'] = record_audio(DURATION, SAMPLE_RATE)
        feats = extract_features(st.session_state['audio'], SAMPLE_RATE)

        if feats is None:
            st.error("Too quiet or too much wind. Try again, closer to hive.")
            st.session_state['feats'] = None
        else:
            st.session_state['feats'] = feats
            peak_freq = feats[0]
            st.success(f"Peak: {peak_freq:.0f} Hz | Centroid: {feats[1]:.0f} Hz")

            model = load_model()
            if model:
                pred = model.predict([feats])[0]
                proba = np.max(model.predict_proba([feats]))
                st.write(f"**ML Guess: {pred.replace('_',' ')}** [{proba*100:.0f}% confidence]")
            else:
                # fallback rules until we have data
                if 380 <= peak_freq <= 520: pred = "Queen_Piping"
                elif 260 <= peak_freq <= 380: pred = "Agitated"
                elif 180 <= peak_freq <= 260: pred = "Calm"
                else: pred = "Low_Rumble"
                st.write(f"**Rule-based Guess: {pred.replace('_',' ')}** [Collect more data for ML]")

with col2:
    st.write("**1. Record first, then log what you saw:**")
    label = st.radio("What was the hive actually like?", LABELS, key="label", horizontal=True)

    if st.button("💾 Save This Sample") and 'feats' in st.session_state:
        if st.session_state['feats'] is None:
            st.warning("Record a good sample first")
        else:
            feats = st.session_state['feats']
            new_row = pd.DataFrame([[*feats, label]],
                                   columns=['peak_freq', 'centroid', 'bandwidth', 'energy', 'label'])
            if os.path.exists(DATA_FILE):
                df = pd.read_csv(DATA_FILE)
                df = pd.concat([df, new_row], ignore_index=True)
            else:
                df = new_row
            df.to_csv(DATA_FILE, index=False)
            st.success(f"Saved! You now have {len(df)} samples.")
            train_model() # retrain after each save
            st.rerun()

# --- Show your dataset ---
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    st.write(f"### Your Dataset: {len(df)} recordings")
    st.dataframe(df, hide_index=True)
