import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import io
import time
from datetime import datetime
import os
import csv

st.set_page_config(page_title="Beehive Listener v3.1", layout="wide")
st.title("🐝 Beehive Listener v3.1 - Data Collector")
st.caption("Record 10s of audio + log conditions. Builds your hive dataset.")

DATA_FILE = "hive_logs.csv"

def analyze_audio(audio_bytes):
    sample_rate, data = wavfile.read(io.BytesIO(audio_bytes))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    data = data - np.mean(data)
    
    N = len(data)
    yf = rfft(data)
    xf = rfftfreq(N, 1 / sample_rate)
    idx = np.argmax(np.abs(yf))
    peak_hz = xf[idx]
    
    magnitudes = np.abs(yf)
    centroid = np.sum(xf * magnitudes) / np.sum(magnitudes)
    rms = np.sqrt(np.mean(data**2))
    
    low_band_energy = np.mean(magnitudes[xf < 80])
    total_energy = np.mean(magnitudes)
    wind_warning = (low_band_energy / total_energy) > 0.4

    return round(peak_hz, 1), round(centroid, 1), round(rms, 5), wind_warning

def save_log(row):
    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# --- UI ---
st.error("⚠️ DO NOT TOUCH OR OPEN THE HIVE BEFORE RECORDING")
st.info("Approach slowly. Place phone 10-20cm from entrance. Wait 5s of silence before recording.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Record Audio")
    
    if st.button("⏳ Start 5s Countdown Then Record"):
        placeholder = st.empty()
        for i in range(5, 0, -1):
            placeholder.warning(f"Stand still... Recording starts in {i}s. Do not touch hive.")
            time.sleep(1)
        placeholder.empty()
        st.session_state.start_rec = True
    
    if st.session_state.get("start_rec", False):
        audio_bytes = st.audio_input("Recording now... Hold still for 10s")
        if audio_bytes:
            st.session_state.audio_bytes = audio_bytes
            st.session_state.start_rec = False
    else:
        audio_bytes = st.session_state.get("audio_bytes", None)

with col2:
    st.subheader("2. Log Conditions")
    temp_c = st.slider("Temperature °C", 5, 40, 20)
    time_of_day = st.selectbox("Time of Day", ["Morning 6-11", "Afternoon 11-4", "Evening 4-8"])
    weather = st.selectbox("Weather", ["Sunny/Calm", "Windy", "Cloudy", "Raining"])
    strength = st.selectbox("Hive Strength Guess", ["Weak/Nuc", "Medium", "Booming/Double"])
    label = st.selectbox("Ground Truth - What you saw", 
                         ["Calm/Foraging", "Agitated/Defensive", "Queenless Roar", 
                          "Queen Piping", "Swarm Prep", "Robbing", "Unknown"])

if audio_bytes:
    with st.spinner("Analyzing..."):
        peak_hz, centroid, rms, wind = analyze_audio(audio_bytes.getvalue())
    
    st.subheader("3. Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Peak Hz", f"{peak_hz} Hz")
    m2.metric("Centroid", f"{centroid} Hz")
    m3.metric("Energy", f"{rms}")
    
    if wind:
        st.warning("⚠️ High wind/rain noise detected <80Hz. Data may be noisy.")
    
    if st.button("✅ Save This Log", type="primary"):
        log_row = {
            "timestamp": datetime.now().isoformat(),
            "peak_hz": peak_hz, "centroid_hz": centroid, "rms_energy": rms,
            "temp_c": temp_c, "time_of_day": time_of_day, "weather": weather,
            "strength": strength, "label": label, "wind_flag": wind
        }
        save_log(log_row)
        st.success(f"Saved! Total logs: {sum(1 for _ in open(DATA_FILE)) - 1}")
        st.session_state.audio_bytes = None # Clear for next hive

st.divider()
if st.checkbox("Show all saved logs"):
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "hive_data.csv")
    else:
        st.info("No logs yet. Record + Save your first hive.")
