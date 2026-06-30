import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import io
from datetime import datetime
import os
import csv

st.set_page_config(page_title="Beehive Listener v3.5", layout="wide")
st.title("🐝 Beehive Listener v3.5 - Simple")

# Big mic CSS
st.markdown("""
<style>
div[data-testid="stAudioInput"] button {
    width: 100% !important;
    height: 120px !important;
    font-size: 24px !important;
    border-radius: 20px !important;
}
</style>
""", unsafe_allow_html=True)

DATA_FILE = "hive_logs.csv"

def interpret_sound(peak_hz, centroid, rms, wind):
    if wind: return "Noisy Data", "Wind/rain <80Hz is high. Re-record on a calmer day."
    activity = "Low" if rms < 0.005 else "Medium" if rms < 0.015 else "High"
    if peak_hz > 450 and centroid > 400: return "Agitated/Defensive", f"{activity} energy + High pitch {peak_hz}Hz. Guard bees alarmed."
    elif 250 <= peak_hz <= 350 and 200 <= centroid <= 300: return "Calm/Foraging", f"{activity} energy + Low hum {peak_hz}Hz. Normal working hive."
    elif peak_hz > 500 and rms > 0.01: return "Queenless Roar", f"{activity} energy + Very high {peak_hz}Hz. Desperate, uneven buzzing."
    elif 350 <= peak_hz <= 450 and centroid > 350: return "Swarm Prep/Piping", f"{activity} energy + Mid-high {peak_hz}Hz. Queen piping possible."
    else: return "Unknown Pattern", f"{activity} energy at {peak_hz}Hz. Log as 'Unknown'."

def analyze_audio(audio_bytes):
    sample_rate, data = wavfile.read(io.BytesIO(audio_bytes))
    if data.ndim > 1: data = data.mean(axis=1)
    data = data.astype(np.float32) - np.mean(data)
    N = len(data)
    yf = rfft(data)
    xf = rfftfreq(N, 1 / sample_rate)
    magnitudes = np.abs(yf)
    peak_hz = xf[np.argmax(magnitudes)]
    centroid = np.sum(xf * magnitudes) / np.sum(magnitudes)
    rms = np.sqrt(np.mean(data**2))
    wind_warning = (np.mean(magnitudes[xf < 80]) / np.mean(magnitudes)) > 0.4
    return round(peak_hz, 1), round(centroid, 1), round(rms, 5), wind_warning

def save_log(row):
    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(row)

st.error("⚠️ DO NOT TOUCH HIVE. Phone 10-20cm from entrance. Stand still.")
st.info("Tap the big mic → Hold 10s → Stop → Log conditions")

audio_bytes = st.audio_input(" ", key="recorder")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Log Conditions")
    temp_c = st.slider("Temp °C", 5, 40, 20)
    time_of_day = st.selectbox("Time", ["Morning 6-11", "Afternoon 11-4", "Evening 4-8"])
    weather = st.selectbox("Weather", ["Sunny/Calm", "Windy", "Cloudy", "Raining"])
    strength = st.selectbox("Strength", ["Weak/Nuc", "Medium", "Booming/Double"])
    label = st.selectbox("You saw:", 
                         ["Calm/Foraging", "Agitated/Defensive", "Queenless Roar", 
                          "Queen Piping", "Swarm Prep", "Robbing", "Unknown"])

if audio_bytes:
    with st.spinner("Analyzing..."):
        peak_hz, centroid, rms, wind = analyze_audio(audio_bytes.getvalue())
        ml_label, ml_reason = interpret_sound(peak_hz, centroid, rms, wind)
    
    st.subheader("ML Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Peak Hz", f"{peak_hz} Hz")
    m2.metric("Centroid", f"{centroid} Hz")
    m3.metric("Energy", f"{rms}")
    
    st.success(f"**ML says: {ml_label}**")
    st.caption(ml_reason)
    if wind: st.warning("⚠️ Wind noise detected <80Hz")
    
    if st.button("✅ Save Log", type="primary", use_container_width=True):
        log_row = {"timestamp": datetime.now().isoformat(), "peak_hz": peak_hz, "centroid_hz": centroid, "rms_energy": rms,
                   "temp_c": temp_c, "time": time_of_day, "weather": weather, "strength": strength, 
                   "user_label": label, "ml_label": ml_label, "wind_flag": wind}
        save_log(log_row)
        st.success(f"Saved! Total: {sum(1 for _ in open(DATA_FILE)) - 1}")
        st.rerun()

if st.checkbox("Show logs"):
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), "hive_data.csv")
