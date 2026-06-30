# 🐝 BeeHive Listener v2 - Learning Edition

A phone app that listens to your hive and learns if it’s Calm, Agitated, Queen Piping, or Low Rumble. It gets smarter every time you log what you actually saw.

Built with Python + Streamlit. Runs on iPhone via Safari.

### How it works
1. **Record**: 5s audio near the hive entrance
2. **Analyze**: App shows Peak Hz + ML guess
3. **Feedback**: Tell it what you saw when you opened the hive
4. **Learn**: After 5-10 logs per state, it switches from rules to ML

Features used: Peak Frequency, Spectral Centroid, Bandwidth, Total Energy

### Run it yourself on Replit [Free]
1. Click `Import from GitHub` on Replit
2. Paste this repo URL: `https://github.com/jbroadway14/beehive-listener.git`
3. Hit `Run`
4. Open the URL in `Safari on iPhone` and allow microphone access

### Requirements
