import sounddevice as sd
import numpy as np
import joblib
import librosa
import time
from segment_audio import extract_features_from_window

# Load model and scaler
clf = joblib.load("chatter_rf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Parameters
duration = 1.0  # seconds
sr = 44100

def predict_from_audio(audio):
    feat = extract_features_from_window(audio, sr).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred = clf.predict(feat_scaled)[0]
    return pred

def print_status(pred):
    if pred == 1:
        print("\033[91m🔴 CHATTER DETECTED\033[0m")
    else:
        print("\033[92m🟢 No Chatter\033[0m")

print("🔊 Starting real-time chatter detection (Press Ctrl+C to stop)...")

try:
    while True:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()  # Wait for recording to finish
        audio = audio.flatten()
        pred = predict_from_audio(audio)
        print_status(pred)
        time.sleep(0.1)  # small delay
except KeyboardInterrupt:
    print("\n Stopped.")
