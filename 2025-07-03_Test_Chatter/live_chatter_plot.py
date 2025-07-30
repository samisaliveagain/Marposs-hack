import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib
from segment_audio import extract_features_from_window

# Load model and scaler
clf = joblib.load("chatter_rf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Parameters
sr = 44100
window_duration = 1.0  # in seconds
window_size = int(window_duration * sr)

# Set up live plot
fig, ax = plt.subplots(figsize=(10, 4))
x = np.linspace(0, window_duration, window_size)
y = np.zeros(window_size)
line, = ax.plot(x, y, lw=1)
ax.set_ylim(-1, 1)
ax.set_xlim(0, window_duration)
ax.set_title("Live Waveform with Chatter Detection")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")

# Status box
status_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=14,
                      bbox=dict(facecolor='white', alpha=0.8))

def predict_chatter(audio):
    feat = extract_features_from_window(audio, sr).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred = clf.predict(feat_scaled)[0]
    return pred

def update(frame):
    audio = sd.rec(window_size, samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = audio.flatten()

    pred = predict_chatter(audio)

    # Update waveform
    line.set_ydata(audio)
    line.set_color('red' if pred == 1 else 'green')
    status = "🔴 CHATTER DETECTED" if pred == 1 else "🟢 No Chatter"
    status_text.set_text(status)
    status_text.set_color('red' if pred == 1 else 'green')

    return line, status_text

ani = animation.FuncAnimation(fig, update, interval=int(window_duration * 1000))
plt.tight_layout()
plt.show()
