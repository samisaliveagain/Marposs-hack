import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
from segment_audio import segment_audio, extract_features_from_window
import os

def predict_windows(file_path):
    # Load model and scaler from the processed directory
    clf = joblib.load(os.path.join("models", "chatter_rf_model.joblib"))
    scaler = joblib.load(os.path.join("models", "scaler.joblib"))

    # Segment the input audio file
    windows, sr = segment_audio(file_path)
    y_full, _ = librosa.load(file_path, sr=sr)  # Reload full signal for plotting

    # Predict chatter for each window
    predictions = []
    for i, w in enumerate(windows):
        feat = extract_features_from_window(w, sr).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        pred = clf.predict(feat_scaled)[0]
        predictions.append(pred)
        print(f"Window {i}: {'Chatter' if pred == 1 else 'Non-Chatter'}")

    return y_full, sr, predictions, len(windows)

def plot_prediction_overlay(y_full, sr, predictions, window_duration=1.0, overlap=0.5):
    # Plot waveform colored by predicted chatter vs non-chatter
    hop_duration = window_duration * (1 - overlap)
    samples_per_window = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)

    plt.figure(figsize=(12, 4))
    for i, label in enumerate(predictions):
        start = i * hop_samples
        end = start + samples_per_window
        if end > len(y_full):
            break
        t = np.linspace(start / sr, end / sr, end - start)
        color = "red" if label == 1 else "green"
        plt.plot(t, y_full[start:end], color=color, linewidth=0.8)

    plt.title("Waveform Colored by Chatter Prediction")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def main():
    file_path = "data/10k_450.wav"  # Replace with your test file path
    y_full, sr, predictions, num_windows = predict_windows(file_path)
    plot_prediction_overlay(y_full, sr, predictions)

if __name__ == "__main__":
    main()
