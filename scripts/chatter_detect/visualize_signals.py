import librosa
import librosa.display
import matplotlib.pyplot as plt

def plot_waveform(y, sr, title):
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_spectrogram(y, sr, title):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram - {title}")
    plt.tight_layout()
    plt.show()

def plot_mfcc(y, sr, title):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, cmap='viridis')
    plt.colorbar()
    plt.title(f"MFCC - {title}")
    plt.tight_layout()
    plt.show()

def main():
    # Load both signals (full length)
    y_chatter, sr = librosa.load("data/chatter.wav", sr=None)
    y_nonchatter, _ = librosa.load("data/non_chatter.wav", sr=None)

    # Plot chatter signal
    plot_waveform(y_chatter, sr, "Chatter")
    plot_spectrogram(y_chatter, sr, "Chatter")
    plot_mfcc(y_chatter, sr, "Chatter")

    # Plot non-chatter signal
    plot_waveform(y_nonchatter, sr, "Non-Chatter")
    plot_spectrogram(y_nonchatter, sr, "Non-Chatter")
    plot_mfcc(y_nonchatter, sr, "Non-Chatter")

    

if __name__ == "__main__":
    import numpy as np
    main()
