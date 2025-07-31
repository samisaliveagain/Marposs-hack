import os
import librosa
import numpy as np

# --- Step 1: Segment Audio ---
def segment_audio(file_path, window_duration=1.0, overlap=0.5):
    """
    Splits audio into overlapping windows using original sampling rate.

    Args:
        file_path (str): Path to the audio file.
        window_duration (float): Duration of each window in seconds.
        overlap (float): Overlap ratio (between 0 and 1).

    Returns:
        Tuple: (list of audio windows, sampling rate)
    """
    y, sr = librosa.load(file_path, sr=None)
    window_size = int(window_duration * sr)
    hop_length = int(window_size * (1 - overlap))

    windows = []
    for start in range(0, len(y) - window_size + 1, hop_length):
        end = start + window_size
        windows.append(y[start:end])

    return windows, sr

# --- Step 2: Feature Extraction ---
def extract_features_from_window(y, sr):
    """
    Extracts a fixed-length feature vector from a single audio window.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.hstack([
        mfccs.mean(axis=1), mfccs.std(axis=1),
        spectral_centroid.mean(), spectral_centroid.std(),
        zcr.mean(), zcr.std(),
        rms.mean(), rms.std()
    ])
    return features

def build_dataset(windows, sr, label):
    """
    Extract features and assign labels to list of windows.
    """
    X, y = [], []
    for win in windows:
        X.append(extract_features_from_window(win, sr))
        y.append(label)
    return np.array(X), np.array(y)

# --- Step 3: Main Processing ---
def main():
    # File paths
    chatter_path = "data/chatter.wav"    # wav file with chatter
    nonchatter_path = "data/non_chatter.wav"  # wav file without chatter

    # Segment audio
    chatter_windows, sr1 = segment_audio(chatter_path)
    nonchatter_windows, sr2 = segment_audio(nonchatter_path)

    assert sr1 == sr2, "Sampling rates must match"
    sr = sr1

    # Extract features & labels
    X_chatter, y_chatter = build_dataset(chatter_windows, sr, label=1)
    X_nonchatter, y_nonchatter = build_dataset(nonchatter_windows, sr, label=0)

    # Combine
    X = np.vstack([X_chatter, X_nonchatter])
    y = np.concatenate([y_chatter, y_nonchatter])

    print(f"Final dataset: X shape = {X.shape}, y shape = {y.shape}")
    
    # Create output directory if it doesn't exist
    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)

    # Save to folder
    output_path = os.path.join(output_dir, "chatter_dataset.npz")
    np.savez(output_path, X=X, y=y)
    print(f"Saved dataset to {output_path}")

if __name__ == "__main__":
    main()
