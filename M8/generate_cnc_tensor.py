import os
import re
import numpy as np
import pandas as pd

def extract_id_from_filename(filename):
    match = re.search(r'_ID(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_cnc_folder(folder_path, label):
    signal_data_dict = {}
    file_id_map = {}
    run_names_set = None

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath, sep=';', engine='python')
            signal_name = filename.split('_')[2]
            file_id = extract_id_from_filename(filename)
            file_id_map[signal_name] = file_id

            runs = df.columns[1:]
            if run_names_set is None:
                run_names_set = set(runs)
            else:
                run_names_set = run_names_set & set(runs)
            signal_data_dict[signal_name] = df

    if not run_names_set:
        raise ValueError("No common runs found across signal files.")

    common_runs = sorted(run_names_set)
    num_runs = len(common_runs)
    num_features = len(signal_data_dict)
    max_time = 0

    feature_run_data = {}
    id_per_run = []

    for feature, df in signal_data_dict.items():
        run_matrices = []
        file_id = file_id_map[feature]
        for run in common_runs:
            if run not in df.columns:
                run_array = np.array([np.nan])
            else:
                series = pd.to_numeric(df[run], errors='coerce')
                run_array = series.to_numpy()
            run_matrices.append(run_array)
            if feature == sorted(signal_data_dict.keys())[0]:
                id_per_run.append(file_id)
            max_time = max(max_time, np.count_nonzero(~np.isnan(run_array)))
        feature_run_data[feature] = run_matrices

    tensor = np.zeros((num_runs, num_features + 1, max_time), dtype=np.float32)
    mask = np.zeros((num_runs, max_time), dtype=np.uint8)

    feature_list = sorted(feature_run_data.keys())
    for f_idx, feature in enumerate(feature_list):
        for n_idx, run_data in enumerate(feature_run_data[feature]):
            valid_len = np.count_nonzero(~np.isnan(run_data))
            tensor[n_idx, f_idx, :valid_len] = run_data[:valid_len]
            if f_idx == 0:
                mask[n_idx, :valid_len] = 1

    for n_idx, id_val in enumerate(id_per_run):
        tensor[n_idx, -1, :] = id_val  # PassID feature

    labels = np.full((num_runs,), label, dtype=np.uint8)
    return tensor, mask, labels, feature_list + ["PassID"]

def normalize_combined(tensor, mask):
    signal_data = tensor[:, :-1, :]  # exclude PassID
    pass_id_channel = tensor[:, -1:, :]
    N, F, T = signal_data.shape

    # Flatten and mask
    flattened = signal_data.transpose(1, 0, 2).reshape(F, -1)
    flat_mask = mask.reshape(1, -1).repeat(F, axis=0)
    flattened_masked = np.where(flat_mask == 1, flattened, np.nan)

    means = np.nanmean(flattened_masked, axis=1)
    stds = np.nanstd(flattened_masked, axis=1)
    stds[stds == 0] = 1e-6

    norm_data = np.zeros_like(signal_data)
    for f in range(F):
        norm_data[:, f, :] = (signal_data[:, f, :] - means[f]) / stds[f]

    norm_tensor = np.concatenate([norm_data, pass_id_channel], axis=1)
    norm_factors = [(float(means[i]), float(stds[i])) for i in range(F)]
    return norm_tensor, norm_factors

# MAIN
if __name__ == "__main__":
    pos_path = "IO/TN22"
    neg_path = "NIO/TN22"

    print("Loading positive samples...")
    pos_tensor, pos_mask, pos_labels, features = load_cnc_folder(pos_path, label=1)
    print("  Loaded:", pos_tensor.shape)

    print("Loading negative samples...")
    neg_tensor, neg_mask, neg_labels, _ = load_cnc_folder(neg_path, label=0)
    print("  Loaded:", neg_tensor.shape)

    # Merge
    full_tensor = np.concatenate([pos_tensor, neg_tensor], axis=0)
    full_mask = np.concatenate([pos_mask, neg_mask], axis=0)
    y_labels = np.concatenate([pos_labels, neg_labels], axis=0)

    print("Normalizing...")
    norm_tensor, norm_factors = normalize_combined(full_tensor, full_mask)
    print("Final tensor shape (N, F+1, T):", norm_tensor.shape)
    print("Labels shape:", y_labels.shape)

    print("\nNormalization factors:")
    for f, (mu, sigma) in zip(features[:-1], norm_factors):  # exclude PassID
        print(f"  {f}: mean = {mu:.4f}, std = {sigma:.4f}")
