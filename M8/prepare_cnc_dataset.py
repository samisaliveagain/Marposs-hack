import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
            run_names_set = run_names_set & set(runs) if run_names_set else set(runs)
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
            series = pd.to_numeric(df[run], errors='coerce') if run in df.columns else pd.Series([np.nan])
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
        tensor[n_idx, -1, :] = id_val

    labels = np.full((num_runs,), label, dtype=np.uint8)
    return tensor, mask, labels, feature_list + ["PassID"]

def normalize_tensor(tensor, mask):
    signal_data = tensor[:, :-1, :]
    pass_id_channel = tensor[:, -1:, :]
    N, F, T = signal_data.shape

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
    norm_factors = {f"SIG{i+1}": {"mean": float(means[i]), "std": float(stds[i])} for i in range(F)}
    return norm_tensor, norm_factors

def main():
    pos_tensor, pos_mask, pos_labels, features = load_cnc_folder("IO/TN22", label=1)
    neg_tensor, neg_mask, neg_labels, _ = load_cnc_folder("NIO/TN22", label=0)

    full_tensor = np.concatenate([pos_tensor, neg_tensor], axis=0)
    full_mask = np.concatenate([pos_mask, neg_mask], axis=0)
    y_labels = np.concatenate([pos_labels, neg_labels], axis=0)

    norm_tensor, norm_factors = normalize_tensor(full_tensor, full_mask)

    # Remove PassID before splitting
    X = norm_tensor[:, :-1, :]
    y = y_labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    os.makedirs("split_cnc_data", exist_ok=True)
    np.save("split_cnc_data/X_train.npy", X_train)
    np.save("split_cnc_data/X_test.npy", X_test)
    np.save("split_cnc_data/y_train.npy", y_train)
    np.save("split_cnc_data/y_test.npy", y_test)

    with open("split_cnc_data/norm_factors.json", "w") as f:
        json.dump(norm_factors, f, indent=2)

    print("Dataset prepared and saved to split_cnc_data/")
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

if __name__ == "__main__":
    main()
