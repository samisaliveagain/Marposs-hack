import os
import re
import json
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Extract ID from filename using regex pattern "_ID###"
def extract_id_from_filename(filename):
    match = re.search(r'_ID(\d+)', filename)
    return int(match.group(1)) if match else -1

# Load CNC signal data from folder, return tensor, mask, labels, and feature names
def load_cnc_folder(folder_path, label):
    all_tensors = []
    all_masks = []
    all_labels = []
    feature_names = []

    file_paths = glob(os.path.join(folder_path, "*.csv"))
    if not file_paths:
        raise ValueError("No CSV files found.")

    # Group files by their pass ID
    id_groups = defaultdict(list)
    for fp in file_paths:
        id_ = extract_id_from_filename(fp)
        id_groups[id_].append(fp)

    for pass_id, group_files in sorted(id_groups.items()):
        signal_data = {}
        run_names_set = None

        # Load each CSV and build the common run set across all signals
        for path in sorted(group_files):
            signal = os.path.basename(path).split('_')[2]
            df = pd.read_csv(path, sep=';', engine='python')
            if df.shape[1] < 2:
                continue
            run_cols = df.columns[1:]
            run_names_set = run_names_set & set(run_cols) if run_names_set else set(run_cols)
            signal_data[signal] = df

        if not run_names_set:
            continue

        common_runs = sorted(run_names_set)
        num_runs = len(common_runs)
        num_features = len(signal_data)
        max_time = 0
        feature_run_data = {}

        # Organize run data per signal
        for sig, df in signal_data.items():
            run_matrices = []
            for run in common_runs:
                series = pd.to_numeric(df[run], errors='coerce')
                run_arr = series.to_numpy()
                run_matrices.append(run_arr)
                max_time = max(max_time, np.count_nonzero(~np.isnan(run_arr)))
            feature_run_data[sig] = run_matrices

        # Create tensor (runs × features+1 × time)
        tensor = np.zeros((num_runs, num_features + 1, max_time), dtype=np.float32)
        mask = np.zeros((num_runs, max_time), dtype=np.uint8)

        sorted_feats = sorted(feature_run_data.keys())
        if not feature_names:
            feature_names = sorted_feats

        for f_idx, feat in enumerate(sorted_feats):
            for r_idx, run_arr in enumerate(feature_run_data[feat]):
                valid_len = np.count_nonzero(~np.isnan(run_arr))
                tensor[r_idx, f_idx, :valid_len] = run_arr[:valid_len]
                if f_idx == 0:
                    mask[r_idx, :valid_len] = 1  # Only mark mask on one feature

        # Add PassID as final channel
        tensor[:, -1, :] = pass_id
        label_arr = np.full((num_runs,), label, dtype=np.uint8)

        all_tensors.append(tensor)
        all_masks.append(mask)
        all_labels.append(label_arr)

    # Pad all tensors to same max_time
    max_time = max(t.shape[2] for t in all_tensors)
    padded_tensors = []
    padded_masks = []
    for t, m in zip(all_tensors, all_masks):
        pad_width = max_time - t.shape[2]
        if pad_width > 0:
            t = np.pad(t, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
            m = np.pad(m, ((0, 0), (0, pad_width)), mode='constant')
        padded_tensors.append(t)
        padded_masks.append(m)

    # Combine all data
    full_tensor = np.concatenate(padded_tensors, axis=0)
    full_mask = np.concatenate(padded_masks, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)
    return full_tensor, full_mask, full_labels, feature_names + ["PassID"]

# Normalize tensor using mask-aware mean/std normalization
def normalize_tensor(tensor, mask):
    signal_data = tensor[:, :-1, :]
    pass_id_channel = tensor[:, -1:, :]
    N, F, T = signal_data.shape

    # Flatten and apply mask
    flat = signal_data.transpose(1, 0, 2).reshape(F, -1)
    flat_mask = mask.reshape(1, -1).repeat(F, axis=0)
    flat_masked = np.where(flat_mask == 1, flat, np.nan)

    # Compute per-feature mean and std
    means = np.nanmean(flat_masked, axis=1)
    stds = np.nanstd(flat_masked, axis=1)
    stds[stds == 0] = 1e-6  # Avoid divide-by-zero

    # Normalize data
    norm_data = np.zeros_like(signal_data)
    for f in range(F):
        norm_data[:, f, :] = (signal_data[:, f, :] - means[f]) / stds[f]

    norm_tensor = np.concatenate([norm_data, pass_id_channel], axis=1)
    norm_tensor = np.nan_to_num(norm_tensor, nan=0.0)

    # Store normalization factors
    norm_factors = {f"SIG{i+1}": {"mean": float(means[i]), "std": float(stds[i])} for i in range(F)}

    # Print summary
    print("Normalization factors:")
    for i, (mean, std) in enumerate(zip(means, stds)):
        print(f"  SIG{i+1}: mean = {mean:.4f}, std = {std:.4f}")

    return norm_tensor, norm_factors

# Main data preparation routine
def main():
    tool = "TN23"  # Change tool name as needed
    io_path = f"../../M8/IO/{tool}"
    nio_path = f"../../M8/NIO/{tool}"

    print(f"Preparing dataset for tool: {tool}")
    print(f"  Loading from: {io_path} and {nio_path}")
    
    pos_tensor, pos_mask, pos_labels, features = load_cnc_folder(io_path, label=1)
    neg_tensor, neg_mask, neg_labels, _ = load_cnc_folder(nio_path, label=0)

    # Align lengths (in case positive/negative differ slightly in time steps)
    max_time = max(pos_tensor.shape[2], neg_tensor.shape[2])
    def pad(t, m):
        pad_len = max_time - t.shape[2]
        if pad_len > 0:
            t = np.pad(t, ((0, 0), (0, 0), (0, pad_len)), mode='constant')
            m = np.pad(m, ((0, 0), (0, pad_len)), mode='constant')
        return t, m

    pos_tensor, pos_mask = pad(pos_tensor, pos_mask)
    neg_tensor, neg_mask = pad(neg_tensor, neg_mask)

    # Combine tensors and masks
    full_tensor = np.concatenate([pos_tensor, neg_tensor], axis=0)
    full_mask = np.concatenate([pos_mask, neg_mask], axis=0)
    y_labels = np.concatenate([pos_labels, neg_labels], axis=0)

    # Normalize and check
    norm_tensor, norm_factors = normalize_tensor(full_tensor, full_mask)
    assert not np.isnan(norm_tensor).any(), "NaNs found after normalization!"

    # Final split
    X_train, X_test, y_train, y_test = train_test_split(norm_tensor, y_labels, stratify=y_labels, test_size=0.50, random_state=42)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    assert not np.isnan(X_train).any(), "X_train contains NaNs"

    # Save output files
    out_dir = f"split_cnc_data/{tool}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/X_train.npy", X_train)
    np.save(f"{out_dir}/X_test.npy", X_test)
    np.save(f"{out_dir}/y_train.npy", y_train)
    np.save(f"{out_dir}/y_test.npy", y_test)

    with open(f"{out_dir}/norm_factors.json", "w") as f:
        json.dump(norm_factors, f, indent=2)
    with open(f"{out_dir}/feature_names.json", "w") as f:
        json.dump(features, f, indent=2)

    # Final summary
    print("Dataset preparation complete.")
    print(f"  Train set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Test set : {X_test.shape}, Labels: {y_test.shape}")

if __name__ == "__main__":
    main()
