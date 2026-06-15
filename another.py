"""
# Federated Learning Framework for Privacy-Preserving Healthcare IoT Data Mining

This notebook implements the full experimental workflow described in the research proposal:

- MHEALTH wearable-sensor data loading
- subject-as-client non-IID federated partitioning
- sliding-window segmentation
- local client normalization
- centralized, local-only, and federated CNN-LSTM training
- FedAvg aggregation
- optional differential privacy-style Gaussian noise on model updates
- performance, convergence, communication-cost, and scalability reporting
"""


# ============================================================
# 1. Environment setup
# ============================================================
# In Google Colab, uncomment this line if any dependency is missing:
# %pip -q install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib

import os
import re
import glob
import math
import json
import time
import zipfile
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))


# ============================================================
# 2. Research configuration
# ============================================================
CONFIG = {
    # Set DATA_PATH to a local CSV, a folder containing MHEALTH .log files,
    # or a MHEALTH zip file. If empty, the notebook tries to download MHEALTH.
    "DATA_PATH": "/Users/shahjalalmd/Desktop/Research/dataset/mhealth_raw_data.csv",
    "OUTPUT_DIR": "results_federated_healthcare_iot",
    "WINDOW_SIZE": 128,
    "STEP_SIZE": 64,
    "TEST_SIZE": 0.20,
    "CENTRALIZED_VAL_SIZE": 0.15,
    "MIN_CLIENT_WINDOWS": 5,
    "FED_ROUNDS": 20,
    "LOCAL_EPOCHS": 5,
    "CENTRALIZED_EPOCHS": 12,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 1e-3,
    "DP_NOISE_STD": 0.0,   # e.g. 0.001 or 0.005 for privacy-noise experiments
    "MAX_CLIENTS": None,   # set an integer for quick scalability experiments
    "AGGREGATION_ALGORITHMS": ["FedAvg", "FedProx", "FedAdam", "FedYogi", "FedNova"],
    "SERVER_LEARNING_RATE": 1.0,
    "SERVER_MOMENTUM": 0.9,
    "SERVER_BETA_1": 0.9,
    "SERVER_BETA_2": 0.99,
    "SERVER_TAU": 1e-6,
    "FEDPROX_MU": 0.01,
}

OUTPUT_DIR = Path(CONFIG["OUTPUT_DIR"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Outputs will be saved to:", OUTPUT_DIR.resolve())


# ============================================================
# 3. MHEALTH loading utilities
# ============================================================
MHEALTH_URLS = [
    "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip",
]

MHEALTH_COLUMNS = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ecg_1", "ecg_2",
    "left_ankle_acc_x", "left_ankle_acc_y", "left_ankle_acc_z",
    "left_ankle_gyro_x", "left_ankle_gyro_y", "left_ankle_gyro_z",
    "left_ankle_mag_x", "left_ankle_mag_y", "left_ankle_mag_z",
    "right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z",
    "right_arm_gyro_x", "right_arm_gyro_y", "right_arm_gyro_z",
    "right_arm_mag_x", "right_arm_mag_y", "right_arm_mag_z",
    "label",
]


def download_mhealth(destination: Path) -> Optional[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "mhealth_dataset.zip"
    if zip_path.exists():
        return zip_path

    for url in MHEALTH_URLS:
        try:
            print(f"Trying to download MHEALTH from: {url}")
            urllib.request.urlretrieve(url, zip_path)
            print("Downloaded:", zip_path)
            return zip_path
        except Exception as exc:
            print(f"Download failed for {url}: {exc}")
    return None


def extract_zip(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extracted"
    if marker.exists():
        return destination
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination)
    marker.write_text("ok")
    return destination


def subject_from_filename(path: str) -> str:
    match = re.search(r"subject(\d+)", Path(path).name, re.IGNORECASE)
    return f"subject_{match.group(1)}" if match else Path(path).stem


def load_mhealth_logs(path: Path) -> pd.DataFrame:
    if path.is_file() and path.suffix.lower() == ".zip":
        extract_dir = path.with_suffix("")
        path = extract_zip(path, extract_dir)

    log_files = sorted(glob.glob(str(path / "**" / "*.log"), recursive=True))
    if not log_files:
        raise FileNotFoundError(f"No MHEALTH .log files found under {path}")

    frames = []
    for file in log_files:
        df = pd.read_csv(file, sep=r"\s+", header=None)
        if df.shape[1] != len(MHEALTH_COLUMNS):
            raise ValueError(f"{file} has {df.shape[1]} columns, expected {len(MHEALTH_COLUMNS)}")
        df.columns = MHEALTH_COLUMNS
        df["subject"] = subject_from_filename(file)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = data[data["label"] != 0].reset_index(drop=True)  # remove null/rest class
    return data


def load_csv_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if "subject" not in lower and "client" not in lower:
        df["subject"] = "subject_0"
    if "label" not in lower and "activity" not in lower:
        df = df.rename(columns={df.columns[-1]: "label"})
        lower = {c.lower(): c for c in df.columns}

    label_col = lower.get("label", lower.get("activity"))
    if label_col and 0 in set(pd.Series(df[label_col]).dropna().unique()):
        df = df[df[label_col] != 0].reset_index(drop=True)
    return df


def make_synthetic_mhealth_like_data(num_subjects=8, samples_per_subject=1200, num_features=23, num_classes=6):
    print("WARNING: Real MHEALTH data was not found. Creating synthetic data so the code can be tested end-to-end.")
    rows = []
    for subject_id in range(1, num_subjects + 1):
        subject_shift = np.random.normal(0, 0.7, size=num_features)
        for t in range(samples_per_subject):
            label = 1 + ((t // 160 + subject_id) % num_classes)
            class_signal = np.sin(np.linspace(0, np.pi, num_features) * label)
            noise = np.random.normal(0, 0.35, size=num_features)
            features = class_signal + subject_shift + noise
            rows.append([*features, label, f"subject_{subject_id}"])
    columns = [f"sensor_{i:02d}" for i in range(num_features)] + ["label", "subject"]
    return pd.DataFrame(rows, columns=columns)


def load_dataset(config: dict) -> pd.DataFrame:
    data_path = Path(config["DATA_PATH"]).expanduser() if config["DATA_PATH"] else None

    if data_path and data_path.exists() and data_path.is_file() and data_path.suffix.lower() == ".csv":
        print("Loading CSV dataset:", data_path)
        return load_csv_dataset(data_path)

    if data_path and data_path.exists():
        print("Loading MHEALTH logs from:", data_path)
        return load_mhealth_logs(data_path)

    cache_dir = Path("data_cache")
    zip_path = download_mhealth(cache_dir)
    if zip_path:
        return load_mhealth_logs(zip_path)

    return make_synthetic_mhealth_like_data()


raw_df = load_dataset(CONFIG)
print(raw_df.shape)
raw_df.head()


# ============================================================
# 4. Sliding-window segmentation and client construction
# ============================================================
def infer_columns(df: pd.DataFrame) -> Tuple[str, str, List[str]]:
    lower = {c.lower(): c for c in df.columns}
    label_col = lower.get("label", lower.get("activity", df.columns[-1]))
    subject_col = lower.get("subject", lower.get("client", None))
    if subject_col is None:
        subject_col = "subject"
        df[subject_col] = "subject_0"

    feature_cols = [
        c for c in df.columns
        if c not in {label_col, subject_col} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    return label_col, subject_col, feature_cols


def majority_label(values: np.ndarray):
    labels, counts = np.unique(values, return_counts=True)
    return labels[np.argmax(counts)]


def create_windows(
    df: pd.DataFrame,
    label_col: str,
    subject_col: str,
    feature_cols: List[str],
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_windows, y_windows, subjects = [], [], []

    for subject, group in df.groupby(subject_col, sort=True):
        group = group.reset_index(drop=True)
        X_values = group[feature_cols].astype("float32").values
        y_values = group[label_col].values

        if len(group) < window_size:
            continue

        for start in range(0, len(group) - window_size + 1, step_size):
            end = start + window_size
            X_windows.append(X_values[start:end])
            y_windows.append(majority_label(y_values[start:end]))
            subjects.append(subject)

    return np.array(X_windows, dtype="float32"), np.array(y_windows), np.array(subjects)


label_col, subject_col, feature_cols = infer_columns(raw_df)
print("Label column:", label_col)
print("Subject/client column:", subject_col)
print("Number of sensor features:", len(feature_cols))

X, y_raw, subjects = create_windows(
    raw_df,
    label_col=label_col,
    subject_col=subject_col,
    feature_cols=feature_cols,
    window_size=CONFIG["WINDOW_SIZE"],
    step_size=CONFIG["STEP_SIZE"],
)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print("Window tensor:", X.shape)
print("Encoded labels:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
print("Clients:", pd.Series(subjects).nunique())
pd.DataFrame({"subject": subjects, "label": y}).groupby("subject").size().rename("windows").to_frame().head()


# ============================================================
# 5. Non-IID train/test split by client windows
# ============================================================
def split_by_window(X, y, subjects, test_size=0.2):
    indices = np.arange(len(X))
    stratify = y if len(np.unique(y)) > 1 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=SEED,
        stratify=stratify,
    )
    return train_idx, test_idx


def normalize_per_client(
    X_train,
    X_test,
    train_subjects,
    test_subjects,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, StandardScaler]]:
    X_train_norm = np.empty_like(X_train, dtype="float32")
    X_test_norm = np.empty_like(X_test, dtype="float32")
    scalers = {}

    global_scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))

    for subject in np.unique(train_subjects):
        train_mask = train_subjects == subject
        scaler = StandardScaler().fit(X_train[train_mask].reshape(-1, X_train.shape[-1]))
        scalers[subject] = scaler
        X_train_norm[train_mask] = scaler.transform(
            X_train[train_mask].reshape(-1, X_train.shape[-1])
        ).reshape(X_train[train_mask].shape)

    for i, subject in enumerate(test_subjects):
        scaler = scalers.get(subject, global_scaler)
        X_test_norm[i] = scaler.transform(X_test[i].reshape(-1, X_test.shape[-1])).reshape(X_test[i].shape)

    return X_train_norm, X_test_norm, scalers


train_idx, test_idx = split_by_window(X, y, subjects, CONFIG["TEST_SIZE"])

X_train_raw, y_train, subjects_train = X[train_idx], y[train_idx], subjects[train_idx]
X_test_raw, y_test, subjects_test = X[test_idx], y[test_idx], subjects[test_idx]

X_train, X_test, client_scalers = normalize_per_client(
    X_train_raw,
    X_test_raw,
    subjects_train,
    subjects_test,
)

client_ids = sorted(pd.Series(subjects_train).unique())
if CONFIG["MAX_CLIENTS"] is not None:
    client_ids = client_ids[: int(CONFIG["MAX_CLIENTS"])]

client_data = {}
for client in client_ids:
    mask = subjects_train == client
    if mask.sum() >= CONFIG["MIN_CLIENT_WINDOWS"]:
        client_data[client] = (X_train[mask], y_train[mask])

print("Training windows:", X_train.shape)
print("Test windows:", X_test.shape)
print("Federated clients retained:", len(client_data))
pd.DataFrame({
    "client": list(client_data.keys()),
    "train_windows": [len(v[0]) for v in client_data.values()],
}).head(20)


# ============================================================
# 6. Model builders: CNN-LSTM proposal model and baselines
# ============================================================
num_classes = len(np.unique(y))
input_shape = X_train.shape[1:]


def build_cnn_lstm(input_shape=input_shape, num_classes=num_classes, lr=CONFIG["LEARNING_RATE"]):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(96, dropout=0.20, recurrent_dropout=0.0),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_lstm_edge_model")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(input_shape=input_shape, num_classes=num_classes, lr=CONFIG["LEARNING_RATE"]):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(96),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="lstm_baseline")
    model.compile(optimizer=optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_shape=input_shape, num_classes=num_classes, lr=CONFIG["LEARNING_RATE"]):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="cnn_baseline")
    model.compile(optimizer=optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp(input_shape=input_shape, num_classes=num_classes, lr=CONFIG["LEARNING_RATE"]):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="mlp_baseline")
    model.compile(optimizer=optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


MODEL_BUILDERS = {
    "CNN-LSTM": build_cnn_lstm,
    "CNN": build_cnn,
    "LSTM": build_lstm,
    "MLP": build_mlp,
}

proposal_model = build_cnn_lstm()
proposal_model.summary()


# ============================================================
# 7. Evaluation helpers
# ============================================================
def predict_classes(model, X_eval):
    probs = model.predict(X_eval, batch_size=CONFIG["BATCH_SIZE"], verbose=0)
    return np.argmax(probs, axis=1)


def evaluate_model(model, X_eval, y_eval, model_name: str, extra: Optional[dict] = None):
    y_pred = predict_classes(model, X_eval)
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision_weighted": precision_score(y_eval, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_eval, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_eval, y_pred, average="weighted", zero_division=0),
    }
    if extra:
        metrics.update(extra)

    print(f"\n=== {model_name} ===")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, indent=2))
    print(classification_report(y_eval, y_pred, target_names=[str(c) for c in label_encoder.classes_], zero_division=0))
    return metrics, y_pred


def plot_confusion(y_true, y_pred, title, path):
    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.show()


def model_size_bytes(model) -> int:
    return int(sum(np.asarray(w).nbytes for w in model.get_weights()))


# ============================================================
# 8. Centralized training baseline
# ============================================================
centralized_model = build_cnn_lstm()
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

start = time.time()
centralized_history = centralized_model.fit(
    X_train,
    y_train,
    validation_split=CONFIG["CENTRALIZED_VAL_SIZE"],
    epochs=CONFIG["CENTRALIZED_EPOCHS"],
    batch_size=CONFIG["BATCH_SIZE"],
    callbacks=[early_stop],
    verbose=1,
)
centralized_seconds = time.time() - start

centralized_metrics, centralized_pred = evaluate_model(
    centralized_model,
    X_test,
    y_test,
    "Centralized CNN-LSTM",
    {"training_seconds": centralized_seconds},
)
plot_confusion(
    y_test,
    centralized_pred,
    "Centralized CNN-LSTM Confusion Matrix",
    OUTPUT_DIR / "centralized_confusion_matrix.png",
)


# ============================================================
# 9. Local-only edge baseline
# ============================================================
local_metrics = []
local_predictions = []

for client, (Xc, yc) in client_data.items():
    if len(np.unique(yc)) < 2:
        continue
    model = build_cnn_lstm()
    model.fit(Xc, yc, epochs=CONFIG["LOCAL_EPOCHS"], batch_size=CONFIG["BATCH_SIZE"], verbose=0)
    metrics, _ = evaluate_model(model, X_test, y_test, f"Local-only {client}", {"client": client, "client_windows": len(Xc)})
    local_metrics.append(metrics)

local_df = pd.DataFrame(local_metrics)
local_df.to_csv(OUTPUT_DIR / "local_only_client_metrics.csv", index=False)
local_summary = {
    "model": "Local-only average",
    "accuracy": local_df["accuracy"].mean() if len(local_df) else np.nan,
    "precision_weighted": local_df["precision_weighted"].mean() if len(local_df) else np.nan,
    "recall_weighted": local_df["recall_weighted"].mean() if len(local_df) else np.nan,
    "f1_weighted": local_df["f1_weighted"].mean() if len(local_df) else np.nan,
}
local_summary


# ============================================================
# 10. Federated aggregation with optional privacy noise
# ============================================================
def weighted_average_weights(local_weight_sets, local_sizes):
    total = float(np.sum(local_sizes))
    averaged = []
    for weights in zip(*local_weight_sets):
        averaged.append(np.sum([w * (n / total) for w, n in zip(weights, local_sizes)], axis=0))
    return averaged


def zeros_like_weights(weights):
    return [np.zeros_like(w, dtype=np.float32) for w in weights]


def add_gaussian_noise(weights, std):
    if std <= 0:
        return weights
    return [w + np.random.normal(0.0, std, size=w.shape).astype(w.dtype) for w in weights]


def train_local_model(
    model,
    X_local,
    y_local,
    epochs,
    batch_size,
    aggregation="FedAvg",
    global_trainable_weights=None,
    fedprox_mu=0.0,
):
    if aggregation != "FedProx":
        history = model.fit(X_local, y_local, epochs=epochs, batch_size=batch_size, verbose=0)
        steps = int(math.ceil(len(X_local) / batch_size)) * epochs
        return history, max(steps, 1)

    if global_trainable_weights is None:
        raise ValueError("FedProx requires the current global trainable weights.")

    global_tensors = [
        tf.constant(w, dtype=var.dtype)
        for w, var in zip(global_trainable_weights, model.trainable_variables)
    ]
    dataset = (
        tf.data.Dataset.from_tensor_slices((X_local.astype("float32"), y_local.astype("int64")))
        .shuffle(min(len(X_local), 2048), seed=SEED)
        .batch(batch_size)
    )
    optimizer = optimizers.Adam(learning_rate=CONFIG["LEARNING_RATE"])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean()
    steps = 0

    for _ in range(epochs):
        for xb, yb in dataset:
            with tf.GradientTape() as tape:
                logits = model(xb, training=True)
                task_loss = loss_fn(yb, logits)
                prox_loss = tf.add_n([
                    tf.reduce_sum(tf.square(var - global_var))
                    for var, global_var in zip(model.trainable_variables, global_tensors)
                ])
                loss = task_loss + (fedprox_mu / 2.0) * prox_loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss.update_state(loss)
            steps += 1

    return {"loss": [float(train_loss.result().numpy())]}, max(steps, 1)


def aggregate_server_update(
    global_weights,
    averaged_client_weights,
    optimizer_state,
    algorithm="FedAvg",
    server_lr=1.0,
    beta1=0.9,
    beta2=0.99,
    tau=1e-6,
):
    algorithm = algorithm.strip()
    delta = [client_w - server_w for client_w, server_w in zip(averaged_client_weights, global_weights)]

    if algorithm in {"FedAvg", "FedProx", "FedNova"}:
        return averaged_client_weights, optimizer_state

    if algorithm == "FedAvgM":
        velocity = optimizer_state.get("velocity") or zeros_like_weights(global_weights)
        velocity = [beta1 * v + d for v, d in zip(velocity, delta)]
        new_weights = [w + server_lr * v for w, v in zip(global_weights, velocity)]
        optimizer_state["velocity"] = velocity
        return new_weights, optimizer_state

    if algorithm == "FedAdam":
        m = optimizer_state.get("m") or zeros_like_weights(global_weights)
        v = optimizer_state.get("v") or zeros_like_weights(global_weights)
        m = [beta1 * old_m + (1.0 - beta1) * d for old_m, d in zip(m, delta)]
        v = [beta2 * old_v + (1.0 - beta2) * np.square(d) for old_v, d in zip(v, delta)]
        new_weights = [w + server_lr * mi / (np.sqrt(vi) + tau) for w, mi, vi in zip(global_weights, m, v)]
        optimizer_state["m"] = m
        optimizer_state["v"] = v
        return new_weights, optimizer_state

    if algorithm == "FedYogi":
        m = optimizer_state.get("m") or zeros_like_weights(global_weights)
        v = optimizer_state.get("v") or zeros_like_weights(global_weights)
        m = [beta1 * old_m + (1.0 - beta1) * d for old_m, d in zip(m, delta)]
        v = [
            old_v - (1.0 - beta2) * np.square(d) * np.sign(old_v - np.square(d))
            for old_v, d in zip(v, delta)
        ]
        new_weights = [w + server_lr * mi / (np.sqrt(np.maximum(vi, 0.0)) + tau) for w, mi, vi in zip(global_weights, m, v)]
        optimizer_state["m"] = m
        optimizer_state["v"] = v
        return new_weights, optimizer_state

    raise ValueError(f"Unknown aggregation algorithm: {algorithm}")


def fednova_aggregate_weights(global_weights, local_weight_sets, local_sizes, local_steps):
    total = float(np.sum(local_sizes))
    client_weights = [size / total for size in local_sizes]
    tau_eff = float(np.sum([p * steps for p, steps in zip(client_weights, local_steps)]))
    new_weights = []

    for layer_idx, global_layer in enumerate(global_weights):
        normalized_update = np.zeros_like(global_layer, dtype=np.float32)
        for p, steps, local_weights in zip(client_weights, local_steps, local_weight_sets):
            normalized_update += p * (local_weights[layer_idx] - global_layer) / max(float(steps), 1.0)
        new_weights.append(global_layer + tau_eff * normalized_update)

    return new_weights


def federated_train(
    model_builder: Callable[[], tf.keras.Model],
    client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    X_eval,
    y_eval,
    rounds: int,
    local_epochs: int,
    noise_std: float = 0.0,
    aggregation: str = "FedAvg",
    server_lr: float = CONFIG["SERVER_LEARNING_RATE"],
    beta1: float = CONFIG["SERVER_BETA_1"],
    beta2: float = CONFIG["SERVER_BETA_2"],
    tau: float = CONFIG["SERVER_TAU"],
):
    global_model = model_builder()
    global_weights = global_model.get_weights()
    optimizer_state = {}
    history = []
    total_comm_bytes = 0

    for rnd in range(1, rounds + 1):
        round_start = time.time()
        local_weight_sets = []
        local_sizes = []
        local_steps = []

        for client, (Xc, yc) in client_data.items():
            local_model = model_builder()
            local_model.set_weights(global_weights)
            global_trainable_weights = [var.numpy() for var in local_model.trainable_variables]
            _, steps = train_local_model(
                local_model,
                Xc,
                yc,
                epochs=local_epochs,
                batch_size=CONFIG["BATCH_SIZE"],
                aggregation=aggregation,
                global_trainable_weights=global_trainable_weights,
                fedprox_mu=CONFIG["FEDPROX_MU"],
            )

            client_weights = add_gaussian_noise(local_model.get_weights(), noise_std)
            local_weight_sets.append(client_weights)
            local_sizes.append(len(Xc))
            local_steps.append(steps)

            # One download of global weights + one upload of client weights.
            total_comm_bytes += model_size_bytes(local_model) * 2

        if aggregation == "FedNova":
            global_weights = fednova_aggregate_weights(global_weights, local_weight_sets, local_sizes, local_steps)
        else:
            averaged_client_weights = weighted_average_weights(local_weight_sets, local_sizes)
            global_weights, optimizer_state = aggregate_server_update(
                global_weights,
                averaged_client_weights,
                optimizer_state,
                algorithm=aggregation,
                server_lr=server_lr,
                beta1=beta1,
                beta2=beta2,
                tau=tau,
            )
        global_model.set_weights(global_weights)

        y_pred = predict_classes(global_model, X_eval)
        round_metrics = {
            "aggregation": aggregation,
            "round": rnd,
            "accuracy": accuracy_score(y_eval, y_pred),
            "precision_weighted": precision_score(y_eval, y_pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_eval, y_pred, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_eval, y_pred, average="weighted", zero_division=0),
            "round_seconds": time.time() - round_start,
            "cumulative_comm_mb": total_comm_bytes / (1024 ** 2),
            "clients": len(client_data),
            "noise_std": noise_std,
            "mean_local_steps": float(np.mean(local_steps)),
        }
        history.append(round_metrics)
        print(
            f"{aggregation} round {rnd:02d} | acc={round_metrics['accuracy']:.4f} "
            f"f1={round_metrics['f1_weighted']:.4f} "
            f"comm={round_metrics['cumulative_comm_mb']:.2f} MB"
        )

    return global_model, pd.DataFrame(history)


fed_model, fed_history = federated_train(
    build_cnn_lstm,
    client_data,
    X_test,
    y_test,
    rounds=CONFIG["FED_ROUNDS"],
    local_epochs=CONFIG["LOCAL_EPOCHS"],
    noise_std=CONFIG["DP_NOISE_STD"],
    aggregation="FedAvg",
)
fed_history.to_csv(OUTPUT_DIR / "federated_training_history.csv", index=False)

fed_metrics, fed_pred = evaluate_model(
    fed_model,
    X_test,
    y_test,
    "Federated CNN-LSTM FedAvg",
    {
        "fed_rounds": CONFIG["FED_ROUNDS"],
        "local_epochs": CONFIG["LOCAL_EPOCHS"],
        "communication_mb": float(fed_history["cumulative_comm_mb"].iloc[-1]),
        "dp_noise_std": CONFIG["DP_NOISE_STD"],
        "aggregation": "FedAvg",
    },
)
plot_confusion(y_test, fed_pred, "Federated CNN-LSTM Confusion Matrix", OUTPUT_DIR / "federated_confusion_matrix.png")


# ============================================================
# 11. Aggregation algorithm comparison: FedAvg vs FedProx vs FedAdam vs FedYogi vs FedNova
# ============================================================
aggregation_runs = {
    "FedAvg": {
        "model": fed_model,
        "history": fed_history,
        "metrics": fed_metrics,
    }
}

aggregation_comparison_rows = []
fed_metrics_row = dict(fed_metrics)
fed_metrics_row["aggregation"] = "FedAvg"
aggregation_comparison_rows.append(fed_metrics_row)

for aggregation_name in CONFIG["AGGREGATION_ALGORITHMS"]:
    if aggregation_name == "FedAvg":
        continue

    print(f"\n===== Training aggregation algorithm: {aggregation_name} =====")
    model, hist = federated_train(
        build_cnn_lstm,
        client_data,
        X_test,
        y_test,
        rounds=CONFIG["FED_ROUNDS"],
        local_epochs=CONFIG["LOCAL_EPOCHS"],
        noise_std=CONFIG["DP_NOISE_STD"],
        aggregation=aggregation_name,
        server_lr=CONFIG["SERVER_LEARNING_RATE"],
        beta1=CONFIG["SERVER_BETA_1"],
        beta2=CONFIG["SERVER_BETA_2"],
        tau=CONFIG["SERVER_TAU"],
    )
    metrics, pred = evaluate_model(
        model,
        X_test,
        y_test,
        f"Federated CNN-LSTM {aggregation_name}",
        {
            "aggregation": aggregation_name,
            "fed_rounds": CONFIG["FED_ROUNDS"],
            "local_epochs": CONFIG["LOCAL_EPOCHS"],
            "communication_mb": float(hist["cumulative_comm_mb"].iloc[-1]),
            "dp_noise_std": CONFIG["DP_NOISE_STD"],
        },
    )
    plot_confusion(
        y_test,
        pred,
        f"{aggregation_name} CNN-LSTM Confusion Matrix",
        OUTPUT_DIR / f"{aggregation_name.lower()}_confusion_matrix.png",
    )
    aggregation_runs[aggregation_name] = {"model": model, "history": hist, "metrics": metrics}
    aggregation_comparison_rows.append(metrics)

aggregation_comparison_df = (
    pd.DataFrame(aggregation_comparison_rows)
    .sort_values(["f1_weighted", "accuracy"], ascending=False)
    .reset_index(drop=True)
)
aggregation_history_df = pd.concat(
    [run["history"] for run in aggregation_runs.values()],
    ignore_index=True,
)

aggregation_comparison_df.to_csv(OUTPUT_DIR / "aggregation_algorithm_comparison.csv", index=False)
aggregation_history_df.to_csv(OUTPUT_DIR / "aggregation_algorithm_history.csv", index=False)

best_aggregation_name = aggregation_comparison_df.iloc[0]["aggregation"]
best_aggregation_model = aggregation_runs[best_aggregation_name]["model"]
best_aggregation_history = aggregation_runs[best_aggregation_name]["history"]

print(f"Best aggregation algorithm: {best_aggregation_name}")
aggregation_comparison_df


# ============================================================
# 12. Optional model-family comparison
# ============================================================
comparison_rows = []

for name, builder in MODEL_BUILDERS.items():
    print(f"\nTraining federated model family: {name}")
    model, hist = federated_train(
        builder,
        client_data,
        X_test,
        y_test,
        rounds=max(2, min(4, CONFIG["FED_ROUNDS"])),
        local_epochs=CONFIG["LOCAL_EPOCHS"],
        noise_std=0.0,
        aggregation=best_aggregation_name if "best_aggregation_name" in globals() else "FedAvg",
    )
    metrics, _ = evaluate_model(model, X_test, y_test, f"Federated {name}")
    metrics["aggregation"] = best_aggregation_name if "best_aggregation_name" in globals() else "FedAvg"
    metrics["comparison_rounds"] = int(hist["round"].max())
    metrics["communication_mb"] = float(hist["cumulative_comm_mb"].iloc[-1])
    comparison_rows.append(metrics)

model_comparison_df = pd.DataFrame(comparison_rows).sort_values("f1_weighted", ascending=False)
model_comparison_df.to_csv(OUTPUT_DIR / "model_family_comparison.csv", index=False)
model_comparison_df


# ============================================================
# 13. Communication-cost comparison
# ============================================================
def centralized_data_transfer_bytes(X_data, y_data):
    return int(X_data.nbytes + y_data.nbytes)


communication_df = pd.DataFrame([
    {
        "approach": "Centralized raw-data upload",
        "communication_mb": centralized_data_transfer_bytes(X_train, y_train) / (1024 ** 2),
        "privacy_note": "Raw patient/sensor windows leave edge devices",
    },
    {
        "approach": f"Federated {best_aggregation_name if 'best_aggregation_name' in globals() else 'FedAvg'} parameter exchange",
        "communication_mb": float((best_aggregation_history if "best_aggregation_history" in globals() else fed_history)["cumulative_comm_mb"].iloc[-1]),
        "privacy_note": "Only model parameters are exchanged",
    },
])
communication_df.to_csv(OUTPUT_DIR / "communication_cost_comparison.csv", index=False)
communication_df


# ============================================================
# 14. Visualizations for thesis/proposal reporting
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(fed_history["round"], fed_history["accuracy"], marker="o", label="Accuracy")
plt.plot(fed_history["round"], fed_history["f1_weighted"], marker="s", label="Weighted F1")
plt.xlabel("Federated round")
plt.ylabel("Score")
plt.title("Federated CNN-LSTM convergence")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "federated_convergence.png", dpi=160)
plt.show()

if "aggregation_history_df" in globals():
    plt.figure(figsize=(9, 5))
    for aggregation_name, group in aggregation_history_df.groupby("aggregation"):
        plt.plot(group["round"], group["f1_weighted"], marker="o", label=aggregation_name)
    plt.xlabel("Federated round")
    plt.ylabel("Weighted F1-score")
    plt.title("Aggregation algorithm convergence comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aggregation_convergence_comparison.png", dpi=160)
    plt.show()

if "aggregation_comparison_df" in globals():
    plot_df = aggregation_comparison_df.sort_values("f1_weighted")
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["aggregation"], plot_df["f1_weighted"])
    plt.xlabel("Weighted F1-score")
    plt.title("Final F1-score by aggregation algorithm")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aggregation_f1_comparison.png", dpi=160)
    plt.show()

    plot_df = aggregation_comparison_df.sort_values("accuracy")
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["aggregation"], plot_df["accuracy"])
    plt.xlabel("Accuracy")
    plt.title("Final accuracy by aggregation algorithm")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "aggregation_accuracy_comparison.png", dpi=160)
    plt.show()

if "model_comparison_df" in globals():
    plt.figure(figsize=(8, 5))
    plot_df = model_comparison_df.sort_values("f1_weighted")
    plt.barh(plot_df["model"], plot_df["f1_weighted"])
    plt.xlabel("Weighted F1-score")
    plt.title("Federated model-family comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_family_comparison.png", dpi=160)
    plt.show()

plt.figure(figsize=(7, 4))
plt.bar(communication_df["approach"], communication_df["communication_mb"])
plt.ylabel("Communication (MB)")
plt.title("Communication-cost comparison")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "communication_cost_comparison.png", dpi=160)
plt.show()


# ============================================================
# 15. Save final models, metrics, and experiment metadata
# ============================================================
summary_rows = [
    centralized_metrics,
    local_summary,
    fed_metrics,
]
if "aggregation_comparison_df" in globals():
    for _, row in aggregation_comparison_df.iterrows():
        summary_rows.append(row.to_dict())

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "final_experiment_summary.csv", index=False)

centralized_model.save(OUTPUT_DIR / "centralized_cnn_lstm.keras")
fed_model.save(OUTPUT_DIR / "federated_cnn_lstm.keras")
if "best_aggregation_model" in globals():
    best_aggregation_model.save(OUTPUT_DIR / "best_aggregation_cnn_lstm.keras")

metadata = {
    "proposal_title": "Federated Learning Framework for Privacy-Preserving Data Mining in Edge-Cloud Environments",
    "dataset": "MHEALTH wearable sensor dataset or compatible healthcare IoT sensor CSV",
    "num_windows": int(len(X)),
    "train_windows": int(len(X_train)),
    "test_windows": int(len(X_test)),
    "num_clients": int(len(client_data)),
    "window_size": int(CONFIG["WINDOW_SIZE"]),
    "step_size": int(CONFIG["STEP_SIZE"]),
    "num_classes": int(num_classes),
    "class_labels": [str(x) for x in label_encoder.classes_],
    "aggregation_algorithms": CONFIG["AGGREGATION_ALGORITHMS"],
    "best_aggregation": best_aggregation_name if "best_aggregation_name" in globals() else "FedAvg",
    "config": CONFIG,
}

with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved outputs:")
for path in sorted(OUTPUT_DIR.glob("*")):
    print("-", path)

summary_df
