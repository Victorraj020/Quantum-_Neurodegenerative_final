"""
SIH 2026 | Problem Statement 26139
Training & Artifact Generation Script for Early Parkinson's Voice Screening Tool

This script trains:
  1. A Variational Quantum Classifier (VQC) using PennyLane (4 qubits).
  2. A Classical Random Forest Baseline on PCA-reduced voice features.

It exports all required model artifacts to the models/ directory:
  - models/scaler.joblib
  - models/pca.joblib
  - models/angle_scaler.joblib
  - models/rf_model.joblib
  - models/vqc_weights.npy
  - models/vqc_bias.npy
  - models/feature_names.json
  - models/benchmark.json
  - models/sample_patients.json
  - models/pca_loadings.json
"""

import os
import time
import json
import joblib
import numpy as np_plain
import pandas as pd
import pennylane as qml
from pennylane import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------------------------------------------
# 1. RESOLVE DATASET PATH & ENSURE OUTPUT DIRECTORY
# -------------------------------------------------------------------
CWD = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
    os.path.join(CWD, "data", "raw", "parkinsons_voice.csv"),
    os.path.join(CWD, "parkinsons_voice.csv"),
    os.path.join("data", "raw", "parkinsons_voice.csv"),
    "parkinsons_voice.csv"
]

CSV_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        CSV_PATH = p
        break

if CSV_PATH is None:
    raise FileNotFoundError(
        "Dataset not found! Checked 'data/raw/parkinsons_voice.csv' and 'parkinsons_voice.csv'."
    )

print(f"[+] Using Dataset File: {CSV_PATH}")

MODELS_DIR = os.path.join(CWD, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 2. LOAD DATASET & PREPROCESS
# -------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"[+] Loaded Dataset: {df.shape[0]} recordings, {df.shape[1]} columns.")

y = df["status"].values
X_raw = df.drop(columns=["name", "status"])
feature_names = X_raw.columns.tolist()

# Save feature names
with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names, f, indent=2)

# Step A: Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))

# Step B: PCA (n_components=4 for 4-qubit input)
N_QUBITS = 4
pca = PCA(n_components=N_QUBITS, random_state=42)
X_pca = pca.fit_transform(X_scaled)
joblib.dump(pca, os.path.join(MODELS_DIR, "pca.joblib"))

# Step C: Angle Scaler bounded to [0, pi]
angle_scaler = MinMaxScaler(feature_range=(0, np_plain.pi))
X_quantum = angle_scaler.fit_transform(X_pca)
joblib.dump(angle_scaler, os.path.join(MODELS_DIR, "angle_scaler.joblib"))

# Step D: Train/Test Split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X_quantum, y, np_plain.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)

print(f"[+] Data Split: Train size = {len(X_train)}, Test size = {len(X_test)}")

# -------------------------------------------------------------------
# 3. TRAIN PENNYLANE VARIATIONAL QUANTUM CLASSIFIER (VQC)
# -------------------------------------------------------------------
print("\n[+] Training Variational Quantum Classifier (VQC)...")
dev = qml.device("default.qubit", wires=N_QUBITS)
N_LAYERS = 3


@qml.qnode(dev)
def quantum_circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x):
    return quantum_circuit(weights, x) + bias


def square_loss(labels, predictions):
    return np.mean((np.array(labels) - np.array(predictions)) ** 2)


def cost_function(weights, bias, X_b, y_b):
    y_mapped = np.array([1.0 if label == 1 else -1.0 for label in y_b])
    preds = [variational_classifier(weights, bias, x) for x in X_b]
    return square_loss(y_mapped, preds)


np_plain.random.seed(42)
weights = np.array(0.01 * np_plain.random.randn(N_LAYERS, N_QUBITS), requires_grad=True)
bias = np.array(0.0, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.10)
batch_size = 32
n_epochs = 35

t0_q = time.time()
for epoch in range(n_epochs):
    batch_idx = np_plain.random.choice(len(X_train), batch_size, replace=False)
    X_batch = [X_train[i] for i in batch_idx]
    y_batch = [y_train[i] for i in batch_idx]

    weights, bias = opt.step(
        lambda w, b: cost_function(w, b, X_batch, y_batch), weights, bias
    )
quantum_train_time = time.time() - t0_q

# Save Quantum Weights & Bias
np_plain.save(os.path.join(MODELS_DIR, "vqc_weights.npy"), np_plain.array(weights))
np_plain.save(os.path.join(MODELS_DIR, "vqc_bias.npy"), np_plain.array(bias))
print(f"[+] Saved VQC weights & bias. Training took {quantum_train_time:.2f}s.")

# Evaluate VQC
raw_q_test_preds = [variational_classifier(weights, bias, x) for x in X_test]
y_pred_q = np_plain.array([1 if p > 0.0 else 0 for p in raw_q_test_preds])

# -------------------------------------------------------------------
# 4. TRAIN CLASSICAL RANDOM FOREST BASELINE
# -------------------------------------------------------------------
print("\n[+] Training Classical Random Forest Baseline...")
t0_c = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
classical_train_time = time.time() - t0_c

joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.joblib"))
y_pred_c = rf.predict(X_test)
print(f"[+] Saved Random Forest model. Training took {classical_train_time:.4f}s.")

# -------------------------------------------------------------------
# 5. BENCHMARK & PCA LOADINGS ARTIFACTS
# -------------------------------------------------------------------
benchmark = {
    "quantum": {
        "model": "Quantum VQC (4 Qubits)",
        "accuracy": float(accuracy_score(y_test, y_pred_q)),
        "precision": float(precision_score(y_test, y_pred_q, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_q, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_q, zero_division=0)),
        "train_time": float(quantum_train_time),
    },
    "classical": {
        "model": "Classical Random Forest",
        "accuracy": float(accuracy_score(y_test, y_pred_c)),
        "precision": float(precision_score(y_test, y_pred_c, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_c, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_c, zero_division=0)),
        "train_time": float(classical_train_time),
    }
}

with open(os.path.join(MODELS_DIR, "benchmark.json"), "w") as f:
    json.dump(benchmark, f, indent=2)

# Save PCA Loadings for Explainability
pca_components = pca.components_
pca_loadings = {}
for i in range(N_QUBITS):
    loadings = pca_components[i]
    top_indices = np_plain.argsort(np_plain.abs(loadings))[::-1][:4]
    pca_loadings[f"PC{i+1}"] = {
        "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
        "top_features": [
            {
                "feature": feature_names[idx],
                "weight": float(loadings[idx])
            }
            for idx in top_indices
        ]
    }

with open(os.path.join(MODELS_DIR, "pca_loadings.json"), "w") as f:
    json.dump(pca_loadings, f, indent=2)

# Save Sample Patients for UI Dropdown
sample_patients = []
for idx in indices_test[:10]:
    row = df.iloc[idx]
    patient_dict = {
        "sample_id": str(row["name"]),
        "true_status": int(row["status"]),
        "status_label": "Elevated Risk Flagged" if row["status"] == 1 else "Low Risk / Unflagged",
        "features": {feat: float(row[feat]) for feat in feature_names}
    }
    sample_patients.append(patient_dict)

with open(os.path.join(MODELS_DIR, "sample_patients.json"), "w") as f:
    json.dump(sample_patients, f, indent=2)

print("\n[!] ALL ARTIFACTS SUCCESSFULLY GENERATED IN models/ DIRECTORY:")
for fname in [
    "scaler.joblib", "pca.joblib", "angle_scaler.joblib", "rf_model.joblib",
    "vqc_weights.npy", "vqc_bias.npy", "feature_names.json", "benchmark.json",
    "sample_patients.json", "pca_loadings.json"
]:
    fpath = os.path.join(MODELS_DIR, fname)
    print(f"   - {fname} ({'EXISTS' if os.path.exists(fpath) else 'MISSING'})")
