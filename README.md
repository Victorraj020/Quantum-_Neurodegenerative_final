# SIH 2026 — Problem Statement 26139
## Hybrid Quantum-Classical ML Pipeline for Early Parkinson's Voice Screening

> **Disclaimer & Regulatory Framing**:  
> This project is designed exclusively as an **early non-invasive screening and risk-flagging decision-support tool**. It is **not** a diagnostic device and does not provide definitive medical diagnoses.

---

## 📌 Executive Summary

This repository implements a **Hybrid Quantum-Classical Machine Learning (QML) pipeline** to screen for early signs of Parkinson's Disease using acoustic voice biomarkers.

The pipeline combines a **Variational Quantum Classifier (VQC)** built with **PennyLane** and a **Classical Random Forest Baseline** to evaluate how quantum-enhanced feature entanglement performs relative to standard clinical ML baselines.

---

## 📁 Project Directory Structure

```text
Quantum_final/
├── README.md                        # Complete project guide & documentation
├── main.py                          # Main end-to-end Python pipeline script
├── requirements.txt                 # Python dependencies (PennyLane, scikit-learn, etc.)
├── data/
│   ├── raw/
│   │   ├── parkinsons_voice.csv     # Primary dataset (195 voice recordings, 22 features)
│   │   ├── telemonitoring/          # Voice telemonitoring dataset (symptom tracking)
│   │   ├── drawings/                # Spiral & wave drawing image datasets (stretch goals)
│   │   ├── spiral/
│   │   └── wave/
│   └── processed/                   # Stored preprocessed data matrices
├── models/                          # Trained model artifacts & checkpoint saves
├── notebooks/                       # Experimental Jupyter notebooks
└── docs/                            # Technical architecture documentation
```

---

## ⚙️ How the Pipeline Works (`main.py`)

### 1. Data Ingestion & Preprocessing
* **Dataset**: 195 voice recordings with 22 biomedical voice features (MDVP Jitter, Shimmer, Pitch frequencies, HNR, PPE, etc.) and a binary target (`status`: 0 = Low Risk, 1 = Elevated Risk Flagged).
* **Scaling**: Features are normalized using `StandardScaler`.
* **Dimensionality Reduction**: `PCA(n_components=4)` compresses 22 raw voice features down to **4 Principal Components**, retaining **83.87% of total variance** (necessary for low-qubit quantum simulation).
* **Angle Scaling**: Component features are bounded to $[0, \pi]$ using `MinMaxScaler` for rotation angle embedding.

### 2. Quantum Architecture (Variational Quantum Classifier)
* **Simulator**: PennyLane (`default.qubit` simulator, 4 qubits).
* **Embedding**: `AngleEmbedding` mapping the 4 PCA features to qubit rotation angles.
* **Variational Circuit**: `BasicEntanglerLayers` (3 layers of parameterized multi-qubit entanglement).
* **Measurement**: Pauli-Z expectation value on Qubit 0.
* **Optimization**: `AdamOptimizer` (stepsize = 0.10, mini-batch size = 32, 35 epochs).

### 3. Classical Baseline
* A 100-tree `RandomForestClassifier` trained on the exact same 4-component PCA data for benchmarking.

---

## 📊 Benchmark Results

Evaluated on a 20% holdout test set (39 samples):

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | Training Time (s) |
|---|---|---|---|---|---|
| **Quantum VQC (4 Qubits)** | **84.62%** | **82.86%** | **100.00%** | **0.9062** | ~56.99 s |
| **Classical Random Forest** | **89.74%** | **93.10%** | **93.10%** | **0.9310** | ~0.24 s |

> **Key Takeaway**: The **Quantum VQC achieved 100% Sensitivity (Recall)** on the test set. In clinical screening, 100% recall is vital to ensure zero false negatives (no potential risk cases are missed).

---

## 🔍 Biomarker Explainability

The pipeline maps PCA component weights back to original acoustic parameters:
* **PC1 (58.90% Variance)**: Strongest loading from amplitude perturbation parameters (`MDVP:Shimmer(dB)`, `MDVP:Shimmer`, `MDVP:PPQ`).
* **PC2 (11.30% Variance)**: Strongest loading from fundamental pitch frequencies (`MDVP:Fo(Hz)`, `MDVP:Flo(Hz)`).
* **PC3 & PC4**: Controlled by non-linear vocal noise metrics (`DFA`, `RPDE`, `D2`) and absolute micro-jitter (`MDVP:Jitter(Abs)`).

---

## 🛠️ Environment & Installation

### Requirements
* Python 3.9+
* Packages listed in `requirements.txt`:
  ```txt
  pennylane==0.35.1
  pennylane-lightning==0.35.1
  autoray<0.7
  rustworkx==0.14.2
  scikit-learn==1.5.2
  numpy==1.26.4
  pandas==2.2.3
  ```

### Running the Pipeline
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the main pipeline
python main.py
```
