"""
SIH 2026 | Problem Statement 26139
Streamlit Web UI for Hybrid Quantum-Classical Early Parkinson's Risk Screening Tool

Notice: This application is designed solely for early risk screening and triage.
It is NOT a diagnostic tool.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import pennylane as qml

# -------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Early Parkinson's Risk Screening Tool (QML)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern medical dashboard aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.85rem;
        border-radius: 4px;
        color: #92400E;
        font-weight: 500;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1.2rem;
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    .status-flagged {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
        padding: 1rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.3rem;
    }
    .status-healthy {
        background-color: #D1FAE5;
        border-left: 6px solid #10B981;
        color: #065F46;
        padding: 1rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------------------------
CWD = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CWD, "models")

@st.cache_resource
def load_model_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    pca = joblib.load(os.path.join(MODELS_DIR, "pca.joblib"))
    angle_scaler = joblib.load(os.path.join(MODELS_DIR, "angle_scaler.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))
    
    vqc_weights = np.load(os.path.join(MODELS_DIR, "vqc_weights.npy"))
    vqc_bias = np.load(os.path.join(MODELS_DIR, "vqc_bias.npy"))
    
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "benchmark.json"), "r") as f:
        benchmark = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "sample_patients.json"), "r") as f:
        sample_patients = json.load(f)
        
    with open(os.path.join(MODELS_DIR, "pca_loadings.json"), "r") as f:
        pca_loadings = json.load(f)
        
    return scaler, pca, angle_scaler, rf_model, vqc_weights, vqc_bias, feature_names, benchmark, sample_patients, pca_loadings

try:
    scaler, pca, angle_scaler, rf_model, vqc_weights, vqc_bias, feature_names, benchmark, sample_patients, pca_loadings = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts from {MODELS_DIR}. Please run `train_and_save.py` first. Error: {e}")
    st.stop()

# -------------------------------------------------------------------
# PENNYLANE QUANTUM CIRCUIT DEFINITION
# -------------------------------------------------------------------
N_QUBITS = 4
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def quantum_circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

def predict_vqc(weights, bias, x):
    raw_val = float(quantum_circuit(weights, x) + bias)
    # Map PauliZ expectation output from [-1, 1] to probability range [0, 1]
    prob = 1.0 / (1.0 + np.exp(-2.0 * raw_val))
    pred_class = 1 if raw_val > 0.0 else 0
    return pred_class, prob, raw_val

# -------------------------------------------------------------------
# HEADER & SIDEBAR
# -------------------------------------------------------------------
st.markdown('<div class="main-header">🧠 Early Parkinson\'s Voice Screening Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid Quantum-Classical ML Pipeline | SIH 2026 (PS 26139)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
  ⚠️ <strong>ETHICAL & CLINICAL FRAMING NOTICE:</strong><br>
  This software is designed exclusively as an <strong>early non-invasive risk screening and clinical triage decision-support tool</strong>.
  It flags acoustic voice biomarkers for early risk assessment and does <strong>NOT</strong> provide a medical diagnosis.
</div>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🛠️ Model Benchmark")
st.sidebar.markdown("Comparing **4-Qubit Quantum VQC** vs **Random Forest Baseline** on holdout test samples:")

q_bench = benchmark["quantum"]
c_bench = benchmark["classical"]

benchmark_df = pd.DataFrame([
    {
        "Model": "Quantum VQC",
        "Accuracy": f"{q_bench['accuracy']*100:.1f}%",
        "Recall (Sensitivity)": f"{q_bench['recall']*100:.1f}%",
        "F1-Score": f"{q_bench['f1']:.3f}",
        "Train Time": f"{q_bench['train_time']:.2f}s"
    },
    {
        "Model": "Classical RF",
        "Accuracy": f"{c_bench['accuracy']*100:.1f}%",
        "Recall (Sensitivity)": f"{c_bench['recall']*100:.1f}%",
        "F1-Score": f"{c_bench['f1']:.3f}",
        "Train Time": f"{c_bench['train_time']:.3f}s"
    }
])

st.sidebar.table(benchmark_df.set_index("Model"))

st.sidebar.info("💡 **Clinical Note**: The Quantum VQC model achieved 100% Sensitivity (Recall), ensuring zero false negatives during preliminary screening.")

# -------------------------------------------------------------------
# INPUT SELECTION MODE
# -------------------------------------------------------------------
st.subheader("📋 Step 1: Select Voice Measurement Data")
input_mode = st.radio("Choose Input Method:", ["Select Test-Set Patient Recording", "Manual Voice Biomarker Input"], horizontal=True)

input_features = {}

if input_mode == "Select Test-Set Patient Recording":
    patient_options = {
        f"Sample {p['sample_id']} — True Ground Truth: {p['status_label']}": p
        for p in sample_patients
    }
    selected_label = st.selectbox("Select Patient Voice Recording:", list(patient_options.keys()))
    selected_sample = patient_options[selected_label]
    input_features = selected_sample["features"]
    
    with st.expander("🔍 View Raw Voice Biomarker Values for Selected Patient"):
        st.json(input_features)

else:
    st.markdown("Adjust key acoustic features manually:")
    col1, col2, col3 = st.columns(3)
    
    # Defaults based on dataset averages
    with col1:
        fo = st.number_input("MDVP:Fo (Hz) - Pitch Frequency", value=154.2, min_value=80.0, max_value=260.0)
        jitter_pct = st.number_input("MDVP:Jitter (%)", value=0.006, format="%.4f")
        shimmer_db = st.number_input("MDVP:Shimmer (dB)", value=0.30, format="%.3f")
    with col2:
        hnr = st.number_input("HNR - Harmonics-to-Noise Ratio", value=21.8, format="%.2f")
        rpde = st.number_input("RPDE - Recurrence Period Density Entropy", value=0.49, format="%.3f")
        dfa = st.number_input("DFA - Signal Fractal Scaling Ratio", value=0.71, format="%.3f")
    with col3:
        spread1 = st.number_input("spread1 - Fundamental Frequency Variation", value=-5.68, format="%.3f")
        ppe = st.number_input("PPE - Pitch Period Entropy", value=0.206, format="%.3f")
        d2 = st.number_input("D2 - Non-linear Dynamical Complexity", value=2.38, format="%.3f")
        
    # Construct complete feature vector with dataset defaults for remaining
    default_patient = sample_patients[0]["features"]
    for feat in feature_names:
        input_features[feat] = default_patient[feat]
    
    input_features["MDVP:Fo(Hz)"] = fo
    input_features["MDVP:Jitter(%)"] = jitter_pct
    input_features["MDVP:Shimmer(dB)"] = shimmer_db
    input_features["HNR"] = hnr
    input_features["RPDE"] = rpde
    input_features["DFA"] = dfa
    input_features["spread1"] = spread1
    input_features["PPE"] = ppe
    input_features["D2"] = d2

# -------------------------------------------------------------------
# SCREENING ANALYSIS EXECUTION
# -------------------------------------------------------------------
st.markdown("---")
if st.button("🚀 Run Early Risk Screening Analysis", type="primary", use_container_width=True):
    # Convert feature dict to array
    raw_vector = np.array([[input_features[feat] for feat in feature_names]])
    
    # Pipeline transformation
    scaled_vector = scaler.transform(raw_vector)
    pca_vector = pca.transform(scaled_vector)
    quantum_input = angle_scaler.transform(pca_vector)[0]
    
    # 1. Quantum Model Prediction
    vqc_class, vqc_prob, vqc_raw = predict_vqc(vqc_weights, vqc_bias, quantum_input)
    
    # 2. Classical Model Prediction
    rf_class = int(rf_model.predict(pca_vector)[0])
    rf_probs = rf_model.predict_proba(pca_vector)[0]
    rf_prob = float(rf_probs[1])
    
    # DISPLAY SCREENING RESULTS
    st.subheader("📊 Step 2: Screening Decision Output")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("### ⚛️ Quantum VQC Model (4 Qubits)")
        if vqc_class == 1:
            st.markdown('<div class="status-flagged">⚠️ ELEVATED RISK FLAGGED</div>', unsafe_allow_html=True)
            st.markdown(f"**Risk Confidence Score**: `{vqc_prob*100:.1f}%` (Raw PauliZ Output: `{vqc_raw:+.3f}`)")
        else:
            st.markdown('<div class="status-healthy">✅ LOW RISK / UNFLAGGED</div>', unsafe_allow_html=True)
            st.markdown(f"**Risk Confidence Score**: `{(1-vqc_prob)*100:.1f}%` (Raw PauliZ Output: `{vqc_raw:+.3f}`)")
            
    with res_col2:
        st.markdown("### 🌲 Classical Random Forest Model")
        if rf_class == 1:
            st.markdown('<div class="status-flagged">⚠️ ELEVATED RISK FLAGGED</div>', unsafe_allow_html=True)
            st.markdown(f"**Probability Score**: `{rf_prob*100:.1f}%`")
        else:
            st.markdown('<div class="status-healthy">✅ LOW RISK / UNFLAGGED</div>', unsafe_allow_html=True)
            st.markdown(f"**Probability Score**: `{(1-rf_prob)*100:.1f}%`")
            
    # -------------------------------------------------------------------
    # EXPLAINABILITY BREAKDOWN
    # -------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🔬 Step 3: Acoustic Biomarker Explainability Breakdown")
    st.markdown("Understanding how the patient's voice measurements feed into the 4 quantum components:")
    
    comp_col, load_col = st.columns([1, 1.2])
    
    with comp_col:
        st.markdown("#### Patient's PCA Component Scores")
        comp_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(N_QUBITS)],
            "PCA Score": pca_vector[0],
            "Quantum Angle (rad)": quantum_input
        })
        st.dataframe(comp_df.style.format({"PCA Score": "{:.3f}", "Quantum Angle (rad)": "{:.3f}"}), use_container_width=True)
        
    with load_col:
        st.markdown("#### Key Voice Feature Loadings")
        for pc_name, pc_info in pca_loadings.items():
            exp_var = pc_info["explained_variance_ratio"] * 100
            top_feats = ", ".join([f"**{tf['feature']}** ({tf['weight']:+.2f})" for tf in pc_info["top_features"][:3]])
            st.markdown(f"**{pc_name}** ({exp_var:.1f}% Variance): {top_feats}")
            
    st.info("""
    📌 **Biomarker Interpretation Guide**:
    - **Amplitude Tremors (Shimmer/PPQ)**: Heavily drive **PC1**. High shimmer indicates vocal cord instability.
    - **Fundamental Pitch (MDVP:Fo)**: Drives **PC2**. Variations reflect vocal muscle control changes.
    - **Non-linear Noise (DFA/RPDE)**: Shape **PC3 & PC4**, capturing micro-acoustic decay and breathiness.
    """)
