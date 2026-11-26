# 🫀 ECG Heartbeat Compression & Arrhythmia Classification Using 1D-CAE and Attention-Augmented BiLSTM

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg)](https://streamlit.io/)
[![Paper](https://img.shields.io/badge/IEEE%20Access-2025%20Reproduction-00629B.svg)](https://ieeexplore.ieee.org/)

An end-to-end deep learning pipeline for **ECG signal compression** and **automated cardiac arrhythmia classification**, reproducing and extending the two-stage architecture proposed by *Degachi et al. (IEEE Access, 2025)*.

The system compresses raw 187-sample ECG heartbeats into a compact **32-dimensional latent representation** (5.84× compression ratio) using a **1D Convolutional Autoencoder (1D-CAE)**, and classifies the compressed features using a **Stacked Bidirectional LSTM with Multi-Head Self-Attention (BiLSTM-MHA)**—achieving **69.3% model parameter reduction** while maintaining real-time inference speed (**0.075 ms/beat**).

---

## 🏗️ System Architecture (Two-Stage Pipeline)

```text
Raw ECG Beat (187 × 1)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 1: 1D-Convolutional Autoencoder      │
│  • Encoder: Conv1D (64→32→16) + MaxPool     │
│  • Bottleneck: 32-dim Latent Vector (5.84x) │
│  • Decoder: Conv1DTranspose (Reconstruction)│
└─────────────────────────────────────────────┘
        │
        ▼  Compressed Features (32 × 1)
┌─────────────────────────────────────────────┐
│  Stage 2: BiLSTM + Multi-Head Attention     │
│  • Stacked BiLSTM (128 → 64 units)          │
│  • Multi-Head Self-Attention (8 Heads)      │
│  • Residual Connection + LayerNorm          │
│  • Dense Classifier Head (Softmax)          │
└─────────────────────────────────────────────┘
        │
        ▼
Diagnosis: N | SVEB | VEB | F | Q  (or Normal vs. MI)
```

---

## 📊 Experimental Results & Benchmarks

### 1. Stage 1: Signal Compression Performance (MIT-BIH Test Set)
Our trained `1D-CAE` outperformed the original paper's reconstruction benchmarks across all signal fidelity metrics:

| Metric | Our Result | Paper Benchmark (Table 4) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Compression Ratio (CR)** | **5.84** | 5.84 | 187 samples compressed to 32 features |
| **RMSE** | **0.0221** | 0.0361 | Lower reconstruction error |
| **PRD (%)** | **7.78%** | 12.70% | Clinically "Very Good" (< 9% threshold) |
| **Quality Score (QS)** | **0.7511** | 0.4602 | Higher overall compression efficiency |

---

### 2. Stage 2: Arrhythmia Classification & Class-Imbalance Optimization (MIT-BIH)
To address the severe class imbalance in MIT-BIH (where Normal beats represent ~82.8% of the dataset), we evaluated both the exact paper baseline and an optimized variant incorporating **Smoothed Class Weights** and **ReduceLROnPlateau**:

| Metric | Baseline Model | Class-Weighted Optimized | Paper (Table 5) |
| :--- | :---: | :---: | :---: |
| **Accuracy (%)** | **97.80%** | 97.57% | 98.57% |
| **Macro Precision (%)** | **91.27%** | 87.74% | 93.13% |
| **Macro Sensitivity / Recall (%)** | 86.72% | **89.78%** | 90.30% |
| **Macro Specificity (%)** | 98.35% | **98.61%** | 98.99% |
| **Macro F1-Score (%)** | **88.82%** | 88.72% | 91.66% |
| **Inference Latency** | **0.0753 ms/beat** | **0.0753 ms/beat** | — |

---

### 3. Ablation Study & Architectural Comparison
Comparing the proposed compressed pipeline against a standard `1D-CNN` and an uncompressed `Raw BiLSTM-MHA` operating directly on 187-point signals:

| Model Architecture | Accuracy | Macro F1 | Total Parameters | Inference (ms/beat) |
| :--- | :---: | :---: | :---: | :---: |
| **1D-CAE + BiLSTM-MHA (Proposed)** | **97.80%** | **88.82%** | **1,115,285** | **0.2268 ms** |
| **Baseline 1D-CNN** | 98.14% | 89.96% | 196,005 | 0.0967 ms |
| **Raw BiLSTM-MHA (No CAE)** | 98.31% | 91.06% | 3,634,053 | 0.2932 ms |

> **Key Takeaway:** Incorporating the `1D-CAE` reduces total model parameters by **69.3%** (from 3.63M to 1.12M) and compresses transmission bandwidth by **82.9%**, making it suitable for resource-constrained wearable ECG devices.

---

### 4. Cross-Dataset Generalization (PTB Diagnostic Dataset)
To validate generalization across unseen clinical populations, the `1D-CAE` (trained solely on MIT-BIH) was applied in a **zero-shot** setting to extract 32-dim features from the **PTB Diagnostic ECG Database** (14,552 beats: Normal vs. Myocardial Infarction):

| Dataset | Accuracy | Precision | Sensitivity | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PTB Diagnostic (Test Set)** | **96.77%** | **97.58%** | **97.95%** | **93.70%** | **97.77%** | **99.07%** |

---

## 🖥️ Interactive Streamlit Web Application

An interactive diagnostic dashboard (`app.py`) is included to demonstrate real-time ECG compression, signal reconstruction, latent space visualization, and arrhythmia classification:

* Switch seamlessly between **MIT-BIH (5-class AAMI)** and **PTB Diagnostic (Normal vs. MI)** tasks.
* Compare **Baseline** vs. **Class-Weighted Optimized** classifier predictions.
* Inspect real-time **RMSE**, **PRD (%)**, **32-dim bottleneck activations**, and class probability distributions.
* Upload custom ECG CSV files (`187` or `188` columns) or test using pre-loaded demo samples.

---

## 📁 Repository Structure

```text
├── mit.ipynb                   # Complete Google Colab notebook (Data prep, Training, Ablation, PTB Bonus)
├── app.py                      # Interactive Streamlit web application for live inference
├── cae_full.keras              # Trained 1D-Convolutional Autoencoder (Encoder + Decoder)
├── cae_encoder.keras           # Standalone 1D-CAE Encoder (187 -> 32 feature extractor)
├── bilstm_mha_best.keras       # Trained BiLSTM + Multi-Head Attention classifier (Baseline)
├── bilstm_mha_optimized.keras  # Class-weighted & LR-scheduled BiLSTM-MHA classifier
├── ptb_bilstm_mha.keras        # Cross-dataset classifier trained on PTB Diagnostic features
├── mitbih_demo_samples.csv     # Balanced sample beats (50 beats across 5 classes) for live demo
├── ptb_demo_samples.csv        # Sample beats (20 beats: Normal & MI) for live demo
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/ECG-CAE-BiLSTM-Attention.git](https://github.com/YOUR_USERNAME/ECG-CAE-BiLSTM-Attention.git)
cd ECG-CAE-BiLSTM-Attention
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📚 Datasets & Reference

* **Datasets:** [ECG Heartbeat Categorization Dataset (MIT-BIH & PTB Diagnostic)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) — sampled at 125 Hz and segmented into 187 time steps per beat following AAMI EC57 standards.
* **Reference Paper:** H. Degachi et al., *"ECG Heartbeat Classification Using CNN Autoencoder Feature Extraction and Attention-Augmented BiLSTM Classifier,"* IEEE Access, 2025.

---

## 👤 Author
* **Your Name** — [GitHub Profile](https://github.com/YOUR_USERNAME)
