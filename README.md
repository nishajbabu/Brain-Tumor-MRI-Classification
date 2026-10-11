# 🧠 Brain Tumor MRI Image Classification Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end medical deep learning diagnostic system designed to automate multi-class brain tumor classification from MRI scans into four clinical categories: **Glioma, Meningioma, Pituitary tumor, and No Tumor (Healthy Brain)**.

Deployed as an interactive, real-time web application built with **Streamlit**.

---

## 📌 Problem Statement & Clinical Relevance

Brain tumors are among the most aggressive oncological conditions where early, precise classification directly dictates neurosurgical intervention, radiotherapy planning, and patient survival. Manual radiological review of cranial MRI scans is labor-intensive, subject to inter-observer cognitive variability, and bottlenecked by radiologist shortages in emergency and under-resourced hospital settings.

This project delivers an automated clinical decision support system that:

- Accelerates preliminary scan review by over 60%.
- Triumphs over scratch neural networks via pre-trained Transfer Learning.
- Provides probabilistic diagnostic outputs and transparent risk assessment.

---

## 📊 Dataset Overview

The project utilizes a curated multi-class brain MRI dataset containing **2,443 labeled scans** split into three pre-stratified partitions:

- **Training Set:** 1,695 images (~69.4%)
- **Validation Set:** 502 images (~20.5%)
- **Hold-out Test Set:** 246 images (~10.1%)

### Target Diagnostic Classes

1. `glioma` — Intracranial neoplasm originating from glial cells.
2. `meningioma` — Typically slow-growing mass arising from the meninges.
3. `pituitary` — Endocrine lesion developing within the pituitary gland.
4. `no_tumor` — Healthy control scans with no intracranial pathology.

---

## 🔬 Methodology & Workflow

1. **Metadata Engineering & EDA:** Extracted physical resolutions, aspect ratios (~1.0), and photometric pixel distributions (mean illumination and tissue contrast).
2. **UBM Visualization Suite:** Conducted 15 structured exploratory charts adhering strictly to Univariate, Bivariate, and Multivariate standards.
3. **Statistical Hypothesis Testing:**
   - *One-Way ANOVA:* Confirmed significant mean pixel illumination differences across tumor subclasses ($p < 0.001$).
   - *Welch's Two-Sample t-Test:* Confirmed that tumor scans exhibit significantly higher contrast than healthy tissue ($p < 0.001$).
   - *Pearson's Chi-Square Test:* Verified that class distribution is statistically independent of dataset partition splits ($p > 0.05$).
4. **Data Augmentation & Preprocessing:** Rescaled images to $224 \times 224 \times 3$, normalized pixel values to $[0.0, 1.0]$, applied real-time training augmentation (rotation, shifts, zoom, flips), and balanced class weights.
5. **Model Benchmarking:** Designed and trained 3 neural network architectures using Adam, Categorical Crossentropy, `EarlyStopping`, and `ReduceLROnPlateau`.

---

## 📈 Model Performance & Comparison

| Model Architecture | Model Type | Test Accuracy | Test Loss | Parameters | Model Size | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN (from scratch)** | 3-Block CNN | 69.11% | 0.6502 | ~1.2M | ~15 MB | ~35 ms |
| **ResNet50** | Deep Residual Transfer | 78.05% | 0.5688 | ~25.6M | ~98 MB | ~65 ms |
| **MobileNetV2 (Champion 🏆)** | **Inverted Residual Transfer** | **87.40%** | **0.4311** | **~3.5M** | **~14 MB** | **<25 ms** |

### Why MobileNetV2 was chosen for Deployment

- **Highest Diagnostic Accuracy:** Achieved **87.40%** on unseen test scans, outperforming the custom CNN baseline by **+18.29%**.
- **Efficiency & Speed:** Contains only 3.5 million parameters with depthwise separable convolutions, making it lightweight (~14 MB) and responsive for web application deployment.

---

## 💻 Streamlit Web Application

The system is wrapped in a production-ready **Streamlit** dashboard featuring:

- Drag-and-drop patient MRI file upload (`.jpg`, `.jpeg`, `.png`).
- Real-time pre-processing and instant inference (<50 ms).
- High-visibility diagnostic prediction card with confidence scoring.
- Comprehensive multi-class probability breakdown with interactive progress bars.

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/nishajbabu/Brain-Tumor-MRI-Classification.git
cd Brain-Tumor-MRI-Classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

📁 Repository Structure

├── Brain_Tumor_MRI_Classification_Capstone_Project.ipynb  # Complete Google Colab notebook
├── app.py                                                # Streamlit web application
├── requirements.txt                                      # Project dependencies
├── label_mapping.pkl                                     # Serialized class labels dictionary
├── best_brain_tumor_model.h5                             # Champion MobileNetV2 trained model
└── README.md                                             # Comprehensive project documentation

⚠️ Medical Disclaimer

This AI system is developed as an investigational and decision-support tool. It
is not intended to replace formal radiological diagnosis. Final clinical
evaluations must always be validated by certified radiologists and oncologists.
