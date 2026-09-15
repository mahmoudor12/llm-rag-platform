Ja — hier ist eine deutlich professionellere GitHub-README, stärker auf **Portfolio, ML Engineering und Recruiter/Reviewer-Sicht** optimiert. Ich habe insbesondere Hero-Section, Badges, Quick Start, Pipeline-Architektur, Demo-Platzhalter, Results, XAI und MLOps aufgewertet.

# 🧠 MedVision AI

### Explainable Brain Tumor Classification from MRI

<p align="center">

**End-to-End Deep Learning · Explainable AI · MLOps · Medical Imaging**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow\&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker\&logoColor=white)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6)](https://dvc.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</p>

> ⚠️ **Research Prototype — Not for Clinical Use**
>
> MedVision AI is an educational and research-oriented portfolio project. Model predictions must **not** be used for medical diagnosis or clinical decision-making.

---

## 🚀 Project Overview

**MedVision AI** is an end-to-end computer vision pipeline for classifying brain MRI images into four categories:

| Class           | Description       |
| --------------- | ----------------- |
| 🧠 `glioma`     | Glioma            |
| 🧠 `meningioma` | Meningioma        |
| 🧠 `pituitary`  | Pituitary tumor   |
| ✅ `notumor`     | No tumor detected |

The project combines a pretrained **DenseNet-121** model with a two-stage transfer-learning strategy and **Grad-CAM** explanations.

The complete workflow is designed around reproducible ML engineering principles:

```text
Raw MRI Data
     │
     ▼
Data Validation & Split
     │
     ▼
Preprocessing & Augmentation
     │
     ▼
DenseNet-121 Transfer Learning
     │
     ├──────────────► MLflow Tracking
     │
     ▼
Validation & Model Selection
     │
     ▼
Hold-out Test Evaluation
     │
     ├──────────────► Metrics & Confusion Matrix
     │
     └──────────────► Grad-CAM Explainability
     │
     ▼
FastAPI Inference Service
     │
     ▼
Docker Deployment
```

---

# ✨ Key Features

* 🔬 **Medical Image Classification**
* 🧠 **DenseNet-121** with ImageNet pretraining
* 🔄 **Two-stage Transfer Learning**
* 🎯 **Four-class MRI classification**
* 🔥 **Grad-CAM Explainable AI**
* 📊 **Sensitivity / Specificity / Precision / F1**
* 📈 **ROC-AUC and Confusion Matrix**
* 🧪 **MLflow Experiment Tracking**
* 📦 **DVC-ready Data Versioning**
* 🚀 **FastAPI Inference API**
* 🐳 **Docker-ready Deployment**
* ⚙️ **Centralized YAML Configuration**
* 🛡️ **Confidence Thresholding**
* 🔁 **Reproducible Train/Validation/Test Split**

---

# 🖼️ Demo

## Model Prediction + Grad-CAM

> Replace the placeholder below with an actual generated Grad-CAM image.

<p align="center">

```text
┌──────────────────┬──────────────────┬──────────────────┐
│                  │                  │                  │
│   Original MRI   │    Grad-CAM      │     Overlay      │
│                  │                  │                  │
│                  │                  │                  │
└──────────────────┴──────────────────┴──────────────────┘

Prediction: Glioma
Confidence: 94.7%
Status: Confident prediction
```

</p>

**Example file:**

```text
runs/gradcam_glioma.png
```

---

## 📸 API Demo

Add a screenshot of the FastAPI Swagger interface here:

```text
docs/
└── api_demo.png
```

Recommended README placement:

```markdown
![FastAPI Demo](docs/api_demo.png)
```

---

# 🏗️ System Architecture

```text
                              ┌─────────────────────┐
                              │    MRI Dataset      │
                              │      7,200 Images   │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │ Data Processing     │
                              │                     │
                              │ • Validation        │
                              │ • Stratified Split  │
                              │ • Preprocessing     │
                              └──────────┬──────────┘
                                         │
                                         ▼
                       ┌─────────────────────────────────┐
                       │        Train / Val / Test       │
                       │           70 / 15 / 15          │
                       └────────────────┬────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
               ┌──────────────────┐          ┌──────────────────┐
               │ Training         │          │ Validation       │
               │                  │          │                  │
               │ DenseNet-121     │◄─────────┤ Metrics          │
               │ Transfer Learning│          │ Model Selection  │
               └────────┬─────────┘          └──────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Fine-Tuning      │
               │                  │
               │ Upper Backbone   │
               │ Lower Learning   │
               │ Rate             │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Best Model       │
               │ best_model.pt    │
               └────────┬─────────┘
                        │
              ┌─────────┴──────────┐
              │                    │
              ▼                    ▼
     ┌─────────────────┐   ┌──────────────────┐
     │ Test Evaluation │   │ Grad-CAM         │
     │                 │   │                  │
     │ Accuracy        │   │ Heatmaps         │
     │ F1              │   │ Explanations     │
     │ ROC-AUC         │   │                  │
     │ Sensitivity     │   │                  │
     │ Specificity     │   │                  │
     └────────┬────────┘   └──────────────────┘
              │
              ▼
     ┌─────────────────────┐
     │ FastAPI Inference   │
     │                     │
     │ POST /predict       │
     └──────────┬──────────┘
                │
                ▼
        ┌───────────────┐
        │ Docker        │
        │ Deployment    │
        └───────────────┘

        ─────────────────────────────
        MLflow → Experiments / Metrics
        DVC    → Dataset Versioning
        ─────────────────────────────
```

---

# ⚡ Quick Start

## 1. Clone Repository

```bash
git clone <repo-url>
cd medvision-ai
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

## 4. Prepare Dataset

Download the dataset from Kaggle:

**Brain Tumor MRI Dataset**

```text
data/raw/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

---

# 🔬 Reproducible Pipeline

## Step 1 — Create Data Splits

```bash
python -m medvision.data.split \
    --config configs/config.yaml
```

Expected:

```text
Total:      7200
Train:      5040
Validation: 1080
Test:       1080
```

Generated files:

```text
data/splits/
├── train.csv
├── val.csv
└── test.csv
```

---

## Step 2 — Train Model

```bash
python -m medvision.train \
    --config configs/config.yaml
```

Training consists of two phases.

### Phase 1 — Feature Extraction

```text
DenseNet-121 Backbone
        │
        │ frozen
        ▼
Classification Head
        │
        ▼
4 Classes
```

### Phase 2 — Fine-Tuning

The upper part of DenseNet-121 is unfrozen and trained using a lower learning rate.

```text
DenseNet-121
     │
     ├── Lower Layers → Frozen
     │
     └── Upper Layers → Trainable
                         │
                         ▼
                    Classifier
```

Best checkpoint:

```text
runs/best_model.pt
```

---

## Step 3 — Evaluate

```bash
python -m medvision.evaluate \
    --config configs/config.yaml \
    --split test
```

Generated evaluation artifacts include:

```text
runs/
├── eval_report.json
├── confusion_matrix.png
└── ...
```

---

## Step 4 — Generate Grad-CAM

```bash
python -m medvision.explain \
    --config configs/config.yaml \
    --image data/raw/Testing/glioma/Te-gl_224.jpg \
    --output runs/gradcam_glioma.png
```

Output:

```text
Original MRI
      +
Grad-CAM Heatmap
      +
Prediction / Confidence
```

---

# 🧠 Model Architecture

MedVision AI uses **DenseNet-121** as its feature extractor.

```text
Input
3 × 224 × 224
      │
      ▼
┌──────────────────────┐
│ DenseNet-121         │
│ ImageNet Pretrained  │
└──────────┬───────────┘
           │
           ▼
Global Average Pooling
           │
           ▼
Linear: 1024 → 256
           │
           ▼
ReLU
           │
           ▼
Dropout: 0.3
           │
           ▼
Linear: 256 → 4
           │
           ▼
Softmax
           │
           ▼
┌──────────┴───────────┐
│                      │
Glioma             Meningioma
Pituitary          No Tumor
```

---

# ⚙️ Training Configuration

| Parameter          |             Value |
| ------------------ | ----------------: |
| Backbone           |      DenseNet-121 |
| Pretraining        |          ImageNet |
| Optimizer          |              Adam |
| Initial LR         |            `1e-3` |
| Fine-Tuning LR     |            `1e-4` |
| Weight Decay       |            `1e-4` |
| CPU Batch Size     |              `16` |
| GPU Batch Size     |              `32` |
| Scheduler          | ReduceLROnPlateau |
| Scheduler Factor   |             `0.1` |
| Scheduler Patience |               `3` |
| Early Stopping     |               `7` |
| Loss               |  CrossEntropyLoss |
| Input Size         |       `224 × 224` |

---

# 🧪 Data Augmentation

Training images are augmented using:

* Rotation ±15°
* Horizontal Flip `p=0.5`
* RandomResizedCrop
* Brightness adjustment
* Contrast adjustment

Validation and test data use deterministic preprocessing without random augmentation.

---

# 📊 Results

> ⚠️ **Results will be populated after the final training run.**
>
> The values below are intentionally marked as `TBD` and do **not** represent measured performance.

## Overall Performance

| Metric        | Validation | Test |
| ------------- | ---------: | ---: |
| Accuracy      |        TBD |  TBD |
| Macro F1      |        TBD |  TBD |
| ROC-AUC (OVR) |        TBD |  TBD |

## Per-Class Performance

| Class      | Sensitivity | Specificity | Precision |  F1 |
| ---------- | ----------: | ----------: | --------: | --: |
| Glioma     |         TBD |         TBD |       TBD | TBD |
| Meningioma |         TBD |         TBD |       TBD | TBD |
| Pituitary  |         TBD |         TBD |       TBD | TBD |
| No Tumor   |         TBD |         TBD |       TBD | TBD |

### Evaluation Artifacts

```text
runs/
├── eval_report.json
├── confusion_matrix.png
├── classification_report.txt
└── ...
```

---

# 📈 Results Dashboard

For the final portfolio version, the following artifacts should be added to `docs/results/`:

```text
docs/
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── training_curves.png
    ├── class_metrics.png
    └── gradcam_examples.png
```

Then displayed in the README:

### Confusion Matrix

![Confusion Matrix](docs/results/confusion_matrix.png)

### Training Curves

![Training Curves](docs/results/training_curves.png)

### Grad-CAM Examples

![Grad-CAM Examples](docs/results/gradcam_examples.png)

---

# 🔥 Explainable AI

MedVision AI integrates **Grad-CAM** to visualize image regions that contribute to the model's prediction.

Conceptually:

```text
                  MRI
                   │
                   ▼
            DenseNet-121
                   │
                   ▼
             Target Class
                   │
                   ▼
             Gradients
                   │
                   ▼
          Feature Activation
                   │
                   ▼
             Grad-CAM
                   │
                   ▼
            Heatmap Overlay
```

Example interpretation:

```text
Prediction:
Glioma

Confidence:
0.947

Explanation:
Highlighted regions indicate areas
that contributed strongly to the prediction.
```

> Grad-CAM is an **explanation mechanism, not a medical validation method**. Highlighted regions do not prove that the model identified a clinically meaningful tumor region.

---

# 🧩 MLOps

The project follows a lightweight MLOps architecture.

```text
              ┌───────────────┐
              │ Dataset       │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ DVC           │
              │ Data Version  │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Training      │
              │ PyTorch       │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ MLflow        │
              │ Experiments   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Model         │
              │ best_model.pt │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ FastAPI       │
              │ Inference     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Docker        │
              └───────────────┘
```

---

# 📊 MLflow

Start the MLflow dashboard:

```bash
mlflow ui --backend-store-uri mlruns
```

Open:

```text
http://localhost:5000
```

MLflow tracks:

* Hyperparameters
* Training loss
* Validation loss
* Accuracy
* F1
* Learning rates
* Model artifacts
* Experiment runs

This enables systematic comparison between training experiments.

---

# 🚀 FastAPI

The trained model can be exposed through a REST API.

Conceptually:

```text
Client
  │
  │ POST /predict
  │
  ▼
FastAPI
  │
  ▼
Image Preprocessing
  │
  ▼
DenseNet-121
  │
  ▼
Prediction
  │
  ├── Class
  ├── Confidence
  └── Disclaimer
```

Example response:

```json
{
  "prediction": "glioma",
  "confidence": 0.947,
  "status": "prediction",
  "clinical_disclaimer": "Research prototype. Not for clinical use."
}
```

Predictions below the configured confidence threshold are flagged for manual review.

Current threshold:

```text
0.85
```

> Confidence scores should not be interpreted as calibrated probabilities or as a measure of clinical certainty.

---

# 🐳 Docker

The application is designed to be containerized for reproducible deployment.

Example workflow:

```bash
docker build -t medvision-ai .
```

Run:

```bash
docker run -p 8000:8000 medvision-ai
```

API documentation can then be exposed through FastAPI's automatically generated documentation.

---

# 📁 Project Structure

```text
medvision-ai/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── splits/
│   └── ...
│
├── docs/
│   ├── architecture.png
│   ├── api_demo.png
│   └── results/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── training_curves.png
│       └── gradcam_examples.png
│
├── src/
│   └── medvision/
│       ├── data/
│       │   ├── dataset.py
│       │   ├── transforms.py
│       │   └── split.py
│       │
│       ├── models/
│       │   └── densenet.py
│       │
│       ├── train.py
│       ├── evaluate.py
│       └── explain.py
│
├── runs/
│   ├── best_model.pt
│   ├── eval_report.json
│   └── ...
│
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🛡️ Responsible AI & Limitations

This project explicitly acknowledges several limitations.

### No clinical validation

The model has not undergone clinical validation or regulatory approval.

### No reliable patient IDs

The source dataset does not provide sufficient patient-level identifiers for a robust patient-independent evaluation.

### Potential data leakage

The available images may contain related or augmented samples. A file-level split cannot fully exclude patient-level leakage.

### Dataset bias

Performance on a public dataset does not guarantee generalization to:

* different hospitals
* different scanners
* different MRI protocols
* different patient populations
* different image resolutions
* real-world clinical workflows

### Explainability limitations

Grad-CAM provides an approximate visualization of model attention and should not be treated as causal evidence.

---

# 📚 Dataset

**Brain Tumor MRI Dataset — Kaggle**

Dataset:

[https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset combines images from multiple public sources, including:

* figshare
* SARTAJ
* Br35H

The dataset should be used according to its applicable license and terms.

---

# 🧰 Tech Stack

| Category            | Technology                  |
| ------------------- | --------------------------- |
| Language            | Python                      |
| Deep Learning       | PyTorch                     |
| Computer Vision     | TorchVision, OpenCV, Pillow |
| Model               | DenseNet-121                |
| Data Processing     | pandas, NumPy               |
| Evaluation          | scikit-learn                |
| Explainability      | Grad-CAM                    |
| Experiment Tracking | MLflow                      |
| Data Versioning     | DVC                         |
| API                 | FastAPI                     |
| Server              | Uvicorn                     |
| Containerization    | Docker                      |
| Configuration       | YAML                        |
| Progress            | tqdm                        |

---

# 🗺️ Roadmap

* [x] DenseNet-121 transfer learning
* [x] Two-stage training
* [x] Train/Validation/Test split
* [x] Evaluation pipeline
* [x] Grad-CAM integration
* [x] MLflow experiment tracking
* [x] FastAPI architecture
* [x] Docker-ready structure
* [ ] Final benchmark results
* [ ] Automated CI/CD
* [ ] Model registry integration
* [ ] Patient-level validation dataset
* [ ] External validation
* [ ] Calibration analysis
* [ ] Automated API tests
* [ ] Production monitoring

---

# 🤝 Contributing

Contributions, suggestions and improvements are welcome.

A typical workflow:

```bash
git checkout -b feature/my-feature
```

Make your changes, add tests where appropriate, and open a pull request.

---

# 📄 License

This project is released under the **MIT License**.

See [LICENSE](LICENSE) for details.

The dataset itself may have separate licensing conditions.

---

# 👤 Author

**Mahmoud**

MedVision AI was created as a portfolio project demonstrating practical skills in:

* Deep Learning
* Medical Computer Vision
* Transfer Learning
* Explainable AI
* Data Engineering
* Model Evaluation
* Experiment Tracking
* MLOps
* API Development
* Containerization

---

# ⚠️ Medical Disclaimer

**MedVision AI is a research and educational prototype.**

It is **not a medical device**, has not been clinically validated, and must not be used to diagnose, treat, monitor, or make decisions about patients.

Model outputs, confidence scores and Grad-CAM visualizations are provided for research and demonstration purposes only.

**Always consult a qualified medical professional for medical decisions.**

---

<p align="center">

### 🧠 MedVision AI

**Making medical AI more explainable, reproducible and transparent.**

</p>

Diese Version ist bewusst so aufgebaut, dass sie auf GitHub wie ein **echtes ML-Portfolio-Projekt** wirkt: oben sofort Stack + Zweck, danach Demo/Architektur, dann reproduzierbarer Quick Start, messbare Results, XAI und schließlich MLOps/Limitations. Besonders wichtig wäre als nächster Schritt, die `TBD`-Werte durch die **echten Trainingsergebnisse** zu ersetzen und die drei bis vier `docs/results/*.png`-Grafiken tatsächlich ins Repository zu legen.
