# Deep Learning-Based Bloodstain Pattern Analysis System

**For Crime Scene Reconstruction with Impact Mechanism and Weapon Category Estimation**

> ⚠️ Academic & Research Use Only. Results are computational estimates and require validation by qualified forensic professionals.

---

## 🎯 Project Overview

A complete forensic analysis workstation combining:
- **ResNet50 Deep Learning** for spatter classification
- **OpenCV Computer Vision** for droplet extraction
- **Geometric Algorithms** for 2D convergence and 3D point of origin
- **Interactive Three.js 3D Viewer** for point-of-origin visualization
- **Flask REST API** backend with **HTML/JS/CSS** frontend

---

## 🧩 Four Core Modules

| Module | Description | Technology |
|--------|-------------|------------|
| **Module 1** | Automated Dataset Ingestion | Python, PIL, OpenCV |
| **Module 2** | Spatter Type & Velocity Classifier | PyTorch, ResNet50 |
| **Module 3** | Droplet Extraction & Impact Angle Engine | OpenCV, NumPy |
| **Module 4** | 2D Convergence & 3D Origin Resolver | SciPy, NumPy, Three.js |

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# 1. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python backend/app.py
```

Open **http://localhost:5000** in your browser.

---

## 📁 Project Structure

```
BPA/
├── backend/
│   ├── app.py                    ← Flask entry point
│   ├── config.py                 ← Configuration
│   ├── api/
│   │   ├── cases.py              ← Case CRUD API
│   │   ├── dataset.py            ← Dataset ingestion API
│   │   ├── classification.py     ← ML classifier API + training
│   │   ├── droplets.py           ← Droplet extraction API
│   │   ├── convergence.py        ← 2D/3D point of origin API
│   │   └── reports.py            ← Report generation API
│   ├── ml/
│   │   ├── model.py              ← ResNet50 dual-head model
│   │   ├── train.py              ← Training pipeline
│   │   ├── predict.py            ← Inference
│   │   ├── preprocessing.py      ← DL + CV preprocessing
│   │   └── dataset_ingest.py     ← Module 1 logic
│   ├── computer_vision/
│   │   └── droplet_detection.py  ← Module 3 CV pipeline
│   ├── geometry/
│   │   ├── convergence_2d.py     ← Module 4 — 2D convergence
│   │   └── origin_3d.py          ← Module 4 — 3D point of origin
│   ├── database/
│   │   └── models.py             ← TinyDB models
│   └── utils/
│       ├── logger.py
│       └── file_utils.py
├── frontend/
│   ├── index.html                ← Landing page
│   ├── dashboard.html            ← Main dashboard
│   ├── cases.html                ← Case management
│   ├── analysis.html             ← 7-step analysis pipeline
│   ├── classification.html       ← Standalone classifier
│   ├── droplets.html             ← Droplet analysis table
│   ├── convergence.html          ← 2D canvas visualization
│   ├── reconstruction.html       ← Three.js 3D viewer
│   ├── dataset.html              ← Dataset manager
│   ├── training.html             ← Model training page
│   ├── reports.html              ← Report generation
│   ├── css/main.css              ← Design system
│   └── js/main.js                ← Shared utilities
├── datasets/                     ← Ingested datasets
├── models/                       ← Trained checkpoints
├── uploads/                      ← Case image uploads
├── reports/                      ← Generated reports
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Preparation (Module 1)

Organize images into class subdirectories:

```
my_dataset/
├── Passive Drip/
│   ├── img001.jpg
│   └── ...
├── Transfer Smear/
│   └── ...
├── Medium Velocity Impact Spatter/
│   └── ...
└── High Velocity Impact Spatter/
    └── ...
```

ZIP this folder and upload via **Dataset Manager** page, or ingest a server-side path directly.

The system will:
- Validate images (format, dimensions, corruption)
- Remove duplicates (SHA-256 hash)
- Split 80% train / 10% val / 10% test
- Save metadata and statistics

---

## 🧠 Model Training (Module 2)

1. Go to **Dataset Manager** → upload/ingest dataset
2. Go to **Model Training** → configure hyperparameters:
   - Epochs: 20–100 recommended
   - Learning rate: 1e-4 default
   - Batch size: 16 default
3. Click **Start Training**
4. Monitor live loss curves and accuracy
5. Model saved to `models/bpa_classifier.pth`

**Architecture:** ResNet50 pretrained → Dropout(0.4) → Dual head (spatter 4-class + velocity 3-class)

---

## 🔵 Droplet Analysis (Module 3)

**Algorithm:**
```
Image → Grayscale → CLAHE → Gaussian Blur → Threshold (Otsu/manual)
→ Morphological Close+Open → Contour Detection → Ellipse Fitting
→ Impact Angle: θ = arcsin(W / L)
```

**Measurements per droplet:** centroid, major/minor axis, area, perimeter, aspect ratio, orientation, impact angle

---

## 📐 2D Convergence (Module 4)

**Algorithm:**
1. Filter elongated stains (aspect ratio ≥ 1.5)
2. Construct trajectory lines: `p(t) = centroid + t × direction`
3. Compute pairwise intersections
4. RANSAC outlier removal (2σ)
5. Inverse-distance weighted centroid = convergence point
6. RMS error reported

---

## 🌐 3D Point of Origin (Module 4)

**Mathematical model:**
```
For each droplet at (cx, cy, 0):
  φ = horizontal angle toward convergence point
  α = impact_angle (from arcsin formula)
  direction = (cos(φ)·cos(α), sin(φ)·cos(α), sin(α))
  
Solve: min Σ dist²(X, ray_i)
Via least-squares: (I - Σ ddᵀ) X = Σ (I - ddᵀ) o
```

---

## 🌐 API Reference

```
GET  /api/health
GET  /api/cases
POST /api/cases
GET  /api/cases/<id>
PUT  /api/cases/<id>
DELETE /api/cases/<id>

GET  /api/dataset/list
POST /api/dataset/upload         (ZIP)
POST /api/dataset/ingest         (local path)
GET  /api/dataset/statistics/<name>
DELETE /api/dataset/delete/<name>

POST /api/upload                 (image upload)
POST /api/classify               (run classification)
GET  /api/model/status
POST /api/model/train
GET  /api/model/train/progress

POST /api/droplets/extract
POST /api/droplets/angle

POST /api/convergence/2d
POST /api/reconstruction/3d

POST /api/report/generate
GET  /api/report/download/<case_id>?format=json|pdf
```

---

## ⚠️ Scientific Disclaimers

| Output | Label |
|--------|-------|
| Spatter type | **AI Prediction** |
| Impact velocity | **Model-based Classification** |
| Droplet dimensions | **Computer Vision Measurement** |
| Impact angle | **Geometric Estimate** |
| 2D convergence | **Geometric Estimation** |
| 3D point of origin | **Estimated Point of Origin** |

---

## 🔧 Troubleshooting

**Model not trained error:**
- Upload a dataset via Dataset Manager
- Train the model in the Training page
- Ensure `models/bpa_classifier.pth` exists

**No droplets detected:**
- Try threshold = 0 (Otsu auto)
- Reduce min_area
- Ensure image has sufficient contrast

**Too few droplets for convergence:**
- Need ≥ 3 elongated stains (AR ≥ 1.5)
- Adjust min_area parameters

---

## 📋 Limitations

- Model accuracy depends on training data quality and quantity
- Impact angle formula assumes flat surface and near-vertical perspective
- 3D point of origin calculation assumes ground plane is z=0
- No physical calibration (pixels, not mm) unless scale is provided
- GPU recommended for training large datasets

---

## 🎓 Academic Notes

This system is designed as a **final-year engineering project prototype**. It implements real algorithms documented in forensic science literature:

- Elliptical stain approximation: `θ = arcsin(W/L)`
- Trajectory line intersection for convergence
- Least-squares ray intersection for 3D origin (Kanatani, 1985)
- ResNet50 transfer learning (He et al., 2016)
