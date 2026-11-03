# AI-Based Plant Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.11-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c8a2b.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003b57.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A modern, professional, fully responsive AI-powered web application that helps farmers and agricultural students identify plant diseases by uploading images of plant leaves. Suitable for **college final-year project demonstrations**, agronomic research, and extension fieldwork.

---

## 🌿 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Plant Disease Library](#-plant-disease-library)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Testing the System](#-testing-the-system)
- [Honest AI Transparency](#-honest-ai-transparency)
- [Folder Structure](#-folder-structure)
- [Screenshots & UI Design](#-screenshots--ui-design)

---

## 🌟 Project Overview

Plant diseases severely diminish agricultural crop yield, causing substantial financial loss to farmers and threatening global food security. Traditional identification methods rely on manual physical scouting by trained agronomists, which is slow, expensive, and inaccessible in rural communities.

The **AI-Based Plant Disease Detection System** provides an automated, instant, and accessible web-based diagnostic solution. By combining **Deep Convolutional Neural Networks (CNN)** with **OpenCV computer vision preprocessing**, the system accurately segments leaves, detects fungal/bacterial lesions, computes greenness and damage indices, predicts disease categories, and outputs actionable chemical and organic treatment protocols.

---

## 🚀 Key Features

### 1. 🏡 Modern Agricultural Home Page
- **Hero Section**: Crisp typography with headline *"Detect Plant Diseases with AI"* and subtitle *"Protect Your Crops with Intelligent Plant Disease Detection."*
- **Aesthetic Design**: Agricultural color palette (deep forest greens, fresh mint, cream, and warm amber accents).
- **Interactive Visual Cards**: Live visual sample with simulated lesion contour analysis.
- **Workflow Guide**: 3-step visualization from image capture to diagnosis.

### 2. 🔬 Disease Detection & Optical Scanner
- **Multi-Format Upload**: Drag-and-drop support for **JPG, JPEG, PNG, and WEBP** images.
- **Instant Sample Leaves**: 6 pre-bundled realistic botanical leaf specimens (Tomato Early Blight, Healthy Tomato, Potato Late Blight, Apple Scab, Corn Common Rust, Grape Black Rot) for **1-click evaluation** without needing personal photos.
- **Animated Laser Scanner**: Visual laser scanning beam sweeping across the preview image during processing.
- **Comprehensive Results Card**:
  - Plant species and scientific disease name.
  - Health status badge (**Healthy** vs. **Diseased** with severity rating).
  - Confidence percentage with animated progress meter.
  - Top-3 prediction likelihood distribution.
  - **OpenCV Computer Vision HUD**: Leaf area percentage, lesion density, discrete lesion count, greenness index, and processed diagnostic overlay.
  - Printable / PDF **Diagnostic Report Generator**.

### 3. 📊 Agronomic Surveillance Dashboard
- **Metric Cards**: Total images analyzed, healthy crops detected, diseased crops detected, average AI confidence.
- **Interactive Chart.js Visualizations**:
  - *Disease Distribution* (Doughnut chart).
  - *Plant Species Breakdown* (Bar chart).
  - *Diagnostic Scan Timeline Trends* (Line chart).
- **Recent Scan History Table**: Thumbnails, plant species, disease diagnosis, health status badges, timestamps, and detail inspector modals.
- **History Management**: One-click history purge with safety prompt.

### 4. 📚 Comprehensive Plant Disease Library
Searchable and filterable reference encyclopedia covering:
1. **Tomato Early Blight** (*Alternaria solani*)
2. **Tomato Late Blight** (*Phytophthora infestans*)
3. **Tomato Leaf Mold** (*Passalora fulva*)
4. **Potato Early Blight** (*Alternaria solani*)
5. **Potato Late Blight** (*Phytophthora infestans*)
6. **Apple Scab** (*Venturia inaequalis*)
7. **Grape Black Rot** (*Guignardia bidwellii*)
8. **Corn Common Rust** (*Puccinia sorghi*)
9. **Healthy Leaves** (Tomato, Potato, Corn, Apple, Grape)
- Each item includes: High-definition visual specimen, symptoms, causes, organic treatments, chemical controls, and long-term prevention methods.

### 5. 📖 About & Architecture Page
- Project overview, objectives, and academic motivation.
- Interactive end-to-end architecture pipeline diagram.
- Technology stack specifications.
- Farmer and student empowerment benefits.
- Future roadmap (drone multispectral imaging, IoT soil telemetry, multilingual audio support).

### 6. 📩 Contact & Feedback Portal
- Interactive contact form with role selection (*Farmer, Student, Agronomist, Researcher, Other*).
- Form validation and instant toast notification feedback.
- Persistent logging to SQLite `contact_messages` table.
- Laboratory headquarters location and emergency helpline information.

---

## 🏗️ System Architecture

```
[ User Browser / Device ]
       │
       ▼ (HTTP POST Multipart / JSON)
[ Flask REST API Server (app.py) ]
       │
  ┌────┴──────────────────────────────┐
  ▼                                   ▼
[ OpenCV Engine ]             [ AI Inference Engine ]
 - HSV Color Space Mask        - MobileNetV2 / CNN Model
 - Leaf vs Background Cutout   - Softmax Probabilities
 - Lesion Contour Detection    - Top-3 Candidates Breakdown
 - Greenness & Damage Index    - Transparent Demo AI Mode
  │                                   │
  └────┬──────────────────────────────┘
       │
       ▼
[ SQLite Database (plant_doctor.db) ]
 - Table: detections (history, confidence, severity)
 - Table: contact_messages (inquiries & feedback)
       │
       ▼
[ Frontend SPA (HTML5 / CSS3 / Vanilla JS / Chart.js) ]
 - Live Diagnostic Results & OpenCV HUD Overlay
 - Chart.js Epidemiological Graphs
 - Printable Diagnostic Report
```

---

## 💻 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | HTML5, CSS3, Vanilla ES6+ JS | Glassmorphic, agricultural UI, responsive mobile-first SPA |
| **Icons & Charts**| FontAwesome 6, Chart.js 4.x | Professional agricultural dashboard charts & vectors |
| **Backend** | Python 3.11, Flask 3.1 | RESTful web service, file routing, CORS, and endpoint handlers |
| **Image Processing** | OpenCV (`cv2`), Pillow | Leaf segmentation, lesion contours, HUD visual overlays |
| **Machine Learning**| TensorFlow 2.x / Keras CNN | Deep learning transfer learning classification model |
| **Database** | SQLite3 | ACID-compliant embedded relational data store |

---

## 🔒 Honest AI Transparency

Per core system requirements:
- **No Fake Predictions**: The system **never displays simulated results as real AI predictions**.
- When a trained TensorFlow CNN model (`models/plant_disease_model.keras` or `.h5`) is active, a green badge (`Trained CNN Model Active`) is displayed with exact inference latency.
- When weights are not loaded, the application transparently displays an amber badge (`Demo AI Mode - Agronomic Heuristics`) explicitly disclosing that predictions are derived from computer vision color/lesion heuristics for testing and demonstration.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Git (optional)

### 1. Clone or Open the Project
```bash
cd "c:\Users\user\Documents\ai based"
```

### 2. Activate Virtual Environment
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

### 1. Generate Sample Botanical Leaves (Already Included)
```bash
python create_sample_leaves.py
```

### 2. (Optional) Generate CNN Model Weights
```bash
python train_or_generate_model.py
```

### 3. Start the Flask Server
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Testing the System

An automated unit and integration test suite is included:

```bash
python -m unittest tests/test_app.py
```

**Test Coverage:**
- `test_01_index_route`: Verifies homepage renders correctly with title and hero sections.
- `test_02_status_endpoint`: Validates AI engine and OpenCV availability flags.
- `test_03_sample_images_endpoint`: Asserts 6 sample test leaves are available.
- `test_04_detect_sample_leaf`: Tests leaf detection, confidence calculation, and OpenCV lesion metrics.
- `test_05_detect_healthy_sample_leaf`: Validates healthy leaf identification.
- `test_06_dashboard_stats`: Confirms Chart.js aggregation endpoints return valid distributions.
- `test_07_disease_library`: Tests multi-crop encyclopedia filters.
- `test_08_contact_submission`: Tests SQLite contact form persistence.

---

## 📂 Folder Structure

```
ai-based-plant-disease-detection/
├── app.py                      # Flask web server & REST endpoints
├── database.py                 # SQLite database schema, helpers & seed data
├── model_service.py            # AI prediction pipeline & OpenCV processing
├── create_sample_leaves.py     # Botanical sample leaf generator
├── train_or_generate_model.py  # CNN model architecture builder
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation
├── plant_doctor.db             # SQLite database file
├── models/
│   ├── class_indices.json      # Mapping of 13 disease categories
│   └── plant_disease_model.keras (optional CNN weights)
├── static/
│   ├── css/
│   │   ├── style.css           # Agricultural modern UI styles
│   │   └── dashboard.css       # Dashboard and Chart.js styles
│   ├── js/
│   │   ├── app.js              # SPA navigation & toast notifications
│   │   ├── detector.js         # Leaf upload, scan animation & results
│   │   ├── dashboard.js        # Chart.js charts & history table
│   │   ├── library.js          # Disease library search & modals
│   │   └── contact.js          # Contact form handler
│   ├── images/
│   │   └── sample_leaves/      # Bundled test leaves
│   └── uploads/                # User uploaded & processed leaf images
├── templates/
│   └── index.html              # Modern single-page web application
└── tests/
    └── test_app.py             # Integration test suite
```

---

## 🎓 College Final-Year Project Demonstration Checklist

1. **Working Navigation**: Every single nav link (*Home, Detect Disease, Dashboard, Disease Library, About, Contact*) dynamically updates the view without page reloads.
2. **Interactive Optical Detection**: Test with any of the 6 quick sample pills (*Tomato Early Blight, Healthy Tomato, Potato Late Blight, Apple Scab, Corn Rust, Grape Black Rot*) or upload any leaf photo.
3. **OpenCV Computer Vision**: Click the *OpenCV Vision HUD* sub-tab on the detection card to view computed leaf area %, lesion density %, and the contour overlay image.
4. **Interactive Dashboard**: View real-time Chart.js doughnut, bar, and line graphs reflecting SQLite scan records.
5. **Printable Report**: Click *Export Diagnostic Report* to launch a clean, formatted medical-style plant diagnostic certificate ready for print or PDF save.
6. **SQLite Persistence**: Add scans or submit the contact form, then check the Dashboard and database table.
