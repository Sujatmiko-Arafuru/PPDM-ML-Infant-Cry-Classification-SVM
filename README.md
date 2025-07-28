# 👶 Infant Cry Audio Classification using SVM from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

A comprehensive machine learning system for classifying infant crying sounds using Support Vector Machine (SVM) implemented from scratch with Sequential Minimal Optimization (SMO) algorithm. This project includes a complete pipeline from audio preprocessing to real-time web deployment.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/infant-cry-classification.git
cd infant-cry-classification

# 2. Automatic setup (Recommended)
python setup.py

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Run the Streamlit app
streamlit run streamlit_fix.py
```

The app will open at `http://localhost:8501`

## 📋 Table of Contents

- [Features](#-features)
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **🎵 Audio Processing**: Advanced audio preprocessing with noise reduction and normalization
- **🔧 Feature Extraction**: Comprehensive feature extraction including MFCC, spectral features, and time-domain features
- **🤖 SVM from Scratch**: Custom SVM implementation with SMO algorithm
- **🌐 Web Interface**: Real-time classification via Streamlit web app
- **📊 Visualization**: Interactive plots for audio analysis and predictions
- **🔄 Data Augmentation**: Audio augmentation techniques for improved model robustness
- **📈 Model Evaluation**: Comprehensive evaluation metrics and confusion matrix analysis

## 🎯 Project Overview

This project classifies infant crying sounds into 5 distinct categories:

| Category | Description | Audio Characteristics |
|----------|-------------|---------------------|
| **Belly Pain** | Stomach discomfort cries | Sharp, high-pitched, rhythmic |
| **Burping** | Need to burp | Short, repetitive patterns |
| **Discomfort** | General discomfort | Variable pitch, irregular |
| **Hungry** | Hunger cries | Long, continuous, escalating |
| **Tired** | Fatigue/sleepiness | Soft, whining, decreasing intensity |

## 🏗️ Architecture

```
Raw Audio Files → Preprocessing → Feature Extraction → SVM Training → Web Deployment
       ↓              ↓                ↓                ↓              ↓
   WAV/MP3      Normalization    MFCC + Spectral   SMO Algorithm   Streamlit App
   Files        Noise Filter     Time Features     OvR Strategy    Real-time
                Segmentation     (34 features)     Grid Search     Prediction
```

### Pipeline Stages

1. **Audio Preprocessing** (`preprocess1.py`)
   - Sample rate normalization (22,050 Hz)
   - Noise reduction with Butterworth filter
   - Audio segmentation (2-second segments)
   - Amplitude normalization

2. **Data Augmentation** (`preprocess2.py`)
   - Time stretching (±20%)
   - Pitch shifting (±2 semitones)
   - Gaussian noise addition
   - Time shifting (±0.5s)
   - Filter variations

3. **Feature Extraction** (`preprocess3.py`)
   - **Time Domain**: ZCR, RMS Energy (4 features)
   - **Frequency Domain**: Spectral Centroid, Bandwidth (4 features)
   - **MFCC**: 13 coefficients × 2 (mean & std) (26 features)
   - **Total**: 34 features per audio segment

4. **Model Training** (`svm_python.py`)
   - Custom SVM implementation with SMO
   - One-vs-Rest multiclass strategy
   - Grid search hyperparameter optimization
   - Cross-validation evaluation

## 📊 Dataset

The project uses the [DonateACry Corpus](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus) dataset:

```
dataset ppdm2 asli/
└── donateacry_corpus/
    ├── belly_pain/     # 1,000+ samples
    ├── burping/        # 1,000+ samples
    ├── discomfort/     # 1,000+ samples
    ├── hungry/         # 1,000+ samples
    └── tired/          # 1,000+ samples
```

### Dataset Statistics
- **Total Samples**: 5,000+ audio files
- **Audio Format**: WAV files
- **Duration**: Variable (1-10 seconds)
- **Sample Rate**: Variable (original)
- **Categories**: 5 balanced classes

## 🔧 Installation

### Option 1: Automatic Setup (Recommended)

```bash
# Clone and setup automatically
git clone https://github.com/yourusername/infant-cry-classification.git
cd infant-cry-classification
python setup.py
```

### Option 2: Manual Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/infant-cry-classification.git
cd infant-cry-classification

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Download Dataset

1. Download from [Kaggle Dataset](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus)
2. Extract to `dataset ppdm2 asli/donateacry_corpus/`
3. Ensure folder structure matches the categories above

## 🚀 Usage

### 1. Data Preprocessing

```bash
# Stage 1: Audio preprocessing
python preprocess1.py

# Stage 2: Audio augmentation
python preprocess2.py

# Stage 3: Feature extraction
python preprocess3.py
```

### 2. Model Training

```bash
# Train SVM model
python svm_python.py

# Or use Jupyter notebook
jupyter notebook svm.ipynb
```

### 3. Web Application

```bash
# Run Streamlit app
streamlit run streamlit_fix.py
```

### 4. Programmatic Usage

```python
from deploy_function import BabyCryClassifier

# Initialize classifier
classifier = BabyCryClassifier(
    model_path="best_svm_model.pkl",
    scaler_path="dataset_preprocessed/stage3/scaler.joblib"
)

# Predict single file
result = classifier.predict("audio_file.wav", return_probabilities=True)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")

# Batch prediction
results = classifier.predict_batch(["file1.wav", "file2.wav"])
```

## 📈 Model Performance

### Training Results
- **Best Hyperparameters**:
  - C: 10
  - Kernel: RBF
  - Gamma: 0.1
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~88%
- **Test Accuracy**: ~85%

### Per-Class Performance

| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Belly Pain | 87% | 85% | 86% |
| Burping | 83% | 81% | 82% |
| Discomfort | 84% | 86% | 85% |
| Hungry | 92% | 90% | 91% |
| Tired | 89% | 88% | 88% |

### Confusion Matrix Analysis
- **Strong Performance**: Hungry and Tired categories
- **Challenging**: Discomfort vs Burping (similar audio characteristics)
- **Overall**: Balanced performance across all categories

## 🔌 API Documentation

### BabyCryClassifier Class

```python
class BabyCryClassifier:
    def __init__(self, model_path: str, scaler_path: str)
    
    def predict(self, audio_path: str, return_probabilities: bool = False) -> dict
    def predict_batch(self, audio_paths: List[str]) -> List[dict]
    def get_feature_importance(self) -> dict
```

### AudioPreprocessor Class

```python
class AudioPreprocessor:
    def __init__(self, scaler_path: str)
    
    def preprocess(self, audio_path: str) -> np.ndarray
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray
    def validate_file(self, file_path: str) -> bool
```

## 🛠️ Project Structure

```
infant-cry-classification/
├── dataset ppdm2 asli/          # Original dataset
│   └── donateacry_corpus/
│       ├── belly_pain/
│       ├── burping/
│       ├── discomfort/
│       ├── hungry/
│       └── tired/
├── dataset_preprocessed/        # Processed dataset
│   ├── stage1/                 # Audio preprocessing
│   ├── stage2/                 # Audio augmentation
│   └── stage3/                 # Feature extraction
├── preprocess_config.py        # Configuration parameters
├── preprocess1.py              # Stage 1 preprocessing
├── preprocess2.py              # Stage 2 augmentation
├── preprocess3.py              # Stage 3 feature extraction
├── svm_python.py               # SVM training script
├── svm.ipynb                   # Jupyter notebook
├── deploy_preprocess.py        # Deployment preprocessing
├── deploy_function.py          # Model deployment wrapper
├── streamlit_fix.py            # Streamlit web app
├── svm_classes.py              # SVM implementation
├── setup.py                    # Automatic setup script
├── requirements.txt            # Python dependencies
├── best_svm_model.pkl         # Trained model
└── README.md                   # This file
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Format code
black .
flake8 .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DonateACry Corpus](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus) for the dataset
- [librosa](https://librosa.org/) for audio processing
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [Streamlit](https://streamlit.io/) for web deployment

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername]
- **LinkedIn**: [Your LinkedIn]

---

⭐ **Star this repository if you find it helpful!**
