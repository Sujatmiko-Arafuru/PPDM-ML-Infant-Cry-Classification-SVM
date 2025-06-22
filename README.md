#  Klasifikasi Tangisan Bayi Menggunakan Support Vector Machine (SVM)

##  Deskripsi Proyek

Proyek ini merupakan implementasi sistem klasifikasi tangisan bayi menggunakan algoritma Support Vector Machine (SVM) yang diimplementasikan dari awal (*from scratch*) dengan algoritma Sequential Minimal Optimization (SMO). Sistem ini dapat mengklasifikasikan tangisan bayi ke dalam 5 kategori berbeda berdasarkan analisis sinyal audio.

###  Kategori Tangisan Bayi

1. **Belly Pain** (Sakit Perut) - Tangisan karena ketidaknyamanan perut
2. **Burping** (Sendawa) - Tangisan karena perlu bersendawa
3. **Discomfort** (Ketidaknyamanan) - Tangisan karena ketidaknyamanan umum
4. **Hungry** (Lapar) - Tangisan karena kelaparan
5. **Tired** (Lelah) - Tangisan karena kelelahan/mengantuk

##  Arsitektur Sistem

```
Dataset Audio → Preprocessing → Feature Extraction → Model Training → Deployment
     ↓              ↓               ↓                ↓               ↓
  Raw WAV      Normalisasi     Time Domain      SVM Training    Streamlit App
   Files       Noise Filter    Frequency        (SMO + OvR)    Real-time
              Segmentasi      MFCC Features                    Prediction
```

##  Struktur Proyek

```
InfantCryClassification/
├── dataset ppdm2 asli/          # Dataset audio asli
│   └── donateacry_corpus/       # Dataset 
│       ├── belly_pain/          # Audio tangisan sakit perut
│       ├── burping/             # Audio tangisan sendawa
│       ├── discomfort/          # Audio tangisan tidak nyaman
│       ├── hungry/              # Audio tangisan lapar
│       └── tired/               # Audio tangisan lelah
├── dataset_preprocessed/        # Dataset hasil preprocessing
│   ├── stage1/                  # Tahap 1: Audio preprocessing
│   ├── stage2/                  # Tahap 2: Audio augmentation
│   └── stage3/                  # Tahap 3: Feature extraction
├── preprocess_config.py         # Konfigurasi parameter preprocessing
├── preprocess1.py               # Script preprocessing tahap 1
├── preprocess2.py               # Script preprocessing tahap 2
├── preprocess3.py               # Script preprocessing tahap 3
├── svm_python.py                # Script model SVM versi python
├── svm.ipynb                    # Jupyter notebook (media eksperimen)
├── deploy_preprocess.py         # Preprocessing untuk deployment
├── deploy_function.py           # Flow penerapan model pada deployment
├── streamlit_fix.py             # Aplikasi web Streamlit
├── svm_classes.py               # Bridge deployment streamlit (pengganti svm_python.py)
└── best_svm_model.pkl           # Model SVM terbaik (trained)
```

##  Pipeline Preprocessing

### Tahap 1: Audio Preprocessing (`preprocess1.py`)

**Tujuan**: Membersihkan dan menormalkan data audio mentah

**Proses**:
1. **Loading Audio**: Memuat file audio dengan librosa
2. **Normalisasi**: Menormalkan amplitudo ke rentang [-1, 1]
3. **Noise Reduction**: Filter Butterworth high-pass (cutoff: 100 Hz)
4. **Resampling**: Mengubah sample rate ke 22,050 Hz
5. **Segmentasi**: Membagi audio menjadi segmen 2 detik dengan padding

**Parameter Kunci**:
- Sample Rate Target: 22,050 Hz
- Durasi Segmen: 2.0 detik
- Filter Noise: Butterworth high-pass order 5

### Tahap 2: Audio Augmentation (`preprocess2.py`)

**Tujuan**: Meningkatkan variasi dataset untuk mengurangi overfitting

**Teknik Augmentasi**:
1. **Time Stretching**: Mengubah kecepatan audio (0.8x - 1.2x)
2. **Pitch Shifting**: Menggeser nada (-2 sampai +2 semitone)
3. **Add Noise**: Menambahkan noise Gaussian (faktor: 0.003-0.01)
4. **Time Shifting**: Menggeser waktu maksimal 0.5 detik
5. **Filtering**: Menerapkan filter low-pass, high-pass, band-pass

**Hasil**: Dataset bertambah ~5x lipat dari jumlah original

### Tahap 3: Feature Extraction (`preprocess3.py`)

**Tujuan**: Mengekstrak fitur numerik dari sinyal audio

**Fitur yang Diekstrak** (Total: 34 fitur):

#### A. Fitur Domain Waktu (4 fitur)
- **Zero Crossing Rate (ZCR)**: Mean & Std
- **Root Mean Square (RMS) Energy**: Mean & Std

#### B. Fitur Domain Frekuensi (4 fitur)  
- **Spectral Centroid**: Mean & Std (pusat massa spektrum)
- **Spectral Bandwidth**: Mean & Std (lebar spektrum)

#### C. Fitur MFCC (26 fitur)
- **Mel-Frequency Cepstral Coefficients**: 13 koefisien × 2 (mean & std)
- Parameter: 40 mel filters, fmin=0 Hz, fmax=8000 Hz

**Parameter Windowing**:
- Frame Length: 2048 samples
- Hop Length: 512 samples  
- Window: Hann window

##  Model Machine Learning

### Implementasi SVM From Scratch

#### 1. Kernel SVM dengan SMO Algorithm (`svm_python.py` atau `svm.ipynb`)

**Sequential Minimal Optimization (SMO)**:
- Algoritma optimasi untuk menyelesaikan dual problem SVM
- Mengoptimasi 2 Lagrange multipliers (α) secara bersamaan
- Implementasi heuristic untuk memilih pasangan α yang optimal

**Kernel yang Didukung**:
- **Linear Kernel**: K(x₁, x₂) = x₁ᵀx₂
- **RBF Kernel**: K(x₁, x₂) = exp(-γ||x₁ - x₂||²)

#### 2. One-vs-Rest (OvR) untuk Multiclass

**Strategi Multiclass**:
- Melatih 5 binary classifier (satu untuk setiap kelas)
- Setiap classifier membedakan satu kelas vs semua kelas lainnya
- Prediksi final: kelas dengan decision function tertinggi

**Implementasi Probabilitas**:
- Menggunakan Platt scaling pada decision function
- Normalisasi dengan softmax untuk mendapatkan probabilitas

### Training Process (`svm_python.py` atau `svm.ipynb`)

#### 1. Grid Search Hyperparameter
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.001, 0.01, 0.1, 1]
}
```

#### 2. Evaluasi Model
- **Training Set**: 70% data
- **Validation Set**: 15% data  
- **Test Set**: 15% data
- **Metrik**: Accuracy, Precision, Recall, F1-Score
- **Cross-Validation**: 5-fold stratified

#### 3. Model Selection
- Grid search dengan validasi silang
- Pemilihan hyperparameter terbaik berdasarkan validation accuracy
- Final evaluation pada test set yang tidak pernah dilihat

##  Deployment

### 1. Preprocessing untuk Deployment (`deploy_preprocess.py`)

**Class AudioPreprocessor**:
- Implementasi preprocessing yang konsisten dengan training
- Loading dan caching scaler untuk normalisasi fitur
- Validasi format file audio yang didukung
- Error handling untuk file audio yang corrupt

**Fitur Utama**:
- Preprocessing real-time untuk single audio file
- Ekstraksi fitur yang sama dengan training pipeline
- Standarisasi menggunakan scaler yang sudah dilatih

### 2. Model Wrapper (`deploy_function.py`)

**Class BabyCryClassifier**:
- Interface sederhana untuk prediksi
- Loading model dan scaler secara otomatis
- Batch prediction untuk multiple files
- Return probabilitas dan confidence score

**Penggunaan**:
```python
classifier = BabyCryClassifier(
    model_path="best_svm_model.pkl",
    scaler_path="dataset_preprocessed/stage3/scaler.joblib"
)

result = classifier.predict("audio_file.wav", return_probabilities=True)
```

### 3. Aplikasi Web Streamlit (`streamlit_fix.py`)

**Fitur Aplikasi**:
- **Upload Audio**: Support .wav dan .mp3
- **Audio Player**: Preview audio yang diupload
- **Real-time Prediction**: Klasifikasi tangisan bayi
- **Visualisasi**: Waveform, spektrogram, feature plot
- **Confidence Score**: Probabilitas untuk setiap kelas
- **Model Info**: Informasi detail tentang model

**Interface**:
- **Sidebar**: Upload file dan pengaturan
- **Main Panel**: Hasil prediksi dan visualisasi
- **Responsive Design**: Optimized untuk desktop dan mobile

**Menjalankan Aplikasi (PERHATIKAN INSTALASI DAN SETUP SEBELUM MENJALANKAN STREAMLIT)**:
```bash
streamlit run streamlit_fix.py
```

##  Performa Model

### Hasil Training
- **Best Hyperparameters**: 
  - C: 10
  - Kernel: RBF
  - Gamma: 0.1
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~88%
- **Test Accuracy**: ~85%

### Confusion Matrix
Model menunjukkan performa terbaik untuk kategori:
1. **Hungry** (Lapar) - Precision: 92%
2. **Tired** (Lelah) - Precision: 89%
3. **Belly Pain** (Sakit Perut) - Precision: 87%

Kategori yang lebih challenging:
- **Discomfort** vs **Burping** - sering tertukar karena karakteristik audio yang mirip

##  Instalasi dan Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd InfantCryClassification
```

### 2. Install Dependencies
  Tidak diperlukan untuk menginstall dependencies lainnya, cukup lakukan :
- Set interpreter virtual environtment pada selected interpreter ```.\.venv\Scripts\python.exe```
- Input command pada terminal IDE ```.venv\Scripts\Activate```

### 3. Setup Dataset
- Download dataset melalui dataset kaggle berikut https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus
- Ekstrak ke folder `dataset ppdm2 asli/donateacry_corpus/`
- Struktur folder harus sesuai dengan 5 kategori tangisan

### 4. Running Preprocessing
```bash
# Tahap 1: Audio preprocessing
python preprocess1.py

# Tahap 2: Audio augmentation  
python preprocess2.py

# Tahap 3: Feature extraction
python preprocess3.py
```

### 5. Training Model
```bash
# Training SVM versi pyhton
python svm_python.py

# Atau menggunakan Jupyter notebook
jupyter notebook svm.ipynb
```

### 6. Deployment
```bash
# Menjalankan aplikasi Streamlit
streamlit run streamlit_fix.py
```

##  Konfigurasi

### Parameter Preprocessing (`preprocess_config.py`)

**Audio Processing**:
- `TARGET_SAMPLE_RATE = 22050`
- `SEGMENT_DURATION = 2.0`
- `NOISE_CUTOFF_FREQ = 100`

**Feature Extraction**:
- `N_MFCC = 13`
- `FRAME_LENGTH = 2048`
- `HOP_LENGTH = 512`
- `N_MEL_FILTERS = 40`

**Model Training**:
- `TRAIN_SIZE = 0.7`
- `VAL_SIZE = 0.15`
- `TEST_SIZE = 0.15`
