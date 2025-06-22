"""
Konfigurasi Parameter Preprocessing untuk Deployment
Klasifikasi Tangisan Bayi menggunakan SVM

File ini berisi semua parameter yang digunakan dalam tahapan preprocessing
yang diperlukan untuk deployment model.
"""

import numpy as np

# =============================================================================
# TAHAP 1: AUDIO PREPROCESSING
# =============================================================================

# Parameter Audio Loading & Normalisasi
TARGET_SAMPLE_RATE = 22050  # Hz
NORMALIZATION_METHOD = "librosa.util.normalize"  # Normalisasi ke [-1, 1]

# Parameter Noise Reduction (Butterworth High-pass Filter)
NOISE_REDUCTION_ENABLED = True
NOISE_FILTER_TYPE = "butterworth"
NOISE_FILTER_MODE = "highpass"
NOISE_FILTER_ORDER = 5
NOISE_CUTOFF_FREQ = 100  # Hz

# Parameter Segmentasi Audio
SEGMENT_DURATION = 2.0  # detik
MIN_LAST_SEGMENT = 1.0  # detik
SEGMENT_PADDING_VALUE = 0.0

# =============================================================================
# TAHAP 2: AUGMENTASI AUDIO (Hanya untuk Training)
# =============================================================================

# Parameter Time Stretching
TIME_STRETCH_ENABLED = True
TIME_STRETCH_RATE_RANGE = (0.8, 1.2)

# Parameter Pitch Shifting
PITCH_SHIFT_ENABLED = True
PITCH_SHIFT_STEPS_RANGE = (-2, 2)

# Parameter Add Noise
ADD_NOISE_ENABLED = True
NOISE_FACTOR_RANGE = (0.003, 0.01)
NOISE_DISTRIBUTION = "normal"  # normal, uniform

# Parameter Time Shifting
TIME_SHIFT_ENABLED = True
TIME_SHIFT_MAX_SEC = 0.5

# Parameter Apply Filter
FILTER_ENABLED = True
FILTER_CUTOFF_RANGE = (1000, 4000)  # Hz
FILTER_ORDER = 4
FILTER_TYPES = ['lowpass', 'highpass', 'bandpass']

# =============================================================================
# TAHAP 3: EKSTRAKSI FITUR
# =============================================================================

# Parameter Framing & Windowing
FRAME_LENGTH = 2048
HOP_LENGTH = 512
WINDOW_TYPE = 'hann'  # hann, hamming, rectangular
N_FFT = FRAME_LENGTH

# Parameter MFCC (Domain Cepstral)
N_MFCC = 13
PRE_EMPHASIS_ENABLED = True
PRE_EMPHASIS_COEFF = 0.97
N_MEL_FILTERS = 40
MEL_FMIN = 0  # Hz
MEL_FMAX = 8000  # Hz
LOG_OFFSET = 1e-10  # Untuk menghindari log(0)

# Parameter Fitur Domain Waktu
# Zero Crossing Rate & RMS Energy
TIME_DOMAIN_FEATURES = ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std']

# Parameter Fitur Domain Frekuensi  
# Spectral Centroid & Spectral Bandwidth
FREQ_DOMAIN_FEATURES = ['centroid_mean', 'centroid_std', 'bandwidth_mean', 'bandwidth_std']

# Parameter Fitur MFCC
MFCC_FEATURES = [f'mfcc{i+1}_{stat}' for i in range(N_MFCC) for stat in ['mean', 'std']]

# Total Features
ALL_FEATURES = TIME_DOMAIN_FEATURES + FREQ_DOMAIN_FEATURES + MFCC_FEATURES
N_TOTAL_FEATURES = len(ALL_FEATURES)  # 4 + 4 + 26 = 34 fitur

# =============================================================================
# TAHAP 4: DATASET SPLIT & LABEL ENCODING
# =============================================================================

# Parameter Dataset Split
TRAIN_SIZE = 0.7   # 70%
VAL_SIZE = 0.15    # 15%
TEST_SIZE = 0.15   # 15%
RANDOM_STATE = 42
STRATIFY_ENABLED = True

# Parameter Label Encoding
LABEL_CLASSES = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
N_CLASSES = len(LABEL_CLASSES)
LABEL_ENCODER_TYPE = "LabelEncoder"

# =============================================================================
# TAHAP 5: STANDARISASI FITUR
# =============================================================================

# Parameter Standardization
SCALER_TYPE = "StandardScaler"
SCALER_FILE = "scaler.joblib"
STANDARDIZATION_ENABLED = True

# =============================================================================
# FILE PATHS & EXTENSIONS
# =============================================================================

# Audio File Extensions
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.flac', '.m4a']
DEFAULT_AUDIO_FORMAT = '.wav'

# Model Files
SCALER_PATH = "dataset_preprocessed/stage3/scaler.joblib"
DATASET_PATH = "dataset_preprocessed/stage3/dataset.npz"
BEST_MODEL_PATH = "best_svm_model.pkl"

# =============================================================================
# VALIDASI KONFIGURASI
# =============================================================================

def validate_config():
    """
    Validasi konfigurasi parameter preprocessing
    """
    errors = []
    
    # Validasi sample rate
    if TARGET_SAMPLE_RATE <= 0:
        errors.append("TARGET_SAMPLE_RATE harus > 0")
    
    # Validasi segmentasi
    if SEGMENT_DURATION <= 0:
        errors.append("SEGMENT_DURATION harus > 0")
    
    if MIN_LAST_SEGMENT <= 0 or MIN_LAST_SEGMENT > SEGMENT_DURATION:
        errors.append("MIN_LAST_SEGMENT harus antara 0 dan SEGMENT_DURATION")
    
    # Validasi framing
    if FRAME_LENGTH <= 0 or HOP_LENGTH <= 0:
        errors.append("FRAME_LENGTH dan HOP_LENGTH harus > 0")
    
    if HOP_LENGTH > FRAME_LENGTH:
        errors.append("HOP_LENGTH tidak boleh > FRAME_LENGTH")
    
    # Validasi MFCC
    if N_MFCC <= 0:
        errors.append("N_MFCC harus > 0")
    
    if N_MEL_FILTERS <= N_MFCC:
        errors.append("N_MEL_FILTERS harus > N_MFCC")
    
    # Validasi dataset split
    total_split = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if abs(total_split - 1.0) > 1e-6:
        errors.append(f"Total dataset split harus = 1.0, sekarang = {total_split}")
    
    # Validasi label classes
    if len(LABEL_CLASSES) != N_CLASSES:
        errors.append("Jumlah LABEL_CLASSES tidak sesuai dengan N_CLASSES")
    
    # Validasi total fitur
    expected_features = 4 + 4 + (N_MFCC * 2)  # time + freq + mfcc
    if N_TOTAL_FEATURES != expected_features:
        errors.append(f"N_TOTAL_FEATURES tidak sesuai. Expected: {expected_features}, Got: {N_TOTAL_FEATURES}")
    
    if errors:
        raise ValueError("Konfigurasi tidak valid:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_segment_length():
    """Menghitung panjang segmen dalam sampel"""
    return int(SEGMENT_DURATION * TARGET_SAMPLE_RATE)

def get_min_segment_length():
    """Menghitung panjang minimum segmen dalam sampel"""
    return int(MIN_LAST_SEGMENT * TARGET_SAMPLE_RATE)

def get_nyquist_frequency():
    """Menghitung frekuensi Nyquist"""
    return TARGET_SAMPLE_RATE // 2

def get_mel_frequency_range():
    """Mendapatkan range frekuensi Mel"""
    return (MEL_FMIN, min(MEL_FMAX, get_nyquist_frequency()))

def print_config_summary():
    """
    Menampilkan ringkasan konfigurasi preprocessing
    """
    print("=" * 80)
    print("RINGKASAN KONFIGURASI PREPROCESSING")
    print("=" * 80)
    
    print(f"\n PARAMETER AUDIO:")
    print(f"  - Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"  - Segment Duration: {SEGMENT_DURATION} detik ({get_segment_length()} sampel)")
    print(f"  - Min Last Segment: {MIN_LAST_SEGMENT} detik ({get_min_segment_length()} sampel)")
    
    print(f"\n PARAMETER EKSTRAKSI FITUR:")
    print(f"  - Frame Length: {FRAME_LENGTH}")
    print(f"  - Hop Length: {HOP_LENGTH}")
    print(f"  - Window Type: {WINDOW_TYPE}")
    print(f"  - N MFCC: {N_MFCC}")
    print(f"  - N Mel Filters: {N_MEL_FILTERS}")
    print(f"  - Mel Freq Range: {MEL_FMIN}-{MEL_FMAX} Hz")
    
    print(f"\n FITUR YANG DIEKSTRAK:")
    print(f"  - Domain Waktu: {len(TIME_DOMAIN_FEATURES)} fitur")
    print(f"  - Domain Frekuensi: {len(FREQ_DOMAIN_FEATURES)} fitur") 
    print(f"  - MFCC: {len(MFCC_FEATURES)} fitur")
    print(f"  - Total: {N_TOTAL_FEATURES} fitur")
    
    print(f"\n  LABEL & DATASET:")
    print(f"  - Jumlah Kelas: {N_CLASSES}")
    print(f"  - Kelas: {', '.join(LABEL_CLASSES)}")
    print(f"  - Split Dataset: Train {TRAIN_SIZE*100:.0f}% | Val {VAL_SIZE*100:.0f}% | Test {TEST_SIZE*100:.0f}%")
    
    print("=" * 80)

if __name__ == "__main__":
    # Validasi konfigurasi
    try:
        validate_config()
        print("✅ Konfigurasi valid!")
        print_config_summary()
    except ValueError as e:
        print(f"❌ Error dalam konfigurasi: {e}") 