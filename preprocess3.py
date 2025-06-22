import os
import numpy as np
import pandas as pd
import soundfile as sf
from glob import glob
from tqdm import tqdm
from scipy.fft import fft
from scipy import signal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import time
import joblib
import datetime

# Fungsi untuk menampilkan status tahapan
def print_step(step_name, start=True):
    """
    Menampilkan status tahapan dengan format yang jelas
    
    Parameters:
    - step_name: nama tahapan
    - start: True jika tahapan dimulai, False jika selesai
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    if start:
        print(f"\n{'=' * 80}")
        print(f"[{timestamp}] MEMULAI: {step_name}")
        print(f"{'-' * 80}")
        return time.time()
    else:
        end_time = time.time()
        duration = end_time - start
        print(f"{'-' * 80}")
        print(f"[{timestamp}] SELESAI: {step_name} (Waktu: {duration:.2f} detik)")
        print(f"{'=' * 80}\n")
        return end_time

# Fungsi-fungsi untuk ekstraksi fitur domain waktu
def read_audio(file_path):
    """
    Membaca file audio menggunakan soundfile
    
    Parameters:
    - file_path: path ke file audio
    
    Returns:
    - audio: sinyal audio
    - sr: sample rate
    """
    audio, sr = sf.read(file_path)
    # Jika audio stereo, ambil channel pertama saja
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    return audio, sr

def frame_audio(audio, frame_length=2048, hop_length=512):
    """
    Membagi audio menjadi frame-frame
    
    Parameters:
    - audio: sinyal audio
    - frame_length: panjang frame
    - hop_length: jarak antar frame
    
    Returns:
    - frames: array frame audio
    """
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.zeros((num_frames, frame_length))
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[i] = audio[start:end]
    
    return frames

def compute_zero_crossing_rate(frames):
    """
    Menghitung zero crossing rate
    
    Parameters:
    - frames: array frame audio
    
    Returns:
    - zcr: zero crossing rate
    """
    # Hitung sign changes
    signs = np.sign(frames)
    signs[signs == 0] = -1  # Menganggap 0 sebagai negatif
    
    # Hitung perubahan tanda
    sign_changes = np.abs(np.diff(signs, axis=1))
    
    # Hitung ZCR
    zcr = np.sum(sign_changes, axis=1) / (2 * (frames.shape[1] - 1))
    
    return zcr

def compute_rms_energy(frames):
    """
    Menghitung RMS energy
    
    Parameters:
    - frames: array frame audio
    
    Returns:
    - rms: RMS energy
    """
    # Hitung RMS
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    
    return rms

def extract_time_domain_features(audio, frame_length=2048, hop_length=512):
    """
    Ekstraksi fitur domain waktu
    
    Parameters:
    - audio: sinyal audio
    - frame_length: panjang frame
    - hop_length: jarak antar frame
    
    Returns:
    - time_features: dictionary fitur domain waktu
    """
    # Framing
    frames = frame_audio(audio, frame_length, hop_length)
    
    # Ekstrak ZCR
    zcr = compute_zero_crossing_rate(frames)
    
    # Ekstrak RMS Energy
    rms = compute_rms_energy(frames)
    
    # Hitung statistik
    time_features = {
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr),
        'rms_mean': np.mean(rms),
        'rms_std': np.std(rms)
    }
    
    return time_features

# Fungsi-fungsi untuk ekstraksi fitur domain frekuensi
def apply_window(frames, window_type='hann'):
    """
    Menerapkan window function pada frame
    
    Parameters:
    - frames: array frame audio
    - window_type: jenis window
    
    Returns:
    - windowed_frames: frame audio yang sudah di-window
    """
    if window_type == 'hann':
        window = np.hanning(frames.shape[1])
    elif window_type == 'hamming':
        window = np.hamming(frames.shape[1])
    else:
        window = np.ones(frames.shape[1])
    
    return frames * window

def compute_fft(frames):
    """
    Menghitung FFT dari frame
    
    Parameters:
    - frames: array frame audio
    
    Returns:
    - magnitude: magnitude spektrum
    """
    # Hitung FFT
    spectrum = fft(frames)
    
    # Ambil setengah pertama saja (karena simetris)
    n_fft = frames.shape[1]
    magnitude = np.abs(spectrum[:, :n_fft//2 + 1])
    
    return magnitude

def compute_spectral_centroid(magnitude, freqs):
    """
    Menghitung spectral centroid
    
    Parameters:
    - magnitude: magnitude spektrum
    - freqs: array frekuensi
    
    Returns:
    - centroid: spectral centroid
    """
    # Normalisasi magnitude
    magnitude_sum = np.sum(magnitude, axis=1, keepdims=True)
    normalized_magnitude = magnitude / (magnitude_sum + 1e-10)
    
    # Hitung centroid
    centroid = np.sum(normalized_magnitude * freqs, axis=1)
    
    return centroid

def compute_spectral_bandwidth(magnitude, freqs, centroid):
    """
    Menghitung spectral bandwidth
    
    Parameters:
    - magnitude: magnitude spektrum
    - freqs: array frekuensi
    - centroid: spectral centroid
    
    Returns:
    - bandwidth: spectral bandwidth
    """
    # Normalisasi magnitude
    magnitude_sum = np.sum(magnitude, axis=1, keepdims=True)
    normalized_magnitude = magnitude / (magnitude_sum + 1e-10)
    
    # Reshape centroid untuk broadcasting
    centroid_reshaped = centroid.reshape(-1, 1)
    
    # Hitung bandwidth
    deviation = np.square(freqs - centroid_reshaped)
    bandwidth = np.sqrt(np.sum(deviation * normalized_magnitude, axis=1))
    
    return bandwidth

def extract_frequency_domain_features(audio, sr, frame_length=2048, hop_length=512):
    """
    Ekstraksi fitur domain frekuensi
    
    Parameters:
    - audio: sinyal audio
    - sr: sample rate
    - frame_length: panjang frame
    - hop_length: jarak antar frame
    
    Returns:
    - freq_features: dictionary fitur domain frekuensi
    """
    # Framing dan windowing
    frames = frame_audio(audio, frame_length, hop_length)
    windowed_frames = apply_window(frames)
    
    # Hitung FFT
    magnitude = compute_fft(windowed_frames)
    
    # Hitung frekuensi untuk setiap bin FFT
    freqs = np.linspace(0, sr/2, magnitude.shape[1])
    
    # Ekstrak spectral centroid
    centroid = compute_spectral_centroid(magnitude, freqs)
    
    # Ekstrak spectral bandwidth
    bandwidth = compute_spectral_bandwidth(magnitude, freqs, centroid)
    
    # Hitung statistik
    freq_features = {
        'centroid_mean': np.mean(centroid),
        'centroid_std': np.std(centroid),
        'bandwidth_mean': np.mean(bandwidth),
        'bandwidth_std': np.std(bandwidth)
    }
    
    return freq_features

# Fungsi-fungsi untuk ekstraksi MFCC (domain cepstral)
def pre_emphasis(signal, alpha=0.97):
    """
    Menerapkan pre-emphasis filter
    
    Parameters:
    - signal: sinyal audio
    - alpha: koefisien filter
    
    Returns:
    - emphasized_signal: sinyal yang sudah di-pre-emphasis
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def compute_power_spectrum(magnitude):
    """
    Menghitung power spectrum dari magnitude
    
    Parameters:
    - magnitude: magnitude spektrum
    
    Returns:
    - power: power spektrum
    """
    return (magnitude ** 2) / magnitude.shape[1]

def hz_to_mel(hz):
    """
    Konversi frekuensi dari Hz ke skala Mel
    
    Parameters:
    - hz: frekuensi dalam Hz
    
    Returns:
    - mel: frekuensi dalam skala Mel
    """
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    """
    Konversi frekuensi dari skala Mel ke Hz
    
    Parameters:
    - mel: frekuensi dalam skala Mel
    
    Returns:
    - hz: frekuensi dalam Hz
    """
    return 700 * (10 ** (mel / 2595) - 1)

def create_mel_filterbank(n_filters=40, fmin=0, fmax=8000, n_fft=2048, sr=22050):
    """
    Membuat mel filterbank
    
    Parameters:
    - n_filters: jumlah filter
    - fmin: frekuensi minimum (Hz)
    - fmax: frekuensi maksimum (Hz)
    - n_fft: ukuran FFT
    - sr: sample rate
    
    Returns:
    - filterbank: mel filterbank
    """
    # Konversi fmin dan fmax ke skala mel
    mel_fmin = hz_to_mel(fmin)
    mel_fmax = hz_to_mel(fmax)
    
    # Buat titik-titik mel yang berjarak sama
    mel_points = np.linspace(mel_fmin, mel_fmax, n_filters + 2)
    
    # Konversi kembali ke Hz
    hz_points = mel_to_hz(mel_points)
    
    # Konversi ke bin FFT
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Buat filterbank
    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    
    for i in range(n_filters):
        start, center, end = bin_points[i:i+3]
        
        # Filter segitiga kiri (naik)
        for j in range(start, center):
            filterbank[i, j] = (j - start) / (center - start)
        
        # Filter segitiga kanan (turun)
        for j in range(center, end):
            filterbank[i, j] = (end - j) / (end - center)
    
    return filterbank

def apply_dct(log_mel_spectrum, n_mfcc=13):
    """
    Menerapkan Discrete Cosine Transform (DCT) untuk mendapatkan MFCC
    
    Parameters:
    - log_mel_spectrum: log mel spectrum
    - n_mfcc: jumlah koefisien MFCC
    
    Returns:
    - mfcc: koefisien MFCC
    """
    n_filters = log_mel_spectrum.shape[1]
    
    # Buat matriks DCT
    dct_matrix = np.zeros((n_mfcc, n_filters))
    for i in range(n_mfcc):
        for j in range(n_filters):
            dct_matrix[i, j] = np.cos(np.pi * i * (j + 0.5) / n_filters)
    
    # Terapkan DCT
    mfcc = np.dot(log_mel_spectrum, dct_matrix.T)
    
    return mfcc

def extract_mfcc(audio, sr, n_mfcc=13, frame_length=2048, hop_length=512, pre_emphasis_coeff=0.97):
    """
    Ekstraksi MFCC dari sinyal audio
    
    Parameters:
    - audio: sinyal audio
    - sr: sample rate
    - n_mfcc: jumlah koefisien MFCC
    - frame_length: panjang frame
    - hop_length: jarak antar frame
    - pre_emphasis_coeff: koefisien pre-emphasis
    
    Returns:
    - mfcc: koefisien MFCC
    """
    # Pre-emphasis
    emphasized_signal = pre_emphasis(audio, alpha=pre_emphasis_coeff)
    
    # Framing
    frames = frame_audio(emphasized_signal, frame_length, hop_length)
    
    # Windowing
    windowed_frames = apply_window(frames)
    
    # FFT dan Power Spectrum
    magnitude = compute_fft(windowed_frames)
    power_spectrum = compute_power_spectrum(magnitude)
    
    # Mel Filterbank
    filterbank = create_mel_filterbank(n_filters=40, n_fft=frame_length, sr=sr)
    mel_spectrum = np.dot(power_spectrum, filterbank.T)
    
    # Log Mel Spectrum
    log_mel_spectrum = np.log(mel_spectrum + 1e-10)
    
    # DCT
    mfcc = apply_dct(log_mel_spectrum, n_mfcc=n_mfcc)
    
    return mfcc

def extract_mfcc_features(audio, sr, n_mfcc=13):
    """
    Ekstraksi fitur MFCC
    
    Parameters:
    - audio: sinyal audio
    - sr: sample rate
    - n_mfcc: jumlah koefisien MFCC
    
    Returns:
    - mfcc_features: dictionary fitur MFCC
    """
    # Ekstrak MFCC
    mfcc = extract_mfcc(audio, sr, n_mfcc=n_mfcc)
    
    # Hitung statistik untuk setiap koefisien
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    
    # Buat dictionary fitur
    mfcc_features = {}
    for i in range(n_mfcc):
        mfcc_features[f'mfcc{i+1}_mean'] = mfcc_mean[i]
        mfcc_features[f'mfcc{i+1}_std'] = mfcc_std[i]
    
    return mfcc_features

def extract_all_features(audio_path):
    """
    Ekstraksi semua fitur dari file audio
    
    Parameters:
    - audio_path: path ke file audio
    
    Returns:
    - features: dictionary semua fitur
    """
    # Baca audio
    audio, sr = read_audio(audio_path)
    
    # Ekstrak fitur domain waktu
    time_features = extract_time_domain_features(audio)
    
    # Ekstrak fitur domain frekuensi
    freq_features = extract_frequency_domain_features(audio, sr)
    
    # Ekstrak fitur MFCC
    mfcc_features = extract_mfcc_features(audio, sr)
    
    # Gabungkan semua fitur
    features = {**time_features, **freq_features, **mfcc_features}
    
    return features

def process_dataset(input_dir, output_dir):
    """
    Memproses dataset untuk ekstraksi fitur, label encoding, split dataset, dan standarisasi
    
    Parameters:
    - input_dir: direktori dataset
    - output_dir: direktori output
    
    Returns:
    - None
    """
    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Inisialisasi list untuk menyimpan fitur dan label
    features_list = []
    labels = []
    file_paths = []
    
    # Tahap 1: Ekstraksi Fitur
    start_time = print_step("Tahap 1: Ekstraksi Fitur")
    
    # Iterasi melalui setiap kategori
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        
        if os.path.isdir(category_path):
            # Iterasi melalui setiap file audio dalam kategori
            audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            
            print(f"  Mengekstrak fitur dari {len(audio_files)} file dalam kategori {category}...")
            
            for audio_file in tqdm(audio_files, desc=f"  {category}"):
                audio_path = os.path.join(category_path, audio_file)
                
                try:
                    # Ekstrak semua fitur
                    features = extract_all_features(audio_path)
                    
                    # Tambahkan ke list
                    features_list.append(features)
                    labels.append(category)
                    file_paths.append(audio_path)
                except Exception as e:
                    print(f"  Error mengekstrak fitur dari {audio_path}: {str(e)}")
    
    # Konversi list fitur menjadi DataFrame
    print("  Mengkonversi hasil ekstraksi fitur menjadi DataFrame...")
    features_df = pd.DataFrame(features_list)
    features_df['label'] = labels
    features_df['file_path'] = file_paths
    
    print_step("Tahap 1: Ekstraksi Fitur", start=False)
    
    # Tahap 2: Label Encoding
    start_time = print_step("Tahap 2: Label Encoding")
    
    # Label encoding
    le = LabelEncoder()
    features_df['label_encoded'] = le.fit_transform(features_df['label'])
    
    # Simpan mapping label
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    print("  Label mapping:")
    for i, label in label_mapping.items():
        print(f"    {i}: {label}")
    
    print_step("Tahap 2: Label Encoding", start=False)
    
    # Tahap 3: Split Dataset
    start_time = print_step("Tahap 3: Split Dataset")
    
    # Pisahkan fitur dan label
    X = features_df.drop(['label', 'label_encoded', 'file_path'], axis=1)
    y = features_df['label_encoded']
    
    # Simpan nama fitur
    feature_names = X.columns.tolist()
    print(f"  Jumlah fitur: {len(feature_names)}")
    
    # Split dataset menjadi 3 bagian: training (70%), validation (15%), testing (15%)
    # Pertama split menjadi training (70%) dan sementara (30%)
    X_train, X_temp, y_train, y_temp, train_paths, temp_paths = train_test_split(
        X, y, features_df['file_path'], test_size=0.3, random_state=42, stratify=y
    )
    
    # Kemudian split bagian sementara (30%) menjadi validation (15%) dan testing (15%)
    X_val, X_test, y_val, y_test, val_paths, test_paths = train_test_split(
        X_temp, y_temp, temp_paths, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"  Jumlah data training: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Jumlah data validation: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Jumlah data testing: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    print_step("Tahap 3: Split Dataset", start=False)
    
    # Tahap 4: Standarisasi Fitur
    start_time = print_step("Tahap 4: Standarisasi Fitur")
    
    # Standarisasi fitur (fit hanya pada data training, transform pada semua dataset)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("  Statistik sebelum standarisasi (data training):")
    print(f"    Mean: min={X_train.mean().min():.4f}, max={X_train.mean().max():.4f}")
    print(f"    Std Dev: min={X_train.std().min():.4f}, max={X_train.std().max():.4f}")
    
    print("  Statistik setelah standarisasi (data training):")
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    print(f"    Mean: min={X_train_scaled_df.mean().min():.4f}, max={X_train_scaled_df.mean().max():.4f}")
    print(f"    Std Dev: min={X_train_scaled_df.std().min():.4f}, max={X_train_scaled_df.std().max():.4f}")
    
    print_step("Tahap 4: Standarisasi Fitur", start=False)
    
    # Tahap 5: Simpan Dataset
    start_time = print_step("Tahap 5: Simpan Dataset")
    
    # Simpan dataset
    dataset_path = os.path.join(output_dir, 'dataset.npz')
    np.savez(dataset_path,
             X_train=X_train_scaled,
             X_val=X_val_scaled,
             X_test=X_test_scaled,
             y_train=y_train,
             y_val=y_val,
             y_test=y_test,
             feature_names=feature_names,
             label_mapping=[label_mapping[i] for i in range(len(label_mapping))])
    
    # Simpan DataFrame fitur untuk referensi
    features_path = os.path.join(output_dir, 'features.csv')
    features_df.to_csv(features_path, index=False)
    
    # Simpan scaler
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Simpan paths untuk referensi
    paths_df = pd.DataFrame({
        'file_path': np.concatenate([train_paths, val_paths, test_paths]),
        'split': np.concatenate([
            np.full(len(train_paths), 'train'),
            np.full(len(val_paths), 'validation'),
            np.full(len(test_paths), 'test')
        ]),
        'label': np.concatenate([
            features_df.loc[features_df['file_path'].isin(train_paths), 'label'].values,
            features_df.loc[features_df['file_path'].isin(val_paths), 'label'].values,
            features_df.loc[features_df['file_path'].isin(test_paths), 'label'].values
        ])
    })
    paths_path = os.path.join(output_dir, 'file_paths.csv')
    paths_df.to_csv(paths_path, index=False)
    
    print(f"  Dataset disimpan di {dataset_path}")
    print(f"  Features disimpan di {features_path}")
    print(f"  Scaler disimpan di {scaler_path}")
    print(f"  File paths disimpan di {paths_path}")
    
    print("\n  Dataset berisi:")
    print(f"    - X_train: {X_train_scaled.shape}")
    print(f"    - X_val: {X_val_scaled.shape}")
    print(f"    - X_test: {X_test_scaled.shape}")
    print(f"    - y_train: {y_train.shape}")
    print(f"    - y_val: {y_val.shape}")
    print(f"    - y_test: {y_test.shape}")
    
    print_step("Tahap 5: Simpan Dataset", start=False)
    
    print("\nPemrosesan dataset selesai!")
    print(f"Semua hasil disimpan di {output_dir}")
    print(f"Dataset telah dibagi menjadi 3 bagian:")
    print(f"  - Training: {X_train.shape[0]} sampel ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Validation: {X_val.shape[0]} sampel ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Testing: {X_test.shape[0]} sampel ({X_test.shape[0]/len(X)*100:.1f}%)")

if __name__ == "__main__":
    # Direktori input dan output
    input_directory = "dataset_preprocessed/stage2"
    output_directory = "dataset_preprocessed/stage3"
    
    # Proses dataset
    process_dataset(input_directory, output_directory) 