"""
Utility Preprocessing untuk Deployment
Klasifikasi Tangisan Bayi menggunakan SVM

File ini berisi fungsi-fungsi preprocessing yang diperlukan untuk deployment model ke streamlit,
menggunakan parameter yang sudah didefinisikan dalam preprocessing_config.py
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.fft import fft
import joblib
import os
from preprocess_config import *

class AudioPreprocessor:
    """
    Class untuk preprocessing audio pada deployment
    """
    
    def __init__(self, scaler_path=None):
        """
        Inisialisasi preprocessor
        
        Parameters:
        - scaler_path: path ke file scaler (optional)
        """
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler berhasil dimuat dari {scaler_path}")
        elif scaler_path:
            print(f"⚠️  File scaler tidak ditemukan: {scaler_path}")
    
    def load_and_preprocess_audio(self, audio_path):
        """
        Memuat dan melakukan preprocessing audio sesuai dengan tahap 1
        
        Parameters:
        - audio_path: path ke file audio
        
        Returns:
        - audio: sinyal audio yang sudah dipreprocess
        - sr: sample rate
        """
        try:
            # 1. Memuat audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # 2. Normalisasi audio ke rentang [-1, 1]
            audio = librosa.util.normalize(audio)
            
            # 3. Noise reduction
            if NOISE_REDUCTION_ENABLED:
                audio = self._reduce_noise(audio, sr)
            
            # 4. Resampling jika sample rate berbeda dari target
            if sr != TARGET_SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
                sr = TARGET_SAMPLE_RATE
            
            # 5. Segmentasi (ambil segmen pertama jika audio lebih dari SEGMENT_DURATION)
            segment_length = get_segment_length()
            if len(audio) > segment_length:
                audio = audio[:segment_length]
            elif len(audio) < segment_length:
                # Padding jika audio terlalu pendek
                padded_audio = np.zeros(segment_length)
                padded_audio[:len(audio)] = audio
                audio = padded_audio
            
            return audio, sr
            
        except Exception as e:
            raise Exception(f"Error dalam preprocessing audio: {str(e)}")
    
    def _reduce_noise(self, audio, sr):
        """
        Mengurangi noise menggunakan high-pass filter
        
        Parameters:
        - audio: sinyal audio
        - sr: sample rate
        
        Returns:
        - filtered_audio: audio yang sudah difilter
        """
        # Butterworth high-pass filter
        nyquist = sr / 2
        normal_cutoff = NOISE_CUTOFF_FREQ / nyquist
        b, a = signal.butter(NOISE_FILTER_ORDER, normal_cutoff, btype=NOISE_FILTER_MODE)
        filtered_audio = signal.filtfilt(b, a, audio)
        return filtered_audio
    
    def extract_features(self, audio, sr):
        """
        Ekstraksi semua fitur dari audio
        
        Parameters:
        - audio: sinyal audio
        - sr: sample rate
        
        Returns:
        - features: array fitur yang sudah diekstrak
        """
        # Ekstrak fitur domain waktu
        time_features = self._extract_time_domain_features(audio)
        
        # Ekstrak fitur domain frekuensi
        freq_features = self._extract_frequency_domain_features(audio, sr)
        
        # Ekstrak fitur MFCC
        mfcc_features = self._extract_mfcc_features(audio, sr)
        
        # Gabungkan semua fitur sesuai urutan yang sudah didefinisikan
        all_features = []
        
        # Domain waktu (4 fitur)
        for feature_name in TIME_DOMAIN_FEATURES:
            all_features.append(time_features[feature_name])
        
        # Domain frekuensi (4 fitur)
        for feature_name in FREQ_DOMAIN_FEATURES:
            all_features.append(freq_features[feature_name])
        
        # MFCC (26 fitur)
        for feature_name in MFCC_FEATURES:
            all_features.append(mfcc_features[feature_name])
        
        return np.array(all_features)
    
    def _frame_audio(self, audio):
        """Membagi audio menjadi frame-frame"""
        num_frames = 1 + (len(audio) - FRAME_LENGTH) // HOP_LENGTH
        frames = np.zeros((num_frames, FRAME_LENGTH))
        
        for i in range(num_frames):
            start = i * HOP_LENGTH
            end = start + FRAME_LENGTH
            if end <= len(audio):
                frames[i] = audio[start:end]
        
        return frames
    
    def _apply_window(self, frames):
        """Menerapkan window function"""
        if WINDOW_TYPE == 'hann':
            window = np.hanning(frames.shape[1])
        elif WINDOW_TYPE == 'hamming':
            window = np.hamming(frames.shape[1])
        else:
            window = np.ones(frames.shape[1])
        
        return frames * window
    
    def _extract_time_domain_features(self, audio):
        """Ekstraksi fitur domain waktu"""
        frames = self._frame_audio(audio)
        
        # Zero Crossing Rate
        signs = np.sign(frames)
        signs[signs == 0] = -1
        sign_changes = np.abs(np.diff(signs, axis=1))
        zcr = np.sum(sign_changes, axis=1) / (2 * (frames.shape[1] - 1))
        
        # RMS Energy
        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        
        return {
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms)
        }
    
    def _extract_frequency_domain_features(self, audio, sr):
        """Ekstraksi fitur domain frekuensi"""
        frames = self._frame_audio(audio)
        windowed_frames = self._apply_window(frames)
        
        # FFT
        spectrum = fft(windowed_frames)
        magnitude = np.abs(spectrum[:, :FRAME_LENGTH//2 + 1])
        
        # Frekuensi
        freqs = np.linspace(0, sr/2, magnitude.shape[1])
        
        # Spectral Centroid
        magnitude_sum = np.sum(magnitude, axis=1, keepdims=True)
        normalized_magnitude = magnitude / (magnitude_sum + 1e-10)
        centroid = np.sum(normalized_magnitude * freqs, axis=1)
        
        # Spectral Bandwidth
        centroid_reshaped = centroid.reshape(-1, 1)
        deviation = np.square(freqs - centroid_reshaped)
        bandwidth = np.sqrt(np.sum(deviation * normalized_magnitude, axis=1))
        
        return {
            'centroid_mean': np.mean(centroid),
            'centroid_std': np.std(centroid),
            'bandwidth_mean': np.mean(bandwidth),
            'bandwidth_std': np.std(bandwidth)
        }
    
    def _extract_mfcc_features(self, audio, sr):
        """Ekstraksi fitur MFCC"""
        # Pre-emphasis
        if PRE_EMPHASIS_ENABLED:
            emphasized_signal = np.append(audio[0], audio[1:] - PRE_EMPHASIS_COEFF * audio[:-1])
        else:
            emphasized_signal = audio
        
        # Framing dan windowing
        frames = self._frame_audio(emphasized_signal)
        windowed_frames = self._apply_window(frames)
        
        # FFT dan Power Spectrum
        spectrum = fft(windowed_frames)
        magnitude = np.abs(spectrum[:, :FRAME_LENGTH//2 + 1])
        power_spectrum = (magnitude ** 2) / FRAME_LENGTH
        
        # Mel Filterbank
        filterbank = self._create_mel_filterbank(sr)
        mel_spectrum = np.dot(power_spectrum, filterbank.T)
        
        # Log Mel Spectrum
        log_mel_spectrum = np.log(mel_spectrum + LOG_OFFSET)
        
        # DCT
        mfcc = self._apply_dct(log_mel_spectrum)
        
        # Hitung statistik
        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_std = np.std(mfcc, axis=0)
        
        # Buat dictionary fitur
        mfcc_features = {}
        for i in range(N_MFCC):
            mfcc_features[f'mfcc{i+1}_mean'] = mfcc_mean[i]
            mfcc_features[f'mfcc{i+1}_std'] = mfcc_std[i]
        
        return mfcc_features
    
    def _create_mel_filterbank(self, sr):
        """Membuat mel filterbank"""
        # Konversi ke skala mel
        mel_fmin = 2595 * np.log10(1 + MEL_FMIN / 700)
        mel_fmax = 2595 * np.log10(1 + MEL_FMAX / 700)
        
        # Titik-titik mel
        mel_points = np.linspace(mel_fmin, mel_fmax, N_MEL_FILTERS + 2)
        
        # Konversi kembali ke Hz
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Konversi ke bin FFT
        bin_points = np.floor((FRAME_LENGTH + 1) * hz_points / sr).astype(int)
        
        # Buat filterbank
        filterbank = np.zeros((N_MEL_FILTERS, FRAME_LENGTH // 2 + 1))
        
        for i in range(N_MEL_FILTERS):
            start, center, end = bin_points[i:i+3]
            
            # Filter segitiga kiri (naik)
            for j in range(start, center):
                if center != start:
                    filterbank[i, j] = (j - start) / (center - start)
            
            # Filter segitiga kanan (turun)
            for j in range(center, end):
                if end != center:
                    filterbank[i, j] = (end - j) / (end - center)
        
        return filterbank
    
    def _apply_dct(self, log_mel_spectrum):
        """Menerapkan DCT untuk mendapatkan MFCC"""
        n_filters = log_mel_spectrum.shape[1]
        
        # Buat matriks DCT
        dct_matrix = np.zeros((N_MFCC, n_filters))
        for i in range(N_MFCC):
            for j in range(n_filters):
                dct_matrix[i, j] = np.cos(np.pi * i * (j + 0.5) / n_filters)
        
        # Terapkan DCT
        mfcc = np.dot(log_mel_spectrum, dct_matrix.T)
        
        return mfcc
    
    def standardize_features(self, features):
        """
        Standardisasi fitur menggunakan scaler yang sudah dilatih
        
        Parameters:
        - features: array fitur
        
        Returns:
        - standardized_features: fitur yang sudah distandarisasi
        """
        if self.scaler is None:
            raise Exception("Scaler belum dimuat. Gunakan load_scaler() terlebih dahulu.")
        
        # Reshape untuk scaler (perlu 2D array)
        features_2d = features.reshape(1, -1)
        standardized = self.scaler.transform(features_2d)
        
        return standardized.flatten()
    
    def preprocess_for_prediction(self, audio_path):
        """
        Pipeline lengkap preprocessing untuk prediksi
        
        Parameters:
        - audio_path: path ke file audio
        
        Returns:
        - features: fitur yang siap untuk prediksi
        """
        # 1. Load dan preprocess audio
        audio, sr = self.load_and_preprocess_audio(audio_path)
        
        # 2. Ekstrak fitur
        features = self.extract_features(audio, sr)
        
        # 3. Standardisasi fitur
        if self.scaler is not None:
            features = self.standardize_features(features)
        
        return features
    
    def load_scaler(self, scaler_path):
        """
        Memuat scaler dari file
        
        Parameters:
        - scaler_path: path ke file scaler
        """
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler berhasil dimuat dari {scaler_path}")
        else:
            raise FileNotFoundError(f"File scaler tidak ditemukan: {scaler_path}")

def get_label_name(label_index):
    """
    Mengkonversi index label ke nama label
    
    Parameters:
    - label_index: index label (0-4)
    
    Returns:
    - label_name: nama label
    """
    if 0 <= label_index < len(LABEL_CLASSES):
        return LABEL_CLASSES[label_index]
    else:
        return f"Unknown_{label_index}"

def validate_audio_file(audio_path):
    """
    Validasi file audio
    
    Parameters:
    - audio_path: path ke file audio
    
    Returns:
    - is_valid: True jika valid, False jika tidak
    - message: pesan error jika tidak valid
    """
    if not os.path.exists(audio_path):
        return False, "File tidak ditemukan"
    
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext not in SUPPORTED_AUDIO_FORMATS:
        return False, f"Format file tidak didukung. Didukung: {SUPPORTED_AUDIO_FORMATS}"
    
    try:
        # Coba load audio untuk memastikan file valid
        audio, sr = librosa.load(audio_path, sr=None, duration=0.1)  # Load 0.1 detik saja untuk test
        return True, "File audio valid"
    except Exception as e:
        return False, f"Error membaca file audio: {str(e)}"


if __name__ == "__main__":
    
    print("🧪 Testing Deployment Preprocessing...")
    print("=" * 50)
    
    # Validasi konfigurasi
    try:
        validate_config()
        print("✅ Konfigurasi valid!")
    except ValueError as e:
        print(f"❌ Error konfigurasi: {e}")
        exit(1)
    
    # Inisialisasi preprocessor
    preprocessor = AudioPreprocessor()
    
    # Coba load scaler jika ada
    if os.path.exists(SCALER_PATH):
        preprocessor.load_scaler(SCALER_PATH)
    else:
        print(f"⚠️  Scaler tidak ditemukan di {SCALER_PATH}")
    
    print(f"\n📊 Parameter yang akan digunakan:")
    print(f"  - Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"  - Segment Duration: {SEGMENT_DURATION} detik")
    print(f"  - Total Features: {N_TOTAL_FEATURES}")
    print(f"  - Label Classes: {LABEL_CLASSES}")
    
    print("\n✅ Deployment preprocessing siap digunakan!") 