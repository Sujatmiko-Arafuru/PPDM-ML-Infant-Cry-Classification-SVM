import os
import numpy as np
import librosa
import soundfile as sf
from glob import glob
from tqdm import tqdm
import random
from scipy import signal

def time_stretching(y, sr, rate_range=(0.8, 1.2)):
    """Mempercepat atau memperlambat audio"""
    rate = np.random.uniform(rate_range[0], rate_range[1])
    return librosa.effects.time_stretch(y, rate=rate)

def pitch_shifting(y, sr, n_steps_range=(-2, 2)):
    """Mengubah nada audio"""
    n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def add_noise(y, sr, noise_factor_range=(0.003, 0.01)):
    """Menambahkan noise acak dengan intensitas rendah"""
    noise_factor = np.random.uniform(noise_factor_range[0], noise_factor_range[1])
    noise = np.random.normal(0, noise_factor, len(y))
    return y + noise

def time_shifting(y, sr, shift_max_sec=0.5):
    """Menggeser audio dalam domain waktu"""
    shift_max = int(sr * shift_max_sec)
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(y, shift)

def apply_filter(y, sr, filter_type='lowpass'):
    """Menerapkan filter frekuensi"""
    # Frekuensi cutoff acak antara 1000-4000 Hz
    cutoff = np.random.uniform(1000, 4000)
    # Normalisasi frekuensi cutoff
    nyquist = sr // 2
    normal_cutoff = cutoff / nyquist
    
    # Desain filter
    if filter_type == 'lowpass':
        b, a = signal.butter(4, normal_cutoff, btype='lowpass')
    elif filter_type == 'highpass':
        b, a = signal.butter(4, normal_cutoff, btype='highpass')
    else:   
        b, a = signal.butter(4, [normal_cutoff*0.5, normal_cutoff], btype='bandpass')
    
    # Terapkan filter
    return signal.filtfilt(b, a, y)

def augment_audio(y, sr):
    """Melakukan augmentasi audio dengan berbagai teknik"""
    augmentation_techniques = [
        lambda y, sr: time_stretching(y, sr),
        lambda y, sr: pitch_shifting(y, sr),
        lambda y, sr: add_noise(y, sr),
        lambda y, sr: time_shifting(y, sr),
        lambda y, sr: apply_filter(y, sr)
    ]
    
    # Pilih teknik augmentasi secara acak
    technique = random.choice(augmentation_techniques)
    return technique(y, sr)

def preprocess_stage2(input_dir, output_dir, hungry_samples=400):
    """
    Tahap preprocessing kedua:
    1. Augmentasi data untuk kategori minoritas
    2. Undersampling untuk kategori mayoritas (hungry)
    
    Parameters:
    - input_dir: direktori dataset hasil preprocessing tahap 1
    - output_dir: direktori output untuk data hasil preprocessing tahap 2
    - hungry_samples: jumlah sampel yang diambil dari kategori hungry
    """
    # Membuat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Mendapatkan semua kategori (folder) dalam dataset
    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Memproses setiap kategori
    for category in categories:
        category_input_path = os.path.join(input_dir, category)
        category_output_path = os.path.join(output_dir, category)
        
        # Membuat direktori output untuk kategori
        os.makedirs(category_output_path, exist_ok=True)
        
        # Mendapatkan semua file audio dalam kategori
        audio_files = glob(os.path.join(category_input_path, "*.wav"))
        
        print(f"Memproses kategori {category} dengan {len(audio_files)} file...")
        
        # Jika kategori adalah hungry, lakukan undersampling
        if category == "hungry":
            # Jika jumlah file lebih banyak dari hungry_samples, lakukan undersampling
            if len(audio_files) > hungry_samples:
                # Pilih sampel secara acak
                audio_files = random.sample(audio_files, hungry_samples)
                print(f"  Undersampling: mengambil {hungry_samples} file dari kategori hungry")
            
            # Salin file yang terpilih ke direktori output
            for audio_file in tqdm(audio_files, desc=f"Copying {category}"):
                file_name = os.path.basename(audio_file)
                output_path = os.path.join(category_output_path, file_name)
                
                # Baca dan tulis kembali file audio
                y, sr = librosa.load(audio_file, sr=None)
                sf.write(output_path, y, sr)
        
        # Untuk kategori minoritas, lakukan augmentasi
        else:
            # Salin semua file asli
            for audio_file in tqdm(audio_files, desc=f"Copying original {category}"):
                file_name = os.path.basename(audio_file)
                output_path = os.path.join(category_output_path, file_name)
                
                # Baca dan tulis kembali file audio
                y, sr = librosa.load(audio_file, sr=None)
                sf.write(output_path, y, sr)
            
            # Hitung berapa banyak augmentasi yang diperlukan
            target_count = 400  # Target jumlah file per kategori
            augmentation_count = max(0, target_count - len(audio_files))
            
            print(f"  Augmentasi: membuat {augmentation_count} file baru untuk kategori {category}")
            
            # Lakukan augmentasi
            for i in tqdm(range(augmentation_count), desc=f"Augmenting {category}"):
                # Pilih file secara acak untuk diaugmentasi
                audio_file = random.choice(audio_files)
                file_name = os.path.basename(audio_file)
                base_name, ext = os.path.splitext(file_name)
                
                # Baca file audio
                y, sr = librosa.load(audio_file, sr=None)
                
                # Lakukan augmentasi
                y_augmented = augment_audio(y, sr)
                
                # Simpan hasil augmentasi
                output_filename = f"{base_name}_aug_{i}{ext}"
                output_path = os.path.join(category_output_path, output_filename)
                sf.write(output_path, y_augmented, sr)
    
    print(f"Preprocessing tahap 2 selesai. Data siap pakai disimpan di {output_dir}")
    
    # Hitung jumlah file per kategori hasil preprocessing
    print(f"Jumlah file yang dihasilkan per kategori:")
    categories = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    for category in categories:
        category_path = os.path.join(output_dir, category)
        files = glob(os.path.join(category_path, "*.wav"))
        print(f"- {category}: {len(files)} file")

if __name__ == "__main__":
    # Direktori input dan output
    input_directory = "dataset_preprocessed/stage1"
    output_directory = "dataset_preprocessed/stage2"
    
    # Jalankan preprocessing tahap 2
    preprocess_stage2(
        input_dir=input_directory,
        output_dir=output_directory,
        hungry_samples=400
    )
    
    print("Preprocessing tahap 2 selesai!") 