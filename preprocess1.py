import os
import numpy as np
import librosa
import soundfile as sf
from glob import glob
from tqdm import tqdm
from scipy import signal

def reduce_noise(y, sr):
    """
    Fungsi untuk mengurangi noise pada sinyal audio
    menggunakan filter butterworth high-pass
    """
    # Menggunakan high-pass filter untuk mengurangi noise frekuensi rendah
    # Frekuensi cutoff 100 Hz
    b, a = signal.butter(5, 100/(sr/2), 'highpass')
    y_filtered = signal.filtfilt(b, a, y)
    return y_filtered

def preprocess_stage1(input_dir, output_dir, target_sr=22050, segment_duration=2.0, min_last_segment=1.0):
    """
    Tahap preprocessing pertama:
    1. Normalisasi audio ke rentang [-1, 1]
    2. Noise reduction
    3. Resampling ke sample rate target
    4. Segmentasi menjadi potongan 2 detik dengan padding/trimming
    
    Parameters:
    - input_dir: direktori dataset input
    - output_dir: direktori output untuk data hasil preprocessing
    - target_sr: sample rate target (default: 22050 Hz)
    - segment_duration: durasi segmen dalam detik (default: 2.0 detik)
    - min_last_segment: durasi minimum untuk segmen terakhir (default: 1.0 detik)
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
        
        print(f"Memproses {len(audio_files)} file dalam kategori {category}...")
        
        # Memproses setiap file audio
        for audio_file in tqdm(audio_files):
            file_name = os.path.basename(audio_file)
            file_base = os.path.splitext(file_name)[0]
            
            try:
                # 1. Memuat audio
                y, sr = librosa.load(audio_file, sr=None)
                
                # 2. Normalisasi audio ke rentang [-1, 1]
                y = librosa.util.normalize(y)
                
                # 3. Noise reduction
                y = reduce_noise(y, sr)
                
                # 4. Resampling jika sample rate berbeda dari target
                if sr != target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                
                # 5. Segmentasi audio
                segment_length = int(segment_duration * sr)
                segments = []
                
                for i, start in enumerate(range(0, len(y), segment_length)):
                    end = start + segment_length
                    
                    if end <= len(y):
                        # Segmen penuh
                        segment = y[start:end]
                        segments.append((segment, i))
                    else:
                        # Segmen terakhir (mungkin tidak penuh)
                        last_segment = y[start:]
                        last_duration = len(last_segment) / sr
                        
                        if last_duration >= min_last_segment:
                            # Jika durasi segmen terakhir >= min_last_segment, lakukan padding
                            padded_segment = np.zeros(segment_length)
                            padded_segment[:len(last_segment)] = last_segment
                            segments.append((padded_segment, i))
                        else:
                            # Jika durasi terlalu pendek, abaikan atau gabungkan dengan segmen sebelumnya
                            if len(segments) > 0:
                                # Ambil segmen terakhir
                                last_full_segment, last_idx = segments[-1]
                                # Gabungkan dengan segmen pendek dan trim ke panjang segment_length
                                combined = np.concatenate([last_full_segment, last_segment])
                                trimmed = combined[:segment_length]
                                # Ganti segmen terakhir dengan hasil gabungan
                                segments[-1] = (trimmed, last_idx)
                
                # 6. Simpan segmen-segmen ke file
                for segment, idx in segments:
                    output_filename = f"{file_base}_segment_{idx}.wav"
                    output_path = os.path.join(category_output_path, output_filename)
                    sf.write(output_path, segment, target_sr)
                    
            except Exception as e:
                print(f"Error memproses file {audio_file}: {str(e)}")
    
    print(f"Preprocessing tahap 1 selesai. Data disimpan di {output_dir}")

if __name__ == "__main__":
    # Direktori input dan output
    input_directory = "dataset ppdm2 asli/donateacry_corpus"
    output_directory = "dataset_preprocessed/stage1"
    
    # Jalankan preprocessing tahap 1
    preprocess_stage1(
        input_dir=input_directory,
        output_dir=output_directory,
        target_sr=22050,
        segment_duration=2.0,
        min_last_segment=1.0
    )
    
    print("Preprocessing tahap 1 selesai!")
    print(f"Jumlah file yang dihasilkan per kategori:")
    
    # Hitung jumlah file per kategori hasil preprocessing
    categories = [d for d in os.listdir(output_directory) if os.path.isdir(os.path.join(output_directory, d))]
    for category in categories:
        category_path = os.path.join(output_directory, category)
        files = glob(os.path.join(category_path, "*.wav"))
        print(f"- {category}: {len(files)} file")
