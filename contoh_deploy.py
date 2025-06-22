"""
Contoh Implementasi Deployment
Klasifikasi Tangisan Bayi menggunakan SVM

File ini menunjukkan cara menggunakan preprocessing dan model untuk prediksi
"""

import numpy as np
import joblib
import os
import sys
from deploy_preprocess import AudioPreprocessor, get_label_name, validate_audio_file
from preprocess_config import *

# Import class SVM yang diperlukan untuk loading model
sys.path.append('.')
from svm_classes import KernelSVM, OneVsRestSVM

class BabyCryClassifier:
    """
    Class untuk klasifikasi tangisan bayi
    """
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Inisialisasi classifier
        
        Parameters:
        - model_path: path ke file model SVM
        - scaler_path: path ke file scaler
        """
        # Inisialisasi preprocessor
        self.preprocessor = AudioPreprocessor(scaler_path)
        
        # Load model
        self.model = None
        self.model_info = None
        if model_path and os.path.exists(model_path):
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict) and 'model' in model_data:
                self.model = model_data['model']
                self.model_info = model_data
                print(f"✅ Model berhasil dimuat dari {model_path}")
                print(f"   - Best params: {model_data.get('best_params', 'N/A')}")
                print(f"   - Test accuracy: {model_data.get('test_accuracy', 'N/A'):.4f}")
            else:
                self.model = model_data
                print(f"✅ Model berhasil dimuat dari {model_path}")
        elif model_path:
            print(f"⚠️  File model tidak ditemukan: {model_path}")
        
        # Load scaler jika belum dimuat di preprocessor
        if scaler_path and self.preprocessor.scaler is None:
            self.preprocessor.load_scaler(scaler_path)
    
    def predict(self, audio_path, return_probabilities=False):
        """
        Melakukan prediksi pada file audio
        
        Parameters:
        - audio_path: path ke file audio
        - return_probabilities: apakah mengembalikan probabilitas
        
        Returns:
        - prediction: hasil prediksi
        """
        if self.model is None:
            raise Exception("Model belum dimuat. Gunakan load_model() terlebih dahulu.")
        
        # Validasi file audio
        is_valid, message = validate_audio_file(audio_path)
        if not is_valid:
            raise Exception(f"File audio tidak valid: {message}")
        
        # Preprocessing
        features = self.preprocessor.preprocess_for_prediction(audio_path)
        
        # Reshape untuk prediksi (model memerlukan 2D array)
        features_2d = features.reshape(1, -1)
        
        # Prediksi
        prediction = self.model.predict(features_2d)[0]
        label_name = get_label_name(prediction)
        
        result = {
            'prediction_index': int(prediction),
            'prediction_label': label_name,
            'features_shape': features.shape,
            'audio_file': os.path.basename(audio_path)
        }
        
        # Tambahkan probabilitas jika diminta dan model mendukung
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_2d)[0]
            result['probabilities'] = {
                get_label_name(i): float(prob) 
                for i, prob in enumerate(probabilities)
            }
        elif return_probabilities and hasattr(self.model, 'decision_function'):
            # Untuk SVM, gunakan decision function
            decision_scores = self.model.decision_function(features_2d)[0]
            result['decision_scores'] = {
                get_label_name(i): float(score) 
                for i, score in enumerate(decision_scores)
            }
        
        return result
    
    def predict_batch(self, audio_paths, return_probabilities=False):
        """
        Melakukan prediksi pada multiple file audio
        
        Parameters:
        - audio_paths: list path ke file audio
        - return_probabilities: apakah mengembalikan probabilitas
        
        Returns:
        - results: list hasil prediksi
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path, return_probabilities)
                results.append(result)
            except Exception as e:
                results.append({
                    'audio_file': os.path.basename(audio_path),
                    'error': str(e)
                })
        
        return results
    
    def load_model(self, model_path):
        """
        Memuat model dari file
        
        Parameters:
        - model_path: path ke file model
        """
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict) and 'model' in model_data:
                self.model = model_data['model']
                self.model_info = model_data
                print(f"✅ Model berhasil dimuat dari {model_path}")
                print(f"   - Best params: {model_data.get('best_params', 'N/A')}")
                print(f"   - Test accuracy: {model_data.get('test_accuracy', 'N/A'):.4f}")
            else:
                self.model = model_data
                print(f"✅ Model berhasil dimuat dari {model_path}")
        else:
            raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
    
    def get_model_info(self):
        """
        Mendapatkan informasi tentang model
        
        Returns:
        - info: dictionary informasi model
        """
        if self.model is None:
            return {"error": "Model belum dimuat"}
        
        info = {
            "model_type": type(self.model).__name__,
            "has_predict_proba": hasattr(self.model, 'predict_proba'),
            "has_decision_function": hasattr(self.model, 'decision_function'),
            "scaler_loaded": self.preprocessor.scaler is not None,
            "n_classes": N_CLASSES,
            "label_classes": LABEL_CLASSES,
            "n_features": N_TOTAL_FEATURES
        }
        
        # Tambahkan informasi spesifik SVM jika applicable
        if hasattr(self.model, 'kernel'):
            info["kernel"] = self.model.kernel
        if hasattr(self.model, 'C'):
            info["C"] = self.model.C
        if hasattr(self.model, 'gamma'):
            info["gamma"] = self.model.gamma
        
        # Tambahkan informasi dari model_info jika tersedia
        if self.model_info:
            info["best_params"] = self.model_info.get('best_params', 'N/A')
            info["test_accuracy"] = self.model_info.get('test_accuracy', 'N/A')
            info["validation_accuracy"] = self.model_info.get('validation_accuracy', 'N/A')
            info["training_time"] = self.model_info.get('training_time', 'N/A')
            info["prediction_time"] = self.model_info.get('prediction_time', 'N/A')
        
        return info

# def demo_prediction():
#     """
#     Demo prediksi menggunakan file audio dari dataset
#     """
#     print("\n🎯 Demo Prediksi Klasifikasi Tangisan Bayi")
#     print("=" * 60)
    
#     # Inisialisasi classifier
#     classifier = BabyCryClassifier(
#         model_path=BEST_MODEL_PATH,
#         scaler_path=SCALER_PATH
#     )
    
#     # Tampilkan info model
#     model_info = classifier.get_model_info()
#     print(f"\n📋 Informasi Model:")
#     for key, value in model_info.items():
#         print(f"  - {key}: {value}")
    
#     # Cari beberapa file audio untuk demo
#     test_audio_dir = "dataset_preprocessed/stage2"
#     demo_files = []
    
#     if os.path.exists(test_audio_dir):
#         for category in LABEL_CLASSES:
#             category_path = os.path.join(test_audio_dir, category)
#             if os.path.exists(category_path):
#                 audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
#                 if audio_files:
#                     # Ambil satu file dari setiap kategori
#                     demo_file = os.path.join(category_path, audio_files[0])
#                     demo_files.append(demo_file)
    
#     if not demo_files:
#         print("\n⚠️  Tidak ada file audio demo yang ditemukan.")
#         print("   Pastikan dataset ada di 'dataset_preprocessed/stage2'")
#         return
    
#     print(f"\n🎵 Melakukan prediksi pada {len(demo_files)} file demo...")
    
#     # Prediksi batch
#     results = classifier.predict_batch(demo_files, return_probabilities=True)
    
#     print(f"\n📊 Hasil Prediksi:")
#     print("-" * 60)
    
#     for result in results:
#         if 'error' in result:
#             print(f"❌ {result['audio_file']}: {result['error']}")
#         else:
#             print(f"📁 File: {result['audio_file']}")
#             print(f"🎯 Prediksi: {result['prediction_label']} (index: {result['prediction_index']})")
#             print(f"📐 Fitur: {result['features_shape']}")
            
#             # Tampilkan probabilitas/scores jika ada
#             if 'probabilities' in result:
#                 print(f"📈 Probabilitas:")
#                 for label, prob in result['probabilities'].items():
#                     print(f"     {label}: {prob:.4f}")
#             elif 'decision_scores' in result:
#                 print(f"📈 Decision Scores:")
#                 for label, score in result['decision_scores'].items():
#                     print(f"     {label}: {score:.4f}")
            
#             print("-" * 60)

# def test_single_prediction():
#     """
#     Test prediksi pada satu file audio
#     """
#     print("\n🔍 Test Prediksi Single File")
#     print("=" * 40)
    
#     # Inisialisasi classifier
#     classifier = BabyCryClassifier(
#         model_path=BEST_MODEL_PATH,
#         scaler_path=SCALER_PATH
#     )
    
#     # Cari satu file audio untuk test
#     test_audio_dir = "dataset_preprocessed/stage2"
#     test_file = None
    
#     if os.path.exists(test_audio_dir):
#         for category in LABEL_CLASSES:
#             category_path = os.path.join(test_audio_dir, category)
#             if os.path.exists(category_path):
#                 audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
#                 if audio_files:
#                     test_file = os.path.join(category_path, audio_files[0])
#                     break
    
#     if not test_file:
#         print("⚠️  Tidak ada file audio test yang ditemukan.")
#         return
    
#     print(f"🎵 Testing file: {os.path.basename(test_file)}")
    
#     try:
#         # Prediksi
#         result = classifier.predict(test_file, return_probabilities=True)
        
#         print(f"\n✅ Hasil Prediksi:")
#         print(f"  - File: {result['audio_file']}")
#         print(f"  - Prediksi: {result['prediction_label']}")
#         print(f"  - Index: {result['prediction_index']}")
#         print(f"  - Shape Fitur: {result['features_shape']}")
        
#         if 'probabilities' in result:
#             print(f"  - Probabilitas Tertinggi: {max(result['probabilities'].values()):.4f}")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")

# if __name__ == "__main__":
#     print("🚀 CONTOH DEPLOYMENT KLASIFIKASI TANGISAN BAYI")
#     print("=" * 80)
    
#     # Cek ketersediaan file yang diperlukan
#     print("\n📋 Checking Required Files...")
    
#     files_to_check = [
#         (BEST_MODEL_PATH, "Model SVM"),
#         (SCALER_PATH, "Scaler"),
#         ("dataset_preprocessed/stage2", "Dataset Test")
#     ]
    
#     all_files_exist = True
#     for file_path, description in files_to_check:
#         if os.path.exists(file_path):
#             print(f"  ✅ {description}: {file_path}")
#         else:
#             print(f"  ❌ {description}: {file_path} (tidak ditemukan)")
#             all_files_exist = False
    
#     if not all_files_exist:
#         print("\n⚠️  Beberapa file diperlukan tidak ditemukan.")
#         print("   Pastikan Anda sudah menjalankan training dan preprocessing.")
#         exit(1)
    
#     # Jalankan demo
#     try:
#         test_single_prediction()
#         demo_prediction()
        
#         print("\n🎉 Demo deployment berhasil!")
#         print("\n💡 Cara penggunaan untuk deployment:")
#         print("   1. Import BabyCryClassifier dari file ini")
#         print("   2. Inisialisasi dengan path model dan scaler")
#         print("   3. Gunakan method predict() untuk prediksi single file")
#         print("   4. Gunakan method predict_batch() untuk prediksi multiple files")
        
#     except Exception as e:
#         print(f"\n❌ Error dalam demo: {e}")
#         print("   Pastikan semua dependencies terinstall dan file tersedia.") 