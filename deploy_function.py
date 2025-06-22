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

# Import untuk pickle deserialization - Harus ada di global scope
import svm_classes
from svm_classes import KernelSVM, OneVsRestSVM

# Workaround untuk pickle deserialization - kedua class dibutuhkan
import __main__
__main__.KernelSVM = KernelSVM
__main__.OneVsRestSVM = OneVsRestSVM

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
