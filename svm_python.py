"""
Support Vector Machine for Infant Cry Classification (Python Version)
Implementasi SVM from scratch dengan algoritma Sequential Minimal Optimization (SMO)
dan pendekatan One-vs-Rest untuk klasifikasi multiclass.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import product
import pickle
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed untuk reproducibility
np.random.seed(42)

print("=" * 80)
print("SUPPORT VECTOR MACHINE FOR INFANT CRY CLASSIFICATION")
print("=" * 80)

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n1. LOADING DATASET...")

# Load dataset
dataset_path = 'dataset_preprocessed/stage3/dataset.npz'
data = np.load(dataset_path, allow_pickle=True)

# Extract data
X_train = data['X_train']
X_val = data['X_val']
X_test = data['X_test']
y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']
feature_names = data['feature_names']
label_mapping = data['label_mapping']

print("Dataset berhasil dimuat!")
print(f"Shape X_train: {X_train.shape}")
print(f"Shape X_val: {X_val.shape}")
print(f"Shape X_test: {X_test.shape}")
print(f"Jumlah fitur: {len(feature_names)}")
print(f"Label mapping: {label_mapping}")
print(f"Jumlah kelas: {len(np.unique(y_train))}")

# ============================================================================
# 2. IMPLEMENTASI SVM FROM SCRATCH DENGAN SMO ALGORITHM
# ============================================================================
print("\n2. IMPLEMENTASI SVM FROM SCRATCH...")

class KernelSVM:
    """
    Support Vector Machine dengan Sequential Minimal Optimization (SMO)
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, max_iter=1000, tol=1e-3, eps=1e-3):
        """
        Parameters:
        - C: parameter regularisasi
        - kernel: jenis kernel ('linear' atau 'rbf')
        - gamma: parameter untuk RBF kernel
        - max_iter: maksimum iterasi
        - tol: toleransi untuk konvergensi
        - eps: epsilon untuk SMO
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        
    def _kernel_function(self, x1, x2):
        """Menghitung kernel function"""
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel == 'rbf':
            # RBF kernel: exp(-gamma * ||x1 - x2||^2)
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            if x2.ndim == 1:
                x2 = x2.reshape(1, -1)
            
            sq_dists = np.sum((x1[:, np.newaxis] - x2[np.newaxis, :]) ** 2, axis=2)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError("Kernel tidak didukung. Gunakan 'linear' atau 'rbf'")
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """Menghitung kernel matrix"""
        if X2 is None:
            X2 = X1
        return self._kernel_function(X1, X2)
    
    def _objective_function(self, i, j):
        """Menghitung objective function untuk pasangan alpha i dan j"""
        if i == j:
            return 0
        
        eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
        if eta <= 0:
            return 0
        
        return eta
    
    def _select_second_alpha(self, i, E_i):
        """Memilih alpha kedua untuk optimasi"""
        max_step = 0
        j = -1
        E_j = 0
        
        # Cari alpha yang memberikan langkah terbesar
        for k in range(len(self.alphas)):
            if k == i:
                continue
            E_k = self._compute_error(k)
            step = abs(E_i - E_k)
            if step > max_step:
                max_step = step
                j = k
                E_j = E_k
        
        if j == -1:
            # Jika tidak ada yang cocok, pilih secara random
            j = np.random.choice([k for k in range(len(self.alphas)) if k != i])
            E_j = self._compute_error(j)
            
        return j, E_j
    
    def _compute_error(self, i):
        """Menghitung error untuk sample i"""
        prediction = np.sum(self.alphas * self.y * self.K[:, i]) + self.b
        return prediction - self.y[i]
    
    def _update_alpha_pair(self, i, j):
        """Update pasangan alpha i dan j"""
        E_i = self._compute_error(i)
        E_j = self._compute_error(j)
        
        # Simpan nilai lama
        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]
        
        # Hitung batas L dan H
        if self.y[i] != self.y[j]:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0, self.alphas[i] + self.alphas[j] - self.C)
            H = min(self.C, self.alphas[i] + self.alphas[j])
        
        if L == H:
            return 0
        
        # Hitung eta
        eta = self._objective_function(i, j)
        if eta <= 0:
            return 0
        
        # Update alpha_j
        self.alphas[j] = alpha_j_old + self.y[j] * (E_i - E_j) / eta
        
        # Clip alpha_j
        if self.alphas[j] > H:
            self.alphas[j] = H
        elif self.alphas[j] < L:
            self.alphas[j] = L
        
        # Cek apakah perubahan cukup signifikan
        if abs(self.alphas[j] - alpha_j_old) < self.eps:
            return 0
        
        # Update alpha_i
        self.alphas[i] = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - self.alphas[j])
        
        # Update bias b
        b1 = self.b - E_i - self.y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, i] - \
             self.y[j] * (self.alphas[j] - alpha_j_old) * self.K[i, j]
        
        b2 = self.b - E_j - self.y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, j] - \
             self.y[j] * (self.alphas[j] - alpha_j_old) * self.K[j, j]
        
        if 0 < self.alphas[i] < self.C:
            self.b = b1
        elif 0 < self.alphas[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        return 1
    
    def fit(self, X, y):
        """Training SVM menggunakan SMO algorithm"""
        self.X = X
        self.y = y.astype(float)
        n_samples = X.shape[0]
        
        # Inisialisasi
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # Hitung kernel matrix
        self.K = self._compute_kernel_matrix(X)
        
        # SMO algorithm
        iteration = 0
        num_changed = 0
        examine_all = True
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            if examine_all:
                # Periksa semua sample
                for i in range(n_samples):
                    if self._examine_example(i):
                        num_changed += 1
            else:
                # Periksa hanya non-bound samples
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        if self._examine_example(i):
                            num_changed += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
        
        # Simpan support vectors
        support_vector_idx = self.alphas > self.eps
        self.support_vectors = X[support_vector_idx]
        self.support_vector_labels = y[support_vector_idx]
        self.support_vector_alphas = self.alphas[support_vector_idx]
        
        return self
    
    def _examine_example(self, i):
        """Periksa contoh i untuk optimasi"""
        E_i = self._compute_error(i)
        r_i = E_i * self.y[i]
        
        if (r_i < -self.tol and self.alphas[i] < self.C) or \
           (r_i > self.tol and self.alphas[i] > 0):
            
            # Pilih alpha kedua
            j, E_j = self._select_second_alpha(i, E_i)
            
            if self._update_alpha_pair(i, j):
                return True
        
        return False
    
    def predict(self, X):
        """Prediksi untuk data baru"""
        if not hasattr(self, 'support_vectors'):
            raise ValueError("Model belum di-training!")
        
        # Hitung kernel antara data baru dan support vectors
        K_pred = self._kernel_function(self.support_vectors, X)
        
        # Hitung prediksi
        predictions = np.sum(self.support_vector_alphas.reshape(-1, 1) * 
                           self.support_vector_labels.reshape(-1, 1) * K_pred, axis=0) + self.b
        
        return np.sign(predictions)
    
    def decision_function(self, X):
        """Menghitung decision function untuk data baru"""
        if not hasattr(self, 'support_vectors'):
            raise ValueError("Model belum di-training!")
        
        # Hitung kernel antara data baru dan support vectors
        K_pred = self._kernel_function(self.support_vectors, X)
        
        # Hitung decision function
        decision = np.sum(self.support_vector_alphas.reshape(-1, 1) * 
                         self.support_vector_labels.reshape(-1, 1) * K_pred, axis=0) + self.b
        
        return decision

print("✅ Implementasi KernelSVM selesai!")

# ============================================================================
# 3. IMPLEMENTASI ONE-VS-REST UNTUK MULTICLASS CLASSIFICATION
# ============================================================================
print("\n3. IMPLEMENTASI ONE-VS-REST...")

class OneVsRestSVM:
    """
    One-vs-Rest SVM untuk klasifikasi multiclass
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, max_iter=1000, tol=1e-3, eps=1e-3):
        """
        Parameters sama dengan KernelSVM
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.classifiers = {}
        
    def fit(self, X, y):
        """
        Training One-vs-Rest SVM
        """
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        print(f"Training {self.n_classes} binary classifiers...")
        
        # Training satu classifier untuk setiap kelas
        for i, class_label in enumerate(tqdm(self.classes, desc="Training classifiers")):
            # Buat binary labels: +1 untuk kelas saat ini, -1 untuk yang lain
            binary_y = np.where(y == class_label, 1, -1)
            
            # Training binary SVM
            classifier = KernelSVM(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                max_iter=self.max_iter,
                tol=self.tol,
                eps=self.eps
            )
            
            classifier.fit(X, binary_y)
            self.classifiers[class_label] = classifier
        
        return self
    
    def predict(self, X):
        """
        Prediksi menggunakan One-vs-Rest
        """
        if not self.classifiers:
            raise ValueError("Model belum di-training!")
        
        n_samples = X.shape[0]
        decision_scores = np.zeros((n_samples, self.n_classes))
        
        # Hitung decision function untuk setiap classifier
        for i, class_label in enumerate(self.classes):
            decision_scores[:, i] = self.classifiers[class_label].decision_function(X)
        
        # Pilih kelas dengan decision score tertinggi
        predicted_class_indices = np.argmax(decision_scores, axis=1)
        predictions = self.classes[predicted_class_indices]
        
        return predictions
    
    def predict_proba(self, X):
        """
        Prediksi probabilitas (menggunakan decision function sebagai proxy)
        """
        if not self.classifiers:
            raise ValueError("Model belum di-training!")
        
        n_samples = X.shape[0]
        decision_scores = np.zeros((n_samples, self.n_classes))
        
        # Hitung decision function untuk setiap classifier
        for i, class_label in enumerate(self.classes):
            decision_scores[:, i] = self.classifiers[class_label].decision_function(X)
        
        # Normalisasi menggunakan softmax
        exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities

print("✅ Implementasi OneVsRestSVM selesai!")

# ============================================================================
# 4. GRID SEARCH UNTUK HYPERPARAMETER TUNING
# ============================================================================
print("\n4. GRID SEARCH HYPERPARAMETER TUNING...")

def grid_search_svm(X_train, y_train, X_val, y_val, param_grid):
    """
    Grid search untuk finding best hyperparameters
    """
    best_score = 0
    best_params = None
    best_model = None
    results = []
    
    # Generate semua kombinasi parameter
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    
    print(f"Melakukan grid search dengan {len(param_combinations)} kombinasi parameter...")
    print(f"Parameter yang di-tune: {param_names}")
    
    for i, params in enumerate(tqdm(param_combinations, desc="Grid Search")):
        # Buat dictionary parameter
        current_params = dict(zip(param_names, params))
        
        try:
            # Training model dengan parameter saat ini
            model = OneVsRestSVM(**current_params)
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluasi pada validation set
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            
            # Simpan hasil
            result = {
                'params': current_params.copy(),
                'validation_accuracy': score,
                'train_time': train_time
            }
            results.append(result)
            
            # Update best model jika perlu
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
                best_model = model
            
            print(f"\nKombinasi {i+1}/{len(param_combinations)}:")
            print(f"  Parameters: {current_params}")
            print(f"  Validation Accuracy: {score:.4f}")
            print(f"  Training Time: {train_time:.2f} seconds")
            
        except Exception as e:
            print(f"\nError pada kombinasi {i+1}: {current_params}")
            print(f"  Error: {str(e)}")
            continue
    
    print(f"\n{'='*80}")
    print("HASIL GRID SEARCH:")
    print(f"Best Validation Accuracy: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"{'='*80}")
    
    return best_model, best_params, best_score, results

# Parameter grid untuk linear kernel
linear_param_grid = {
    'kernel': ['linear'],
    'C': [0.1, 1.0, 10.0],
    'max_iter': [500, 1000],
    'tol': [1e-3, 1e-4],
    'eps': [1e-3, 1e-4],
    'gamma': [1.0]  # Tidak digunakan untuk linear, tapi perlu ada
}

# Parameter grid untuk RBF kernel
rbf_param_grid = {
    'kernel': ['rbf'],
    'C': [0.1, 1.0, 10.0],
    'gamma': [0.001, 0.01, 0.1],
    'max_iter': [500, 1000],
    'tol': [1e-3, 1e-4],
    'eps': [1e-3, 1e-4]
}

print("Grid search akan dilakukan dalam 2 tahap:")
print(f"1. Linear kernel: {np.prod([len(v) for v in linear_param_grid.values()])} kombinasi")
print(f"2. RBF kernel: {np.prod([len(v) for v in rbf_param_grid.values()])} kombinasi")

# ============================================================================
# 5. EKSEKUSI GRID SEARCH
# ============================================================================
print("\n5. EKSEKUSI GRID SEARCH...")

print("=" * 80)
print("GRID SEARCH - LINEAR KERNEL")
print("=" * 80)

best_linear_model, best_linear_params, best_linear_score, linear_results = grid_search_svm(
    X_train, y_train, X_val, y_val, linear_param_grid
)

print("=" * 80)
print("GRID SEARCH - RBF KERNEL")
print("=" * 80)

best_rbf_model, best_rbf_params, best_rbf_score, rbf_results = grid_search_svm(
    X_train, y_train, X_val, y_val, rbf_param_grid
)

# Pilih best model secara keseluruhan
if best_linear_score > best_rbf_score:
    best_overall_model = best_linear_model
    best_overall_params = best_linear_params
    best_overall_score = best_linear_score
    best_kernel_type = "Linear"
else:
    best_overall_model = best_rbf_model
    best_overall_params = best_rbf_params
    best_overall_score = best_rbf_score
    best_kernel_type = "RBF"

print("=" * 80)
print("BEST MODEL OVERALL")
print("=" * 80)
print(f"Best Kernel Type: {best_kernel_type}")
print(f"Best Validation Accuracy: {best_overall_score:.4f}")
print(f"Best Parameters:")
for param, value in best_overall_params.items():
    print(f"  {param}: {value}")
print("=" * 80)

# ============================================================================
# 6. TRAINING FINAL MODEL DENGAN BEST PARAMETERS
# ============================================================================
print("\n6. TRAINING FINAL MODEL...")

print("Training final model dengan best parameters...")

# Training ulang model dengan best parameters pada data training
final_model = OneVsRestSVM(**best_overall_params)

start_time = time.time()
final_model.fit(X_train, y_train)
final_train_time = time.time() - start_time

print(f"Training selesai dalam {final_train_time:.2f} detik")

# Evaluasi pada validation set
y_val_pred = final_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation Accuracy: {val_accuracy:.4f}")

# ============================================================================
# 7. EVALUASI FINAL PADA TEST SET
# ============================================================================
print("\n7. EVALUASI FINAL PADA TEST SET...")

print("=" * 80)
print("EVALUASI FINAL PADA TEST SET")
print("=" * 80)

# Prediksi pada test set
start_time = time.time()
y_test_pred = final_model.predict(X_test)
prediction_time = time.time() - start_time

# Hitung akurasi test
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Prediction Time: {prediction_time:.4f} seconds")
print(f"Prediction Speed: {len(y_test)/prediction_time:.2f} samples/second")

# Classification report
print("\nCLASSIFICATION REPORT:")
print("=" * 50)
report = classification_report(y_test, y_test_pred, target_names=label_mapping, zero_division=0)
print(report)

# Classification report dalam bentuk dictionary untuk analisis lebih lanjut
report_dict = classification_report(y_test, y_test_pred, target_names=label_mapping, 
                                   output_dict=True, zero_division=0)

# Tampilkan metrik per kelas
print("\nMETRIK PER KELAS:")
print("=" * 50)
for i, class_name in enumerate(label_mapping):
    if class_name in report_dict:
        metrics = report_dict[class_name]
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
        print(f"  Support: {int(metrics['support'])}")
        print()

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_mapping, yticklabels=label_mapping)
plt.title('Confusion Matrix - SVM Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Tampilkan confusion matrix dalam bentuk tabel
print("\nCONFUSION MATRIX:")
print("=" * 50)
cm_df = pd.DataFrame(cm, index=label_mapping, columns=label_mapping)
print(cm_df)

# Hitung dan tampilkan akurasi per kelas
print("\nAKURASI PER KELAS:")
print("=" * 50)
class_accuracies = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(label_mapping):
    print(f"{class_name}: {class_accuracies[i]:.4f} ({cm.diagonal()[i]}/{cm.sum(axis=1)[i]})")

# ============================================================================
# 8. ANALISIS RESULTS DAN MODEL INFORMATION
# ============================================================================
print("\n8. ANALISIS RESULTS...")

print("=" * 80)
print("RINGKASAN HASIL MODEL")
print("=" * 80)

print(f"Dataset Information:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Validation samples: {X_val.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Number of features: {X_train.shape[1]}")
print(f"  Number of classes: {len(label_mapping)}")
print(f"  Classes: {list(label_mapping)}")

print(f"\nBest Model Configuration:")
for param, value in best_overall_params.items():
    print(f"  {param}: {value}")

print(f"\nModel Performance:")
print(f"  Best Validation Accuracy: {best_overall_score:.4f}")
print(f"  Final Test Accuracy: {test_accuracy:.4f}")
print(f"  Training Time: {final_train_time:.2f} seconds")
print(f"  Prediction Time: {prediction_time:.4f} seconds")

# Informasi tentang support vectors untuk setiap binary classifier
print(f"\nSupport Vectors Information:")
total_support_vectors = 0
for class_label in final_model.classes:
    classifier = final_model.classifiers[class_label]
    n_sv = len(classifier.support_vectors) if hasattr(classifier, 'support_vectors') else 0
    total_support_vectors += n_sv
    print(f"  Class {label_mapping[class_label]}: {n_sv} support vectors")
    
print(f"  Total support vectors: {total_support_vectors}")
print(f"  Support vector ratio: {total_support_vectors/(X_train.shape[0]*len(label_mapping)):.4f}")

# ============================================================================
# 9. SIMPAN MODEL
# ============================================================================
print("\n9. SIMPAN MODEL...")

# Simpan best model
model_filename = 'best_svm_model.pkl'

# Buat dictionary untuk menyimpan semua informasi model
model_package = {
    'model': final_model,
    'best_params': best_overall_params,
    'validation_accuracy': best_overall_score,
    'test_accuracy': test_accuracy,
    'label_mapping': label_mapping,
    'feature_names': feature_names,
    'training_time': final_train_time,
    'prediction_time': prediction_time,
    'grid_search_results': {
        'linear_results': linear_results,
        'rbf_results': rbf_results
    }
}

# Simpan model
with open(model_filename, 'wb') as f:
    pickle.dump(model_package, f)

print(f"Model berhasil disimpan sebagai '{model_filename}'")
print(f"\nUntuk memuat model:")
print(f"with open('{model_filename}', 'rb') as f:")
print(f"    model_package = pickle.load(f)")
print(f"    model = model_package['model']")
print(f"    best_params = model_package['best_params']")

print("\n" + "=" * 80)
print("IMPLEMENTASI SVM SELESAI!")
print("=" * 80)
print(f"✅ Dataset berhasil dimuat dan diproses")
print(f"✅ SVM from scratch dengan SMO algorithm berhasil diimplementasi")
print(f"✅ One-vs-Rest multiclass classification berhasil diimplementasi")
print(f"✅ Grid search hyperparameter tuning berhasil dilakukan")
print(f"✅ Model evaluation dengan berbagai metrik berhasil dilakukan")
print(f"✅ Best model berhasil disimpan dalam format .pkl")
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
print(f"Best Parameters: {best_overall_params}") 