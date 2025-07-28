"""
Definisi Class SVM untuk Deployment
Klasifikasi Tangisan Bayi

File ini berisi class-class SVM yang diperlukan untuk loading model
yang sudah dilatih sebelumnya.
"""

import numpy as np

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
        self.y = y
        n_samples = X.shape[0]
        
        # Inisialisasi parameter
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # Hitung kernel matrix
        self.K = self._compute_kernel_matrix(X, X)
        
        # SMO algorithm
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            if examine_all:
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                for i in range(n_samples):
                    if 0 < self.alphas[i] < self.C:
                        num_changed += self._examine_example(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
                
            iteration += 1
        
        # Simpan support vectors (gunakan nama yang sama dengan implementasi asli)
        sv_indices = self.alphas > self.eps
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = self.alphas[sv_indices]
        
        return self
    
    def _examine_example(self, i):
        """Examine example untuk SMO"""
        y_i = self.y[i]
        alpha_i = self.alphas[i]
        E_i = self._compute_error(i)
        
        r_i = E_i * y_i
        
        if (r_i < -self.tol and alpha_i < self.C) or (r_i > self.tol and alpha_i > 0):
            # Pilih alpha kedua
            j, E_j = self._select_second_alpha(i, E_i)
            return self._update_alpha_pair(i, j)
        
        return 0
    
    def predict(self, X):
        """Prediksi menggunakan SVM yang sudah dilatih"""
        if not hasattr(self, 'support_vectors'):
            raise ValueError("Model belum dilatih. Jalankan fit() terlebih dahulu.")
        
        decision = self.decision_function(X)
        return np.sign(decision)
    
    def decision_function(self, X):
        """Menghitung decision function"""
        if not hasattr(self, 'support_vectors'):
            raise ValueError("Model belum dilatih. Jalankan fit() terlebih dahulu.")
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Hitung kernel antara X dan support vectors
        K_pred = self._kernel_function(self.support_vectors, X)
        
        # Hitung decision function
        decision = np.sum(self.support_vector_alphas.reshape(-1, 1) * 
                         self.support_vector_labels.reshape(-1, 1) * K_pred, axis=0) + self.b
        
        return decision

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
        self.classes = None
    
    def fit(self, X, y):
        """Training One-vs-Rest SVM"""
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        print(f"Training {n_classes} binary classifiers...")
        
        for i, class_label in enumerate(self.classes):
            print(f"Training classifier untuk kelas {class_label} ({i+1}/{n_classes})")
            
            # Buat binary labels (1 untuk kelas ini, -1 untuk yang lain)
            y_binary = np.where(y == class_label, 1, -1)
            
            # Train binary SVM
            classifier = KernelSVM(
                C=self.C, 
                kernel=self.kernel, 
                gamma=self.gamma,
                max_iter=self.max_iter,
                tol=self.tol,
                eps=self.eps
            )
            classifier.fit(X, y_binary)
            
            self.classifiers[class_label] = classifier
        
        return self
    
    def predict(self, X):
        """Prediksi menggunakan One-vs-Rest"""
        if not self.classifiers:
            raise ValueError("Model belum dilatih. Jalankan fit() terlebih dahulu.")
        
        n_samples = X.shape[0]
        decision_scores = np.zeros((n_samples, len(self.classes)))
        
        # Hitung decision score untuk setiap classifier
        for i, class_label in enumerate(self.classes):
            classifier = self.classifiers[class_label]
            scores = classifier.decision_function(X)
            decision_scores[:, i] = scores
        
        # Pilih kelas dengan score tertinggi
        predicted_indices = np.argmax(decision_scores, axis=1)
        predictions = self.classes[predicted_indices]
        
        return predictions
    
    def decision_function(self, X):
        """Menghitung decision function untuk semua kelas"""
        if not self.classifiers:
            raise ValueError("Model belum dilatih. Jalankan fit() terlebih dahulu.")
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        decision_scores = np.zeros((n_samples, len(self.classes)))
        
        # Hitung decision score untuk setiap classifier
        for i, class_label in enumerate(self.classes):
            classifier = self.classifiers[class_label]
            try:
                scores = classifier.decision_function(X)
                decision_scores[:, i] = scores
            except ValueError as e:
                # Jika classifier belum dilatih atau ada masalah, set score ke 0
                print(f"Warning: Error pada classifier {class_label}: {e}")
                decision_scores[:, i] = 0
        
        return decision_scores
    
    def predict_proba(self, X):
        """
        Prediksi probabilitas menggunakan softmax dari decision scores
        """
        decision_scores = self.decision_function(X)
        
        # Gunakan softmax untuk mengkonversi scores ke probabilitas
        exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities 