# Setup Environment untuk Infant Cry Classification

## Prasyarat
- Python 3.8 atau lebih tinggi
- pip (Python package installer)

## Langkah-langkah Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd InfantCryClassification
```

### 2. Setup Otomatis (Recommended)

#### Untuk Windows:
```bash
# Opsi 1: Double-click file setup.bat
# Atau jalankan di Command Prompt/PowerShell:
setup.bat

# Opsi 2: Langsung dengan Python
python setup.py
```

#### Untuk macOS/Linux:
```bash
python3 setup.py
# atau
python setup.py
```

### 3. Setup Manual (Jika Diperlukan)

Jika setup otomatis gagal, ikuti langkah manual berikut:

#### 3.1. Membuat Virtual Environment

#### Untuk Windows:
```bash
# Membuat virtual environment
python -m venv .venv

# Mengaktifkan virtual environment
.venv\Scripts\activate
```

#### Untuk macOS/Linux:
```bash
# Membuat virtual environment
python -m venv .venv

# Mengaktifkan virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install semua package yang diperlukan
pip install -r requirements.txt
```

### 4. Verifikasi Installation
```bash
# Cek apakah semua package terinstall dengan benar
pip list
```

### 5. Setup VS Code (Opsional)

1. Buka VS Code di folder project
2. Tekan `Ctrl+Shift+P` (Windows) atau `Cmd+Shift+P` (macOS)
3. Ketik "Python: Select Interpreter"
4. Pilih interpreter dari virtual environment yang baru dibuat:
   - Windows: `.venv\Scripts\python.exe`
   - macOS/Linux: `.venv/bin/python`

### 6. Menjalankan Aplikasi

#### Streamlit App:
```bash
streamlit run streamlit_fix.py
```

#### Jupyter Notebook:
```bash
jupyter notebook svm.ipynb
```

## Troubleshooting

### Problem: Virtual environment tidak muncul di VS Code
**Solusi:**
1. Pastikan virtual environment sudah dibuat dengan benar
2. Restart VS Code
3. Gunakan Command Palette (`Ctrl+Shift+P`) → "Python: Select Interpreter"
4. Browse manual ke folder `.venv/Scripts/python.exe` (Windows) atau `.venv/bin/python` (macOS/Linux)

### Problem: Package tidak terinstall
**Solusi:**
```bash
# Pastikan virtual environment aktif
# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# Upgrade pip terlebih dahulu
python -m pip install --upgrade pip

# Install ulang requirements
pip install -r requirements.txt
```

### Problem: Error "To modify pip, please run the following command"
**Solusi:**
```bash
# Aktifkan virtual environment terlebih dahulu
.venv\Scripts\activate

# Upgrade pip dengan cara yang benar
python -m pip install --upgrade pip

# Lalu install requirements
pip install -r requirements.txt
```

### Problem: Error saat install librosa atau numba
**Solusi:**
```bash
# Install Microsoft Visual C++ Build Tools (Windows)
# Atau install package satu per satu:
pip install numpy
pip install scipy
pip install librosa
pip install numba
```

### Problem: Setup.py gagal dengan error permission
**Solusi:**
1. Jalankan Command Prompt sebagai Administrator
2. Atau disable antivirus sementara
3. Atau coba setup manual step by step

## Catatan Penting

- **Selalu aktifkan virtual environment** sebelum menjalankan script Python
- File model `best_svm_model.pkl` (76MB) mungkin tidak ter-upload ke GitHub karena ukurannya. Jika diperlukan, gunakan Git LFS atau download terpisah
- Pastikan dataset ada di folder yang benar sesuai struktur project

## Struktur Virtual Environment
```
.venv/
├── Scripts/          # Windows executables
├── Lib/             # Python packages
├── Include/         # Header files
└── pyvenv.cfg       # Configuration
``` 