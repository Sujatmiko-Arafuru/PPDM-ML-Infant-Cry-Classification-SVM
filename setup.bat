@echo off
echo ========================================
echo Setup Infant Cry Classification Project
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python tidak ditemukan! 
    echo    Silakan install Python terlebih dahulu dari https://python.org
    echo    Pastikan mencentang "Add Python to PATH" saat instalasi
    pause
    exit /b 1
)

echo ✅ Python terdeteksi
echo.

REM Run the Python setup script
echo 🚀 Menjalankan setup otomatis...
python setup.py

if errorlevel 1 (
    echo.
    echo ❌ Setup gagal! Coba jalankan manual:
    echo    python setup.py
    pause
    exit /b 1
)

echo.
echo 🎉 Setup selesai!
echo.
echo 📋 Langkah selanjutnya:
echo 1. Aktifkan virtual environment: .venv\Scripts\activate
echo 2. Jalankan aplikasi: streamlit run streamlit_fix.py
echo.
pause 