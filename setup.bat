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
    echo ❌ Setup gagal! 
    echo.
    echo 🔧 Troubleshooting:
    echo 1. Pastikan tidak ada antivirus yang memblokir
    echo 2. Jalankan Command Prompt sebagai Administrator
    echo 3. Coba setup manual: lihat SETUP_ENVIRONMENT.md
    echo 4. Atau coba: python -m pip install -r requirements.txt
    echo.
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