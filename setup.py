"""
Script setup otomatis untuk Infant Cry Classification Project
Menangani pembuatan virtual environment dan instalasi dependencies

Cara menjalankan:
- Windows: python setup.py
- macOS/Linux: python3 setup.py atau python setup.py
"""

import os
import sys
import subprocess
import platform

def run_command(command, description="", allow_failure=False):
    """Menjalankan command dan menangani error"""
    print(f"\n🔄 {description}")
    print(f"Menjalankan: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} berhasil!")
        if result.stdout.strip():
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        if allow_failure:
            print(f"⚠️  {description} gagal, tapi dilanjutkan")
            if e.stderr:
                print(f"Error: {e.stderr.strip()}")
            return True
        else:
            print(f"❌ Error: {e}")
            if e.stdout:
                print(f"Output: {e.stdout.strip()}")
            if e.stderr:
                print(f"Error: {e.stderr.strip()}")
            return False

def check_python_version():
    """Mengecek versi Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 atau lebih tinggi diperlukan!")
        print(f"Versi saat ini: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} terdeteksi")
    return True

def create_virtual_environment():
    """Membuat virtual environment"""
    venv_path = ".venv"
    
    if os.path.exists(venv_path):
        print(f"⚠️  Virtual environment sudah ada di {venv_path}")
        response = input("Hapus dan buat ulang? (y/N): ").lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
            print("🗑️  Virtual environment lama dihapus")
        else:
            print("📁 Menggunakan virtual environment yang ada")
            return True
    
    return run_command(f"python -m venv {venv_path}", 
                      "Membuat virtual environment")

def get_activation_command():
    """Mendapatkan command untuk aktivasi virtual environment"""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"

def install_requirements():
    """Install requirements"""
    if platform.system() == "Windows":
        python_path = ".venv\\Scripts\\python.exe"
        pip_path = ".venv\\Scripts\\pip"
    else:
        python_path = ".venv/bin/python"
        pip_path = ".venv/bin/pip"
    
    # Upgrade pip menggunakan python -m pip (lebih reliable)
    print("\n🔄 Upgrade pip")
    print(f"Menjalankan: {python_path} -m pip install --upgrade pip")
    
    try:
        result = subprocess.run(f"{python_path} -m pip install --upgrade pip", 
                              shell=True, check=True, capture_output=True, text=True)
        print("✅ Upgrade pip berhasil!")
    except subprocess.CalledProcessError as e:
        # Jika upgrade pip gagal, lanjutkan saja (tidak critical)
        print("⚠️  Upgrade pip gagal, tapi akan lanjutkan install dependencies")
        print(f"Error: {e.stderr}")
    
    # Install requirements dengan fallback method
    if run_command(f"{pip_path} install -r requirements.txt", 
                   "Install dependencies dari requirements.txt"):
        return True
    
    # Jika gagal, coba dengan python -m pip
    print("\n⚠️  Instalasi dengan pip gagal, mencoba dengan python -m pip...")
    return run_command(f"{python_path} -m pip install -r requirements.txt", 
                      "Install dependencies dengan python -m pip")

def create_vscode_settings():
    """Membuat settings VS Code untuk interpreter"""
    vscode_dir = ".vscode"
    settings_file = os.path.join(vscode_dir, "settings.json")
    
    if not os.path.exists(vscode_dir):
        os.makedirs(vscode_dir)
    
    if platform.system() == "Windows":
        python_path = "./.venv/Scripts/python.exe"
    else:
        python_path = "./.venv/bin/python"
    
    settings_content = f'''{{
    "python.defaultInterpreterPath": "{python_path}",
    "python.terminal.activateEnvironment": true
}}'''
    
    try:
        with open(settings_file, 'w') as f:
            f.write(settings_content)
        print("✅ VS Code settings dibuat")
        return True
    except Exception as e:
        print(f"⚠️  Gagal membuat VS Code settings: {e}")
        return False

def main():
    """Fungsi utama setup"""
    print("🚀 Setup Infant Cry Classification Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Gagal membuat virtual environment")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Gagal install dependencies")
        sys.exit(1)
    
    # Create VS Code settings
    create_vscode_settings()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup berhasil!")
    print("\n📋 Langkah selanjutnya:")
    print(f"1. Aktifkan virtual environment: {get_activation_command()}")
    print("2. Jika menggunakan VS Code:")
    print("   - Restart VS Code")
    print("   - Tekan Ctrl+Shift+P → 'Python: Select Interpreter'")
    print("   - Pilih interpreter dari .venv")
    print("3. Jalankan aplikasi:")
    print("   streamlit run streamlit_fix.py")
    print("\n📖 Untuk troubleshooting, lihat SETUP_ENVIRONMENT.md")

if __name__ == "__main__":
    main() 