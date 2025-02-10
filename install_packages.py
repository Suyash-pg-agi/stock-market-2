import subprocess
import sys

def install_packages(requirements_file):
    """Install packages from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages from {requirements_file}: {e}")

def install_additional_packages():
    """Install specific packages manually"""
    # torch_cmd = [
    #     sys.executable, "-m", "pip", "install",
    #     "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
    #     "--index-url", "https://download.pytorch.org/whl/cu121"
    # ]
    # tensorflow_cmd = [sys.executable, "-m", "pip", "install", "tensorflow==2.17.0"]
    polygon_cmd = [sys.executable, "-m", "pip", "install", "-U", "polygon-api-client"]

    try:
        # subprocess.check_call(torch_cmd)
        # subprocess.check_call(tensorflow_cmd)
        subprocess.check_call(polygon_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error installing additional packages: {e}")

if __name__ == "__main__":
    requirements_file = "requirements.txt"
    install_packages(requirements_file)
    install_additional_packages()
    print("All packages installed successfully!")
