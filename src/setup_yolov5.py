import os
import subprocess
import sys
import urllib.request
import zipfile
import io
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_DIR = BASE_DIR / "yolov5"

def clone_with_git():
    print("Attempting to clone YOLOv5 repository using Git...")
    subprocess.run(
        ["git", "clone", "https://github.com/ultralytics/yolov5.git", str(YOLO_DIR)],
        check=True
    )
    print("Successfully cloned YOLOv5 using Git.")

def download_zip_fallback():
    print("Git clone failed or Git is not installed.")
    print("Downloading YOLOv5 repository zip archive from GitHub...")
    zip_url = "https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip"
    
    try:
        # Download zip
        with urllib.request.urlopen(zip_url) as response:
            zip_data = response.read()
            
        print("Extracting archive...")
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zip_ref:
            zip_ref.extractall(str(BASE_DIR))
            
        # The zip extracts as yolov5-master. Rename it to yolov5.
        extracted_dir = BASE_DIR / "yolov5-master"
        if extracted_dir.exists():
            if YOLO_DIR.exists():
                shutil.rmtree(str(YOLO_DIR))
            shutil.move(str(extracted_dir), str(YOLO_DIR))
            print("Successfully extracted and setup YOLOv5 folder.")
        else:
            raise FileNotFoundError("Could not locate extracted yolov5-master folder.")
            
    except Exception as e:
        print(f"Error downloading/extracting YOLOv5 zip: {e}", file=sys.stderr)
        print("Please check your internet connection.", file=sys.stderr)
        sys.exit(1)

def main():
    print(f"Project directory: {BASE_DIR}")
    
    # 1. Setup YOLOv5 folder
    if not YOLO_DIR.exists():
        try:
            clone_with_git()
        except Exception:
            download_zip_fallback()
    else:
        print("YOLOv5 repository directory already exists. Skipping acquisition.")

    # 2. Install requirements
    requirements_file = BASE_DIR / "requirements.txt"
    if requirements_file.exists():
        print("Installing dependencies from requirements.txt...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )
            print("Successfully installed dependencies.")
        except Exception as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)
            
    print("\nSetup complete! You are ready to generate data and train your YOLOv5 model.")

if __name__ == "__main__":
    main()
