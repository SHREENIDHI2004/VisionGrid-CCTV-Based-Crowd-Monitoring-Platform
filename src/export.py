import os
import argparse
import subprocess
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_DIR = BASE_DIR / "yolov5"

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Model Export Wrapper")
    parser.add_argument("--weights", type=str, default="runs/train/shapes_model/weights/best.pt", help="Path to model weights (.pt file)")
    parser.add_argument("--include", nargs="+", default=["onnx", "torchscript"], help="Formats to export to (e.g. onnx, torchscript, engine, coreml)")
    
    args = parser.parse_args()
    
    # 1. Check if YOLOv5 dir exists
    if not YOLO_DIR.exists():
        print("YOLOv5 repository not found. Please run src/setup_yolov5.py first.")
        sys.exit(1)
        
    # 2. Check weights path
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file not found at: {weights_path}")
        print("Please train your model first or provide the correct path to weights using --weights.")
        sys.exit(1)
        
    # 3. Formulate export command
    formats = " ".join(args.include)
    cmd = [
        sys.executable,
        str(YOLO_DIR / "export.py"),
        "--weights", str(weights_path),
        "--include"
    ] + args.include
    
    print(f"Launching model export command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("\nExport completed successfully!")
        print(f"Model exported formats: {args.include}")
        print(f"Exported files are located in the same directory as original weights: {weights_path.parent}")
    except subprocess.CalledProcessError as e:
        print(f"Export failed with error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
