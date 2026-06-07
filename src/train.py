import os
import argparse
import subprocess
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_DIR = BASE_DIR / "yolov5"
DATASET_YAML = BASE_DIR / "data" / "dataset.yaml"
HYP_YAML = BASE_DIR / "data" / "hyp.custom.yaml"

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Custom Training Wrapper")
    parser.add_argument("--model", type=str, default="yolov5s.pt", help="Pretrained model weights (e.g., yolov5s.pt, yolov5m.pt, yolov5l.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--name", type=str, default="shapes_model", help="Run name")
    
    args = parser.parse_args()
    
    # 1. Setup YOLOv5 if missing
    if not YOLO_DIR.exists():
        print("YOLOv5 folder not found. Running setup...")
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "setup_yolov5.py")], check=True)
        
    # 2. Check dataset configuration
    if not DATASET_YAML.exists():
        print("Dataset configuration not found. Generating sample shapes dataset first...")
        subprocess.run([sys.executable, str(BASE_DIR / "data" / "generate_dataset.py")], check=True)
        
    # 3. Formulate training command
    cmd = [
        sys.executable,
        str(YOLO_DIR / "train.py"),
        "--img", str(args.img_size),
        "--batch", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--data", str(DATASET_YAML),
        "--weights", args.model,
        "--hyp", str(HYP_YAML),
        "--project", str(BASE_DIR / "runs" / "train"),
        "--name", args.name,
        "--exist-ok"
    ]
    
    print(f"Launching training command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        print(f"Results saved in runs/train/{args.name}")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
