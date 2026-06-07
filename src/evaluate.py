import os
import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_DIR = BASE_DIR / "yolov5"
DATASET_YAML = BASE_DIR / "data" / "dataset.yaml"

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Model Evaluation")
    parser.add_argument("--weights", type=str, default="runs/train/shapes_model/weights/best.pt", help="Path to weights file")
    parser.add_argument("--task", type=str, default="test", choices=["val", "test", "train"], help="Split to evaluate on")
    parser.add_argument("--name", type=str, default="shapes_eval", help="Evaluation run name")
    
    args = parser.parse_args()
    
    # 1. Check if YOLOv5 exists
    if not YOLO_DIR.exists():
        print("YOLOv5 repository not found. Please run src/setup_yolov5.py first.")
        sys.exit(1)
        
    # 2. Check weights path
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file not found at: {weights_path}")
        print("Please train your model first or provide the correct path to weights using --weights.")
        sys.exit(1)

    # 3. Run YOLOv5 val.py command
    cmd = [
        sys.executable,
        str(YOLO_DIR / "val.py"),
        "--weights", str(weights_path),
        "--data", str(DATASET_YAML),
        "--task", args.task,
        "--project", str(BASE_DIR / "runs" / "val"),
        "--name", args.name,
        "--exist-ok"
    ]
    
    print(f"\nRunning test evaluation using YOLOv5 val.py: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 4. Summarize training plots and metrics
    print("\n" + "="*50)
    print("📋 EVALUATION & TRAINING PLOTS SUMMARY")
    print("="*50)
    
    # Look for training metrics and plots in the corresponding training folder
    train_run_dir = weights_path.parent.parent
    
    plots = {
        "Confusion Matrix": "confusion_matrix.png",
        "Precision-Recall (PR) Curve": "PR_curve.png",
        "F1-Confidence Curve": "F1_curve.png",
        "Precision-Confidence Curve": "P_curve.png",
        "Recall-Confidence Curve": "R_curve.png",
        "Training Metrics & Losses Plot": "results.png"
    }
    
    # Display plot locations
    print("\nVisual Evaluation Plots (automatically generated during training):")
    plots_found = False
    for label, filename in plots.items():
        plot_path = train_run_dir / filename
        if plot_path.exists():
            print(f"  - {label}: file:///{plot_path.resolve()}")
            plots_found = True
        else:
            print(f"  - {label}: Not found (was the model training finished?)")
            
    if not plots_found:
        print("  💡 Note: Training plots will be located in your runs/train/<run_name>/ folder after training completes.")
        
    # Parse results.csv if it exists to print final training metrics
    results_csv = train_run_dir / "results.csv"
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            # Standard columns in YOLOv5 results.csv can have leading/trailing spaces in headers
            df.columns = df.columns.str.strip()
            
            # Print last epoch metrics
            last_row = df.iloc[-1]
            print("\nFinal Metrics from Training Run (Last Epoch):")
            print(f"  - Epochs completed: {len(df)}")
            if "metrics/precision" in df.columns:
                print(f"  - Precision: {last_row['metrics/precision']:.4f}")
                print(f"  - Recall: {last_row['metrics/recall']:.4f}")
                print(f"  - mAP@0.5: {last_row['metrics/mAP_0.5']:.4f}")
                print(f"  - mAP@0.5:0.95: {last_row['metrics/mAP_0.5:0.95']:.4f}")
            else:
                # Old/other columns names
                print(f"  - mAP@0.5: {last_row.get('metrics/mAP_0.5', 'N/A')}")
                print(f"  - mAP@0.5:0.95: {last_row.get('metrics/mAP_0.5:0.95', 'N/A')}")
        except Exception as e:
            print(f"\nCould not parse training metrics from results.csv: {e}")
            
    print("\nEvaluation completed. Test evaluation outputs are stored in:")
    print(f"  runs/val/{args.name}/")
    print("="*50)

if __name__ == "__main__":
    main()
