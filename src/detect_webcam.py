import os
import argparse
import time
import cv2
import torch
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_DIR = BASE_DIR / "yolov5"

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Custom Real-time Webcam Inference")
    parser.add_argument("--source", type=int, default=0, help="Webcam source index (default: 0)")
    parser.add_argument("--weights", type=str, default="runs/train/shapes_model/weights/best.pt", help="Path to model weights (.pt file)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # 1. Check if YOLOv5 dir exists
    if not YOLO_DIR.exists():
        print("YOLOv5 repository not found. Please run src/setup_yolov5.py first.")
        sys.exit(1)
        
    # 2. Check weights path
    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Fallback to yolov5s.pt from ultralytics if custom weights don't exist
        print(f"Weights not found at {weights_path}.")
        print("Falling back to yolov5s.pt (COCO pretrained weights)...")
        weights_path = YOLO_DIR / "yolov5s.pt"
        if not weights_path.exists():
            print("Pretrained weights yolov5s.pt not found, downloading...")
            
    # 3. Load YOLOv5 model
    print(f"Loading YOLOv5 model from {weights_path} using local YOLOv5 source...")
    try:
        model = torch.hub.load(str(YOLO_DIR), 'custom', path=str(weights_path), source='local')
        model.conf = args.conf  # NMS confidence threshold
    except Exception as e:
        print(f"Failed to load YOLOv5 model: {e}")
        print("If you haven't run setup or training yet, please run them first.")
        sys.exit(1)

    # 4. Open webcam
    print(f"Opening webcam source: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open webcam source: {args.source}")
        print("Please verify that the webcam is connected and not in use by another app.")
        sys.exit(1)
        
    # Configure webcam properties for better speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nPress 'q' key in the video window to quit.")
    
    prev_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
            
        # Convert BGR (OpenCV) to RGB (YOLOv5)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = model(img_rgb)
        
        # Parse results: xmin, ymin, xmax, ymax, confidence, class, name
        detections = results.pandas().xyxy[0]
        
        # Draw bounding boxes and labels
        for _, det in detections.iterrows():
            xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            conf = det['confidence']
            class_name = det['name']
            
            # Draw box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Label string
            label = f"{class_name} {conf:.2f}"
            
            # Text size
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Draw background label rectangle
            cv2.rectangle(frame, (xmin, ymin - 20), (xmin + w, ymin), (0, 255, 0), -1)
            # Draw text
            cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Draw FPS on the screen
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display the frame
        cv2.imshow("YOLOv5 Real-Time Webcam Detection", frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Inference ended by user.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
