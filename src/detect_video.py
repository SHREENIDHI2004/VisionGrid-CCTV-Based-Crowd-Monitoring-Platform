import os
import argparse
import cv2
import torch
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_DIR = BASE_DIR / "yolov5"

def main():
    parser = argparse.ArgumentParser(description="YOLOv5 Custom Video Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--weights", type=str, default="runs/train/shapes_model/weights/best.pt", help="Path to model weights (.pt file)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="runs/detect/output_video.mp4", help="Path to save output video")
    parser.add_argument("--no-show", action="store_true", help="Do not display the video frame during inference")
    
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

    # 4. Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing Video: {args.source} ({width}x{height} @ {fps} fps, {total_frames} frames)")
    
    # Setup video writer
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}...", end="\r")
        
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
            
        # Write frame to output video
        out.write(frame)
        
        # Display the frame if not disabled
        if not args.no_show:
            cv2.imshow("YOLOv5 Video Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nInference interrupted by user.")
                break
                
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete! Output saved to: {output_path}")

if __name__ == "__main__":
    main()
