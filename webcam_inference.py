from ultralytics import YOLO
import cv2
import time

# Load the trained model (RPi-optimized version)
# train7_rpi: 74% mAP50, 76.9% precision, optimized for RPi
model = YOLO('runs/segment/train7_rpi/weights/best.pt')

# Open webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Optimize for Raspberry Pi - lower resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Try 416 if too slow on RPi
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Try 416 if too slow on RPi
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# FPS tracking
fps_start_time = time.time()
fps_counter = 0
display_fps = 0

print("=" * 50)
print("Real-Time Stitch Line Detection (RPi-Optimized)")
print("=" * 50)
print("Controls:")
print("  'q' - Quit")
print("  's' - Save current frame")
print("  '+' - Increase confidence (reduce false positives)")
print("  '-' - Decrease confidence (more sensitive)")
print("=" * 50)

frame_count = 0
confidence_threshold = 0.35  # Higher threshold to reduce finger detection

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Run YOLOv8 segmentation inference
    # Higher confidence to avoid detecting fingers, higher IOU for better filtering
    results = model(frame, conf=confidence_threshold, iou=0.6, imgsz=416, verbose=False)
    
    # Visualize the results on the frame (hide boxes and labels, show only masks)
    annotated_frame = results[0].plot(boxes=False, labels=False)
    
    # Calculate and display FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        display_fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    
    # Add FPS and confidence info to frame
    cv2.putText(annotated_frame, f'FPS: {display_fps} | Conf: {confidence_threshold:.2f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('YOLOv8 Segmentation - Stitch Line Detection', annotated_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save current frame
        filename = f'stitch_detection_{frame_count}.jpg'
        cv2.imwrite(filename, annotated_frame)
        print(f"Frame saved as {filename}")
    elif key == ord('+') or key == ord('='):
        # Increase confidence threshold (reduce false positives)
        confidence_threshold = min(0.9, confidence_threshold + 0.05)
        print(f"Confidence threshold increased to {confidence_threshold:.2f}")
    elif key == ord('-') or key == ord('_'):
        # Decrease confidence threshold (more sensitive)
        confidence_threshold = max(0.1, confidence_threshold - 0.05)
        print(f"Confidence threshold decreased to {confidence_threshold:.2f}")
    
    frame_count += 1

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
