from ultralytics import YOLO
import cv2
import time

# Load the trained model (using the latest trained model)
model = YOLO('runs/segment/train4/weights/best.pt')

# Open webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# FPS limiting
target_fps = 10
frame_time = 1.0 / target_fps  # Time between frames (0.1 seconds for 10 FPS)
last_frame_time = time.time()

print("Press 'q' to quit")
print(f"Running at {target_fps} FPS")

while True:
    # FPS limiting - wait until enough time has passed
    current_time = time.time()
    elapsed = current_time - last_frame_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)
    last_frame_time = time.time()
    
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Run YOLOv8 segmentation inference on the frame
    results = model(frame, conf=0.3)  # conf=0.5 means 50% confidence threshold
    
    # Visualize the results on the frame (hide boxes and labels, show only masks)
    annotated_frame = results[0].plot(boxes=False, labels=False)
    
    # Display the frame
    cv2.imshow('YOLOv8 Segmentation - Stitch Line Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
