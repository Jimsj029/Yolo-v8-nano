"""
Export YOLOv8 Segmentation Model to Different Formats

This script exports your trained YOLOv8 model to various formats:
- PyTorch (.pt) - For Python projects with PyTorch
- TorchScript (.torchscript) - For C++ applications
- ONNX (.onnx) - For cross-platform inference (OpenCV, ONNX Runtime)
- TensorFlow Lite (.tflite) - For Raspberry Pi, Android, Edge devices
- TensorFlow SavedModel - For TensorFlow applications

Usage:
    python export_model.py
    
Then edit MODEL_PATH to point to your trained model.
"""

from ultralytics import YOLO
import os

# ===== CONFIGURATION =====
# Change this to your trained model path
MODEL_PATH = "runs/segment/train8_precise3/weights/best.pt"

# Image size for exported model (must match training size)
EXPORT_IMGSZ = 416

# Export formats (set to True for formats you want)
EXPORT_FORMATS = {
    'torchscript': False,   # Good for production, no Python dependency
    'onnx': True,           # Universal format, works with OpenCV
    'tflite': True,         # Best for Raspberry Pi / Mobile
    'saved_model': False,   # TensorFlow format
    'openvino': False,      # Intel hardware acceleration
}
# =========================

def main():
    print("=" * 70)
    print("YOLOv8 Model Export Tool")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("\nAvailable models:")
        runs_dir = "runs/segment"
        if os.path.exists(runs_dir):
            for folder in os.listdir(runs_dir):
                weights_path = os.path.join(runs_dir, folder, "weights", "best.pt")
                if os.path.exists(weights_path):
                    print(f"  ✓ {weights_path}")
        return
    
    print(f"📦 Loading model: {MODEL_PATH}\n")
    model = YOLO(MODEL_PATH)
    
    # Get export directory
    model_dir = os.path.dirname(MODEL_PATH)
    print(f"📁 Export directory: {model_dir}\n")
    
    # Export to selected formats
    exported_files = []
    
    if EXPORT_FORMATS['torchscript']:
        print("🔄 Exporting to TorchScript...")
        model.export(format='torchscript', imgsz=EXPORT_IMGSZ)
        exported_files.append(MODEL_PATH.replace('.pt', '.torchscript'))
    
    if EXPORT_FORMATS['onnx']:
        print("🔄 Exporting to ONNX...")
        model.export(format='onnx', imgsz=EXPORT_IMGSZ, simplify=True)
        exported_files.append(MODEL_PATH.replace('.pt', '.onnx'))
    
    if EXPORT_FORMATS['tflite']:
        print("🔄 Exporting to TensorFlow Lite (for Raspberry Pi)...")
        model.export(format='tflite', imgsz=EXPORT_IMGSZ, int8=False)
        exported_files.append(MODEL_PATH.replace('.pt', '_saved_model').replace('.pt', '_float16.tflite'))
    
    if EXPORT_FORMATS['saved_model']:
        print("🔄 Exporting to TensorFlow SavedModel...")
        model.export(format='saved_model', imgsz=EXPORT_IMGSZ)
        exported_files.append(MODEL_PATH.replace('.pt', '_saved_model'))
    
    if EXPORT_FORMATS['openvino']:
        print("🔄 Exporting to OpenVINO...")
        model.export(format='openvino', imgsz=EXPORT_IMGSZ)
        exported_files.append(MODEL_PATH.replace('.pt', '_openvino_model'))
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Export Complete!")
    print("=" * 70)
    print(f"\nOriginal model: {MODEL_PATH}")
    print("\nExported files:")
    for file in exported_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 70)
    print("HOW TO USE IN OTHER PROJECTS:")
    print("=" * 70)
    
    if EXPORT_FORMATS['onnx']:
        print("\n📌 ONNX (works with OpenCV, ONNX Runtime):")
        print("""
import cv2
import numpy as np

# Load ONNX model with OpenCV
net = cv2.dnn.readNetFromONNX('best.onnx')

# Prepare image
img = cv2.imread('image.jpg')
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True)
net.setInput(blob)

# Run inference
outputs = net.forward()
""")
    
    if EXPORT_FORMATS['tflite']:
        print("\n📌 TFLite (Raspberry Pi, Android):")
        print("""
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='best_float16.tflite')
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare image (416x416, normalized 0-1)
img = cv2.imread('image.jpg')
img = cv2.resize(img, (416, 416))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Run inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
""")
    
    print("\n📌 PyTorch .pt (Python + Ultralytics):")
    print("""
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run inference
results = model('image.jpg')

# Get results
for result in results:
    boxes = result.boxes      # Bounding boxes
    masks = result.masks      # Segmentation masks
    result.show()             # Display
    result.save('output.jpg') # Save
""")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
