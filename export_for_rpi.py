"""
Export YOLOv8 model for Raspberry Pi deployment
Exports to multiple formats for optimal RPi performance
"""

from ultralytics import YOLO
import os

# Model to export (train7_rpi - Best model for RPi)
MODEL_PATH = 'runs/segment/train7_rpi/weights/best.pt'

print("=" * 60)
print("YOLOv8 Model Export for Raspberry Pi")
print("=" * 60)

# Load the trained model
print(f"\n📦 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Export formats for Raspberry Pi
export_formats = {
    'ONNX': 'onnx',           # Good CPU performance
    'TensorFlow Lite': 'tflite',  # Best for RPi, mobile devices
    'TensorFlow SavedModel': 'saved_model',  # Alternative TF format
    'TorchScript': 'torchscript',  # PyTorch optimized
}

print("\n🚀 Exporting model to multiple formats...")
print("-" * 60)

exported_models = {}

for format_name, format_type in export_formats.items():
    try:
        print(f"\n📤 Exporting to {format_name}...")
        export_path = model.export(
            format=format_type,
            imgsz=416,  # Same size as training for RPi
            simplify=True,  # Simplify ONNX model
            optimize=True,  # Optimize for inference
        )
        exported_models[format_name] = export_path
        print(f"   ✅ {format_name} export successful!")
        print(f"   📁 Saved to: {export_path}")
    except Exception as e:
        print(f"   ❌ {format_name} export failed: {str(e)}")

print("\n" + "=" * 60)
print("Export Summary")
print("=" * 60)

for format_name, path in exported_models.items():
    if os.path.exists(str(path)):
        size_mb = os.path.getsize(str(path)) / (1024 * 1024)
        print(f"✓ {format_name:20s}: {size_mb:.2f} MB")

print("\n" + "=" * 60)
print("Raspberry Pi Deployment Recommendations")
print("=" * 60)
print("""
1. Best Performance: TensorFlow Lite (.tflite)
   - Optimized for ARM processors
   - Smallest model size
   - Fastest inference on RPi
   
   Usage on RPi:
   ```python
   from ultralytics import YOLO
   model = YOLO('best.tflite')
   results = model('image.jpg')
   ```

2. Alternative: ONNX (.onnx)
   - Good CPU performance
   - Cross-platform compatibility
   
   Requires: pip install onnxruntime
   
   Usage on RPi:
   ```python
   model = YOLO('best.onnx')
   results = model('image.jpg')
   ```

3. Hardware Acceleration (Optional):
   - For Coral TPU: Export to Edge TPU format
   - For Intel NCS: Use OpenVINO format
   
Expected Performance on Raspberry Pi 4:
- TFLite: ~3-7 FPS at 416x416
- ONNX: ~2-5 FPS at 416x416
- PyTorch (.pt): ~1-3 FPS at 416x416

Tips for better FPS on RPi:
1. Use TFLite format
2. Lower resolution (320x320 instead of 416x416)
3. Increase confidence threshold (0.4-0.5)
4. Enable threading in OpenCV
5. Consider hardware accelerators (Coral TPU)
""")

print("=" * 60)
print("✅ Export complete!")
print("=" * 60)
