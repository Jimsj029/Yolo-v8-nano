from ultralytics import YOLO
import torch

print("🔧 Training Configuration")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
print()

# Load a pretrained YOLOv8 nano segmentation model
print("📦 Loading YOLOv8n-seg pretrained model...")
model = YOLO('yolov8n-seg.pt')

# Train the model
print("🚀 Starting training...")
print()

results = model.train(
    data='data.yaml',           # Path to data configuration file
    epochs=100,                 # Number of epochs to train
    imgsz=640,                  # Image size
    batch=16,                   # Batch size (adjust based on your GPU)
    name='train5',              # Name of the training run
    patience=20,                # Early stopping patience
    save=True,                  # Save checkpoints
    device=0,                   # Use GPU 0 (use 'cpu' if no GPU)
    workers=8,                  # Number of worker threads
    project='runs/segment',     # Project directory
    exist_ok=False,             # Don't overwrite existing project
    pretrained=True,            # Use pretrained weights
    optimizer='auto',           # Optimizer (SGD, Adam, AdamW, or auto)
    verbose=True,               # Verbose output
    seed=42,                    # Random seed for reproducibility
    deterministic=True,         # Deterministic training
    single_cls=True,            # Single class training (only stitch_line)
    rect=False,                 # Rectangular training
    cos_lr=False,               # Cosine learning rate scheduler
    close_mosaic=10,            # Disable mosaic augmentation in last N epochs
    amp=True,                   # Automatic Mixed Precision training
    fraction=1.0,               # Fraction of dataset to use
    profile=False,              # Profile ONNX and TensorRT speeds
    freeze=None,                # Freeze first N layers or specific layer indices
    # Learning rate settings
    lr0=0.01,                   # Initial learning rate
    lrf=0.01,                   # Final learning rate factor
    momentum=0.937,             # SGD momentum
    weight_decay=0.0005,        # Optimizer weight decay
    warmup_epochs=3.0,          # Warmup epochs
    warmup_momentum=0.8,        # Warmup momentum
    warmup_bias_lr=0.1,         # Warmup bias learning rate
    # Augmentation settings
    hsv_h=0.015,                # HSV hue augmentation
    hsv_s=0.7,                  # HSV saturation augmentation
    hsv_v=0.4,                  # HSV value augmentation
    degrees=0.0,                # Rotation degrees
    translate=0.1,              # Translation
    scale=0.5,                  # Scale
    shear=0.0,                  # Shear degrees
    perspective=0.0,            # Perspective
    flipud=0.0,                 # Flip up-down probability
    fliplr=0.5,                 # Flip left-right probability
    mosaic=1.0,                 # Mosaic augmentation probability
    mixup=0.0,                  # Mixup augmentation probability
    copy_paste=0.0,             # Copy-paste augmentation probability
)

print()
print("✅ Training complete!")
print(f"   Results saved to: runs/segment/train5")
print(f"   Best model: runs/segment/train5/weights/best.pt")
print(f"   Last model: runs/segment/train5/weights/last.pt")
print()
print("📊 To view results, check:")
print("   - runs/segment/train5/results.png")
print("   - runs/segment/train5/confusion_matrix.png")
print("   - runs/segment/train5/val_batch0_labels.jpg")
print("   - runs/segment/train5/val_batch0_pred.jpg")
print()
print("🎥 To test with webcam, update webcam_inference.py with:")
print("   model = YOLO('runs/segment/train5/weights/best.pt')")
