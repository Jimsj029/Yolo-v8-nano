import json
import os
import shutil
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

# Configuration
SOURCE_FOLDER = "train"  # Folder containing new images from Roboflow
COCO_FILE = os.path.join(SOURCE_FOLDER, "_annotations.coco.json")

# Split ratios (train:valid:test)
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# Create directories if they don't exist
for split in ['train', 'valid', 'test']:
    Path(f'images/{split}').mkdir(parents=True, exist_ok=True)
    Path(f'labels/{split}').mkdir(parents=True, exist_ok=True)

print("📂 Loading COCO annotations...")
# Load COCO annotations
with open(COCO_FILE, 'r') as f:
    coco_data = json.load(f)

# Create mappings
images_dict = {img['id']: img for img in coco_data['images']}
categories_dict = {cat['id']: cat for cat in coco_data['categories']}

print(f"   Found {len(images_dict)} images")
print(f"   Found {len(coco_data['annotations'])} annotations")
print(f"   Categories: {[cat['name'] for cat in coco_data['categories']]}")

# Group annotations by image_id
annotations_by_image = {}
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(ann)

def convert_segmentation_to_yolo(segmentation, img_width, img_height):
    """Convert COCO segmentation to YOLO format (normalized polygon)"""
    if isinstance(segmentation, dict):
        # RLE format - skip for now
        return None
    
    if isinstance(segmentation, list) and len(segmentation) > 0:
        points = segmentation[0] if isinstance(segmentation[0], list) else segmentation
        normalized_points = []
        
        for i in range(0, len(points), 2):
            x = points[i] / img_width
            y = points[i + 1] / img_height
            normalized_points.extend([x, y])
        
        return normalized_points
    
    return None

# Get all image IDs and shuffle them for random split
image_ids = list(images_dict.keys())
random.shuffle(image_ids)

# Calculate split indices
total_images = len(image_ids)
train_end = int(total_images * TRAIN_RATIO)
valid_end = train_end + int(total_images * VALID_RATIO)

# Split the image IDs
train_ids = image_ids[:train_end]
valid_ids = image_ids[train_end:valid_end]
test_ids = image_ids[valid_end:]

print(f"\n📊 Split distribution:")
print(f"   Train: {len(train_ids)} images ({len(train_ids)/total_images*100:.1f}%)")
print(f"   Valid: {len(valid_ids)} images ({len(valid_ids)/total_images*100:.1f}%)")
print(f"   Test:  {len(test_ids)} images ({len(test_ids)/total_images*100:.1f}%)")

# Create split mapping
split_mapping = {}
for img_id in train_ids:
    split_mapping[img_id] = 'train'
for img_id in valid_ids:
    split_mapping[img_id] = 'valid'
for img_id in test_ids:
    split_mapping[img_id] = 'test'

# Process each image
print(f"\n🔄 Processing images...")
processed = 0
skipped = 0

for image_id, image_info in images_dict.items():
    filename = image_info['file_name']
    img_width = image_info['width']
    img_height = image_info['height']
    
    # Get the assigned split
    split = split_mapping[image_id]
    
    # Source and destination paths
    source_img_path = Path(SOURCE_FOLDER) / filename
    dest_img_path = Path(f'images/{split}') / filename
    
    # Check if source image exists
    if not source_img_path.exists():
        print(f"   ⚠️  Image not found: {filename}")
        skipped += 1
        continue
    
    # Copy image to destination
    shutil.copy2(source_img_path, dest_img_path)
    
    # Get annotations for this image
    annotations = annotations_by_image.get(image_id, [])
    
    # Create YOLO label file
    label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    label_path = Path(f'labels/{split}') / label_filename
    
    # Write YOLO format labels
    with open(label_path, 'w') as f:
        for ann in annotations:
            # Get category ID (map to 0 for stitch_line since it's the only class)
            class_id = 0
            
            # Convert segmentation to YOLO format
            segmentation = ann['segmentation']
            normalized_points = convert_segmentation_to_yolo(segmentation, img_width, img_height)
            
            if normalized_points is None:
                continue
            
            # Write: class_id x1 y1 x2 y2 x3 y3 ...
            line = f"{class_id} " + " ".join([f"{p:.6f}" for p in normalized_points])
            f.write(line + '\n')
    
    processed += 1
    if processed % 10 == 0:
        print(f"   Processed {processed}/{total_images} images...")

print(f"\n✅ Dataset organization complete!")
print(f"   ✓ Processed: {processed} images")
print(f"   ✗ Skipped: {skipped} images")
print(f"\n📁 Directory structure:")
print(f"   images/train/   → {len(train_ids)} images")
print(f"   images/valid/   → {len(valid_ids)} images")
print(f"   images/test/    → {len(test_ids)} images")
print(f"   labels/train/   → {len(train_ids)} labels")
print(f"   labels/valid/   → {len(valid_ids)} labels")
print(f"   labels/test/    → {len(test_ids)} labels")
print(f"\n🚀 Ready to train! Run your training script now.")
