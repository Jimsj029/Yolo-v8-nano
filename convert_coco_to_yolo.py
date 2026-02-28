import json
import os
from pathlib import Path

# Load COCO annotations
with open('_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# Create a mapping of image_id to file info
images_dict = {img['id']: img for img in coco_data['images']}

# Filter annotations for only stitch_line (category_id = 1)
stitch_line_annotations = [ann for ann in coco_data['annotations'] if ann['category_id'] == 1]

# Group annotations by image_id
annotations_by_image = {}
for ann in stitch_line_annotations:
    image_id = ann['image_id']
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(ann)

# Convert segmentation annotations to YOLO format
def convert_segmentation_to_yolo(segmentation, img_width, img_height):
    """Convert COCO segmentation to YOLO format (normalized polygon)"""
    # Handle both list and dict formats
    if isinstance(segmentation, dict):
        # RLE format - skip for now, only handle polygon
        return None
    
    # segmentation is a list of polygons
    if isinstance(segmentation, list) and len(segmentation) > 0:
        points = segmentation[0] if isinstance(segmentation[0], list) else segmentation
        normalized_points = []
        
        for i in range(0, len(points), 2):
            x = points[i] / img_width
            y = points[i + 1] / img_height
            normalized_points.extend([x, y])
        
        return normalized_points
    
    return None

# Find which split each image belongs to
def find_image_split(filename):
    """Determine which split (train/valid/test) an image belongs to"""
    for split in ['train', 'valid', 'test']:
        img_path = Path(f'images/{split}/{filename}')
        if img_path.exists():
            return split
    return None

# Process each image and create YOLO label files
processed = 0
skipped = 0

for image_id, image_info in images_dict.items():
    filename = image_info['file_name']
    img_width = image_info['width']
    img_height = image_info['height']
    
    # Find which split this image belongs to
    split = find_image_split(filename)
    
    if split is None:
        skipped += 1
        continue
    
    # Get annotations for this image
    annotations = annotations_by_image.get(image_id, [])
    
    # Create label file path
    label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
    label_path = Path(f'images/{split}/{label_filename}')
    
    # Write YOLO format labels
    with open(label_path, 'w') as f:
        for ann in annotations:
            # Class ID: 0 (since we're only using stitch_line, it becomes class 0)
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

print(f"✅ Conversion complete!")
print(f"   Processed: {processed} images")
print(f"   Skipped: {skipped} images (not found in train/valid/test)")
print(f"   Using only 'stitch_line' annotations (class 0)")
