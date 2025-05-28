import json
import os
import numpy as np
from pycocotools import mask as coco_mask
import cv2

# === Yol ayarları ===
coco_path = 'export_coco-instance_sezaiufuk_AIRCRAFT_GROUPED_v1.json'
output_dir = 'yolo_labels'
os.makedirs(output_dir, exist_ok=True)

# === JSON oku ===
with open(coco_path, 'r') as f:
    coco = json.load(f)

# COCO bileşenleri
images = {img["id"]: img for img in coco["images"]}
categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
annotations = coco["annotations"]

# Kategori isimlerini ID sırasına göre al
sorted_cat_ids = sorted(categories.keys())
cat_id_map = {id_: i for i, id_ in enumerate(sorted_cat_ids)}  # YOLO class ID

def normalize(points, width, height):
    # Convert all values to float
    points = [float(x) for x in points]
    # Create pairs of x,y coordinates
    return [(x / width, y / height) for x, y in zip(points[::2], points[1::2])]

def rle_to_polygon(rle, width, height):
    # Convert RLE to binary mask
    binary_mask = coco_mask.decode(rle)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of points
    points = []
    for point in approx_contour:
        x, y = point[0]
        points.extend([x, y])
    
    return points

# === Annotation'ları işle ===
from collections import defaultdict

image_anns = defaultdict(list)
for ann in annotations:
    image_anns[ann['image_id']].append(ann)

for img_id, anns in image_anns.items():
    img_info = images[img_id]
    width, height = img_info['width'], img_info['height']
    file_basename = os.path.splitext(img_info['file_name'])[0]
    txt_lines = []

    print(f"\nProcessing image {img_id} ({file_basename})")
    print(f"Image size: {width}x{height}")

    for ann in anns:
        if 'segmentation' not in ann or not ann['segmentation']:
            continue
            
        class_id = cat_id_map[ann['category_id']]
        print(f"\nAnnotation for class {class_id}")
        
        # Handle both polygon and RLE formats
        if isinstance(ann['segmentation'], dict):  # RLE format
            try:
                points = rle_to_polygon(ann['segmentation'], width, height)
                if points is None:
                    continue
                norm_points = normalize(points, width, height)
                flat = [f"{x:.6f} {y:.6f}" for x, y in norm_points]
                line = f"{class_id} " + " ".join(flat)
                txt_lines.append(line)
            except Exception as e:
                print(f"Error processing RLE segmentation: {e}")
                continue
        else:  # Polygon format
            for seg in ann['segmentation']:
                try:
                    norm_points = normalize(seg, width, height)
                    flat = [f"{x:.6f} {y:.6f}" for x, y in norm_points]
                    line = f"{class_id} " + " ".join(flat)
                    txt_lines.append(line)
                except Exception as e:
                    print(f"Error processing polygon segmentation: {e}")
                    continue

    if txt_lines:  # Only write if we have valid annotations
        output_path = os.path.join(output_dir, file_basename + '.txt')
        with open(output_path, 'w') as f:
            f.write("\n".join(txt_lines))
        print(f"Wrote {len(txt_lines)} annotations to {output_path}")