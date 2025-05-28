import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def read_yolo_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]

def create_mask_from_yolo(image_shape, annotations):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    h, w = image_shape[:2]
    
    for ann in annotations:
        class_id = int(ann[0])
        points = [float(x) for x in ann[1:]]
        pixel_points = []
        for i in range(0, len(points), 2):
            x = int(points[i] * w)
            y = int(points[i+1] * h)
            pixel_points.append([x, y])
        points = np.array(pixel_points, dtype=np.int32)
        cv2.fillPoly(mask, [points], class_id)
    return mask

def overlay_mask(image, mask, class_names):
    colored_mask = np.zeros_like(image)
    unique_classes = np.unique(mask)
    colors = {
        0: [70, 130, 180],    # Blue for combat
        1: [205, 92, 92],     # Brownish for uav
        2: [218, 165, 32],    # Gold for support
        3: [128, 0, 128],     # Purple for transport
        4: [107, 142, 35],    # Green for helicopter
        5: [139, 0, 0],       # Dark red for tiltrotor
    }
    for class_id in unique_classes:
        if class_id == 0:
            continue
        color = colors.get(class_id, [255, 255, 255])
        colored_mask[mask == class_id] = color
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return overlay, unique_classes

# Sınıf isimleri (görseldeki gibi)
class_names = {
    0: "combat",
    1: "uav",
    2: "support",
    3: "transport",
    4: "helicopter",
    5: "tiltrotor"
}

base_dir = 'yolo_split'
train_img_dir = os.path.join(base_dir, 'images', 'train')
train_label_dir = os.path.join(base_dir, 'labels', 'train')

image_pairs = []
for img_file in os.listdir(train_img_dir):
    if img_file.endswith('.jpg') and not img_file.endswith('_aug.jpg'):
        base_name = os.path.splitext(img_file)[0]
        aug_img = f"{base_name}_aug.jpg"
        if os.path.exists(os.path.join(train_img_dir, aug_img)):
            image_pairs.append((img_file, aug_img))

if image_pairs:
    orig_img, aug_img = random.choice(image_pairs)
    orig_image = cv2.imread(os.path.join(train_img_dir, orig_img))
    orig_annotations = read_yolo_label(os.path.join(train_label_dir, orig_img.replace('.jpg', '.txt')))
    orig_mask = create_mask_from_yolo(orig_image.shape, orig_annotations)
    orig_overlay, orig_classes = overlay_mask(orig_image, orig_mask, class_names)
    aug_image = cv2.imread(os.path.join(train_img_dir, aug_img))
    aug_annotations = read_yolo_label(os.path.join(train_label_dir, aug_img.replace('.jpg', '.txt')))
    aug_mask = create_mask_from_yolo(aug_image.shape, aug_annotations)
    aug_overlay, aug_classes = overlay_mask(aug_image, aug_mask, class_names)
    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(orig_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal + Mask Overlay')
    plt.axis('off')
    orig_text = "\n".join([f"{cid}: {class_names.get(cid, str(cid))}" for cid in orig_classes if cid != 0])
    plt.xlabel(orig_text, fontsize=12)
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(aug_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Augment + Mask Overlay')
    plt.axis('off')
    aug_text = "\n".join([f"{cid}: {class_names.get(cid, str(cid))}" for cid in aug_classes if cid != 0])
    plt.xlabel(aug_text, fontsize=12)
    plt.tight_layout()
    plt.show()
    print(f"Gösterilen görüntü çifti: {orig_img} - {aug_img}")
else:
    print("Karşılaştırılacak görüntü çifti bulunamadı!") 