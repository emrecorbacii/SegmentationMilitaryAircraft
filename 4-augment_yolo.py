import os
import cv2
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

def read_yolo_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]

def write_yolo_label(label_path, annotations):
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(' '.join(ann) + '\n')

def augment_image(image, annotations):
    # Rastgele augmentasyon seç
    augmentations = [
        horizontal_flip,
        vertical_flip,
        rotate,
        adjust_brightness,
        adjust_contrast
    ]
    
    # En az 1, en fazla 3 augmentasyon uygula
    n_augs = random.randint(1, 3)
    selected_augs = random.sample(augmentations, n_augs)
    
    for aug_func in selected_augs:
        image, annotations = aug_func(image, annotations)
    
    return image, annotations

def horizontal_flip(image, annotations):
    h, w = image.shape[:2]
    image = cv2.flip(image, 1)
    
    new_annotations = []
    for ann in annotations:
        class_id = ann[0]
        points = [float(x) for x in ann[1:]]
        
        # Her x koordinatını tersine çevir
        for i in range(0, len(points), 2):
            points[i] = 1.0 - points[i]
        
        new_ann = [class_id] + [str(x) for x in points]
        new_annotations.append(new_ann)
    
    return image, new_annotations

def vertical_flip(image, annotations):
    h, w = image.shape[:2]
    image = cv2.flip(image, 0)
    
    new_annotations = []
    for ann in annotations:
        class_id = ann[0]
        points = [float(x) for x in ann[1:]]
        
        # Her y koordinatını tersine çevir
        for i in range(1, len(points), 2):
            points[i] = 1.0 - points[i]
        
        new_ann = [class_id] + [str(x) for x in points]
        new_annotations.append(new_ann)
    
    return image, new_annotations

def rotate(image, annotations):
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    new_annotations = []
    for ann in annotations:
        class_id = ann[0]
        points = [float(x) for x in ann[1:]]
        new_points = []
        for i in range(0, len(points), 2):
            x, y = points[i] * w, points[i+1] * h
            xy = np.array([x, y, 1.0])
            x_new, y_new = np.dot(M, xy)
            # Normalize ve sınırla
            x_norm = min(max(x_new / w, 0.0), 1.0)
            y_norm = min(max(y_new / h, 0.0), 1.0)
            new_points.extend([x_norm, y_norm])
        new_ann = [class_id] + [str(x) for x in new_points]
        new_annotations.append(new_ann)
    return image, new_annotations

def adjust_brightness(image, annotations):
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image, annotations

def adjust_contrast(image, annotations):
    alpha = random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return image, annotations

def get_class_distribution(label_dir):
    class_counts = defaultdict(int)
    for label_file in os.listdir(label_dir):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                class_id = line.split()[0]
                class_counts[class_id] += 1
    return class_counts

# Ana klasörler
base_dir = 'yolo_split'
train_img_dir = os.path.join(base_dir, 'images', 'train')
train_label_dir = os.path.join(base_dir, 'labels', 'train')

# Başlangıç dağılımını al
print("\nBaşlangıç sınıf dağılımı:")
initial_dist = get_class_distribution(train_label_dir)
for class_id in sorted(initial_dist.keys(), key=lambda x: int(x)):
    print(f"Sınıf {class_id}: {initial_dist[class_id]} örnek")

# Tüm görüntüleri al
all_images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
n_images = len(all_images)

# Augment edilecek görüntü sayısını belirle (örneğin %50)
augment_percentage = 50  # Yüzde olarak
n_to_augment = int(n_images * augment_percentage / 100)

# Rastgele görüntüleri seç
images_to_augment = random.sample(all_images, n_to_augment)

print(f"\nToplam görüntü sayısı: {n_images}")
print(f"Augment edilecek görüntü sayısı: {n_to_augment} (%{augment_percentage})")

# Seçilen görüntüleri augment et
for img_file in images_to_augment:
    base_name = os.path.splitext(img_file)[0]
    label_file = base_name + '.txt'
    
    # Orijinal dosyaları oku
    img_path = os.path.join(train_img_dir, img_file)
    label_path = os.path.join(train_label_dir, label_file)
    
    image = cv2.imread(img_path)
    annotations = read_yolo_label(label_path)
    
    # Augment et
    aug_image, aug_annotations = augment_image(image.copy(), annotations.copy())
    
    # Augment edilmiş dosyaları kaydet
    aug_img_name = f"{base_name}_aug.jpg"
    aug_label_name = f"{base_name}_aug.txt"
    
    cv2.imwrite(os.path.join(train_img_dir, aug_img_name), aug_image)
    write_yolo_label(os.path.join(train_label_dir, aug_label_name), aug_annotations)
    
    print(f"Augmented: {aug_img_name}")

# Son dağılımı al ve göster
print("\nSon sınıf dağılımı:")
final_dist = get_class_distribution(train_label_dir)
for class_id in sorted(final_dist.keys(), key=lambda x: int(x)):
    initial = initial_dist[class_id]
    final = final_dist[class_id]
    increase = final - initial
    print(f"Sınıf {class_id}: {initial} -> {final} örnek (+{increase})")

print("\nAugmentation tamamlandı!") 