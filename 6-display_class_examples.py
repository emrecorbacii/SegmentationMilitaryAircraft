import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Class names from data.yaml
class_names = ['combat', 'uav', 'support', 'transport', 'helicopter', 'tiltrotor']

def get_class_ids_from_label(label_path):
    class_ids = set()
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_ids.add(class_id)
    return class_ids

def find_image_for_class(class_id, train_dir, label_dir):
    for img_file in train_dir.glob('*.jpg'):
        label_file = label_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        if int(line.split()[0]) == class_id:
                            return str(img_file)
    return None

def main():
    train_dir = Path('yolo_split/images/train')
    label_dir = Path('yolo_split/labels/train')
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # Find and display one image for each class
    for class_id, class_name in enumerate(class_names):
        img_path = find_image_for_class(class_id, train_dir, label_dir)
        
        if img_path:
            # Read and display image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[class_id].imshow(img)
            axes[class_id].set_title(f'Class {class_id} ({class_name})')
            axes[class_id].axis('off')
        else:
            axes[class_id].text(0.5, 0.5, f'No image found for {class_name}', ha='center', va='center')
            axes[class_id].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main() 