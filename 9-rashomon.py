import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import torch
from PIL import Image
import os

def get_all_model_masks(model_paths, img_path, resize_shape=(512, 512), threshold=0.5):
    """
    Verilen model yolları ve bir görsel için, her modelin mask tahminini döndürür.
    Çıktı: List of birleşik mask numpy array'leri, her biri [H, W] boyutunda.
    """
    masks = []
    for model_path in model_paths:
        model = YOLO(model_path, task='segment')
        img = Image.open(img_path).convert("RGB")
        img = img.resize(resize_shape)
        results = model(img)
        if (results[0].masks is None or
            results[0].masks.data is None or
            len(results[0].masks.data) == 0):
            mask = np.zeros(resize_shape, dtype=np.float32)
        else:
            # Tüm maskeleri birleştir (mantıksal OR)
            mask_tensor = results[0].masks.data  # [N, H, W]
            mask = torch.any(mask_tensor > threshold, dim=0).cpu().numpy().astype(np.float32)
        masks.append(mask)
    return masks

def M_PDA(model_paths, img_dir, resize_shape=(512, 512), threshold=0.5):

    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_models = len(model_paths)
    conflict_matrix = np.zeros((n_models, n_models), dtype=np.float32)
    conflict_counts = [[[] for _ in range(n_models)] for _ in range(n_models)]

    print("Başlangıçta model_paths:", model_paths)
    print("Başlangıçta n_models:", n_models)

    for img_path in img_files:
        masks = get_all_model_masks(model_paths, img_path, resize_shape=resize_shape)
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    continue
                mask1 = masks[i] > threshold
                mask2 = masks[j] > threshold
                conflict_pixel_count = np.logical_xor(mask1, mask2).sum()
                conflict_counts[i][j].append(conflict_pixel_count)

    total_pixels = len(img_files) * resize_shape[0] * resize_shape[1]  # Tüm görsellerdeki toplam piksel sayısı

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                conflict_matrix[i, j] = 0
            else:
                total_conflict = sum(conflict_counts[i][j])
                conflict_matrix[i, j] = total_conflict / total_pixels  # Piksel başına düşen conflict oranı

    print("conflict_matrix.shape:", conflict_matrix.shape)

    return conflict_matrix

# Kullanım:
model_paths = [
    'models/v1-11n-seg/v1-11n-seg/weights/best.pt',
    'models/v2-8n-seg/v2-8n-seg/weights/best.pt',
    'models/v3-9c-seg/v3-9c-seg/weights/best.pt',
    'models/v4-8n-seg/v4-8n-seg/weights/best.pt'
]

"""
img_path = 'yolo_split/images/test/99_combat.jpg'

masks = get_all_model_masks(model_paths, img_path, resize_shape=(512, 512))

for i, mask in enumerate(masks):
    print(f"Model {i+1} mask shape: {mask.shape}")
    # İstersen burada maskeleri görselleştirebilirsin
    plt.figure()
    plt.imshow(mask, cmap='Greens')
    plt.title(f"Model {i+1} Mask")
    plt.axis('off')
plt.show()
"""

img_dir = 'yolo_split/images/test'

m_pda = M_PDA(model_paths, img_dir, resize_shape=(512, 512), threshold=0.5)
print("M_PDA")
print("m_pda.shape:", m_pda.shape)
print(m_pda)

v_pd = np.max(m_pda, axis=1)

print(v_pd)

lmbd = np.argmin(v_pd)

print(f"Model with lowest pixel discrepancy ({v_pd[lmbd]}) is {model_paths[lmbd]}")

os.makedirs('rashomon', exist_ok=True)
np.savetxt('rashomon/m_pda.txt', m_pda, fmt='%1.4f')

np.savetxt('rashomon/v_pd.txt', v_pd, fmt='%1.4f')
with open('rashomon/v_pd.txt', 'a') as f:
    f.write(f"Model with lowest pixel discrepancy ({v_pd[lmbd]}) is {model_paths[lmbd]}")














