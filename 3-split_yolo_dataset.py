import os
import shutil
import random
from collections import defaultdict

yaml_template = """
train: {train}
val: {val}
test: {test}
nc: {nc}
names: {names}
"""

# Klasörler
img_dir = 'segments/sezaiufuk_AIRCRAFT_GROUPED/v1'
label_dir = 'yolo_labels'
out_dir = 'yolo_split'

# Tam (absolute) çalışma dizini
current_dir = os.path.abspath(os.getcwd())

# Çıktı klasörlerini oluştur
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

# Her sınıf için dosyaları grupla
class_files = defaultdict(list)
for img_file in os.listdir(img_dir):
    if not img_file.endswith('.jpg'):
        continue
    
    base = img_file.split('.')[0]
    label_file = base + '.txt'
    label_path = os.path.join(label_dir, label_file)
    
    if not os.path.exists(label_path):
        continue
        
    # Label dosyasından sınıf ID'lerini al
    with open(label_path, 'r') as f:
        class_ids = set(line.split()[0] for line in f)
        
    # Her sınıf için dosyayı ekle
    for class_id in class_ids:
        class_files[class_id].append((img_file, label_file))

print("Sınıf başına örnek sayıları:")
for class_id, files in class_files.items():
    print(f"Sınıf {class_id}: {len(files)} örnek")

# Tüm dosyaları karıştır
all_files = []
for files in class_files.values():
    all_files.extend(files)
random.shuffle(all_files)

# Her sınıf için dosyaları yeniden grupla
shuffled_class_files = defaultdict(list)
for img_file, label_file in all_files:
    with open(os.path.join(label_dir, label_file), 'r') as f:
        class_ids = set(line.split()[0] for line in f)
    for class_id in class_ids:
        shuffled_class_files[class_id].append((img_file, label_file))

# Her sınıf için ayrı split yap
train_pairs = []
val_pairs = []
test_pairs = []

for class_id, files in shuffled_class_files.items():
    n = len(files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    train_pairs.extend(files[:n_train])
    val_pairs.extend(files[n_train:n_train+n_val])
    test_pairs.extend(files[n_train+n_val:])

print(f"\nSplit sonrası örnek sayıları:")
print(f"Train seti: {len(train_pairs)} görüntü")
print(f"Val seti: {len(val_pairs)} görüntü")
print(f"Test seti: {len(test_pairs)} görüntü")

# Dosyaları kopyala
splits = [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]
for split, split_pairs in splits:
    for img_file, label_file in split_pairs:
        shutil.copy(os.path.join(img_dir, img_file), os.path.join(out_dir, 'images', split, img_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(out_dir, 'labels', split, label_file))

# Sınıf isimlerini al
class_names = sorted(list(shuffled_class_files.keys()), key=lambda x: int(x))

# Sınıf isimleri sözlüğünü oluştur
names_dict_str = ""
class_name_map = {
    '0': 'combat',
    '1': 'uav',
    '2': 'support',
    '3': 'transport',
    '4': 'helicopter',
    '5': 'tiltrotor'
}

for class_id in class_names:
    if class_id in class_name_map:
        names_dict_str += f"  {class_id}: {class_name_map[class_id]}\n"
    else:
        names_dict_str += f"  {class_id}: class_{class_id}\n"

# Tam yolları oluştur
train_path = os.path.join(current_dir, out_dir, 'images', 'train').replace('\\', '/')
val_path = os.path.join(current_dir, out_dir, 'images', 'val').replace('\\', '/')
test_path = os.path.join(current_dir, out_dir, 'images', 'test').replace('\\', '/')

# data.yaml oluştur
yaml_path = os.path.join(out_dir, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_template.format(
        train=train_path,
        val=val_path,
        test=test_path,
        nc=len(class_names),
        names_dict=names_dict_str.rstrip()
    ))
print(f"\ndata.yaml oluşturuldu: {yaml_path}")

# Her split için sınıf dağılımını göster
print("\nSplit'lerdeki sınıf dağılımları:")
for split_name, split_pairs in splits:
    class_counts = defaultdict(int)
    for _, label_file in split_pairs:
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                class_id = line.split()[0]
                class_counts[class_id] += 1
    
    print(f"\n{split_name.upper()} seti sınıf dağılımı:")
    for class_id in sorted(class_counts.keys(), key=lambda x: int(x)):
        print(f"Sınıf {class_id}: {class_counts[class_id]} örnek") 