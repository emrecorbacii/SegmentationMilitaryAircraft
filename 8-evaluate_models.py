import os
import yaml
import shutil
from ultralytics import YOLO

# Modelleri değerlendir
def evaluate_models():
    # Mevcut data.yaml dosyasını kullan
    data_yaml = os.path.join(os.getcwd(), 'yolo_split', 'data.yaml')
    
    # Test sonuçlarını kaydetmek için dizin oluştur
    tests_dir = os.path.join(os.getcwd(), 'tests')
    os.makedirs(tests_dir, exist_ok=True)
    
    # Modelleri bul ve değerlendir
    models_dir = os.path.join(os.getcwd(), 'models')
    
    for model_dir in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_dir, model_dir)
        
        # Sadece dizin ise işlem yap
        if os.path.isdir(model_path):
            # Model weights klasörünü kontrol et
            weights_dir = os.path.join(model_path, 'weights')
            if os.path.isdir(weights_dir):
                # En son ağırlık dosyasını bul (.pt uzantılı)
                pt_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
                
                if pt_files:
                    # Sıralama yaparak en son versiyonu al (best.pt veya last.pt)
                    pt_files.sort()
                    if 'best.pt' in pt_files:
                        model_pt = os.path.join(weights_dir, 'best.pt')
                    else:
                        model_pt = os.path.join(weights_dir, pt_files[-1])
                    
                    print(f"Model değerlendiriliyor: {model_pt}")
                    
                    # Test çıktıları için dizin oluştur
                    model_test_dir = os.path.join(tests_dir, model_dir)
                    os.makedirs(model_test_dir, exist_ok=True)
                    
                    # Modeli yükle ve değerlendir
                    try:
                        model = YOLO(model_pt)
                        # Değerlendirme yaparken 'test' split'ini kullan
                        results = model.val(data=data_yaml, split='test', project=model_test_dir, name='evaluate')
                        
                        # Değerlendirme sonuçlarını kaydet
                        with open(os.path.join(model_test_dir, 'results.txt'), 'w') as f:
                            f.write(f"Model: {model_dir}\n")
                            f.write(f"mAP50: {results.box.map50}\n")
                            f.write(f"mAP50-95: {results.box.map}\n")
                            if hasattr(results, 'seg'):
                                f.write(f"Segmentation mAP50: {results.seg.map50}\n")
                                f.write(f"Segmentation mAP50-95: {results.seg.map}\n")
                        
                        print(f"{model_dir} model değerlendirmesi tamamlandı.")
                    except Exception as e:
                        print(f"Hata oluştu {model_dir} değerlendirilirken: {str(e)}")
                else:
                    print(f"Uyarı: {weights_dir} içinde .pt dosyası bulunamadı.")
            else:
                # weights klasörü yoksa, doğrudan .pt dosyalarını ara
                pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
                
                if pt_files:
                    # Sıralama yaparak en son versiyonu al
                    pt_files.sort()
                    if 'best.pt' in pt_files:
                        model_pt = os.path.join(model_path, 'best.pt')
                    else:
                        model_pt = os.path.join(model_path, pt_files[-1])
                    
                    print(f"Model değerlendiriliyor: {model_pt}")
                    
                    # Test çıktıları için dizin oluştur
                    model_test_dir = os.path.join(tests_dir, model_dir)
                    os.makedirs(model_test_dir, exist_ok=True)
                    
                    # Modeli yükle ve değerlendir
                    try:
                        model = YOLO(model_pt)
                        # Değerlendirme yaparken 'test' split'ini kullan
                        results = model.val(data=data_yaml, split='test', project=model_test_dir, name='evaluate')
                        
                        # Değerlendirme sonuçlarını kaydet
                        with open(os.path.join(model_test_dir, 'results.txt'), 'w') as f:
                            f.write(f"Model: {model_dir}\n")
                            f.write(f"mAP50: {results.box.map50}\n")
                            f.write(f"mAP50-95: {results.box.map}\n")
                            if hasattr(results, 'seg'):
                                f.write(f"Segmentation mAP50: {results.seg.map50}\n")
                                f.write(f"Segmentation mAP50-95: {results.seg.map}\n")
                        
                        print(f"{model_dir} model değerlendirmesi tamamlandı.")
                    except Exception as e:
                        print(f"Hata oluştu {model_dir} değerlendirilirken: {str(e)}")
                else:
                    print(f"Uyarı: {model_path} içinde .pt dosyası bulunamadı.")
    
    print("Tüm modellerin değerlendirmesi tamamlandı.")

if __name__ == "__main__":
    evaluate_models()

