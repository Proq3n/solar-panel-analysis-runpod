import os
import runpod
import torch
import requests
from PIL import Image
from io import BytesIO
import json
import traceback
import sys
# YOLO desteği
import ultralytics
from ultralytics import YOLO

# Model sınıfını ve yardımcı fonksiyonları içe aktar
from model import PanelDefectDetector
from utils import determine_cell_position

# Global model değişkeni (yeniden yüklemeyi önlemek için)
MODEL = None

def check_environment():
    """Çalışma ortamını kontrol eder ve eksik modülleri yükler"""
    print("=" * 50)
    print("ORTAM KONTROLÜ BAŞLADI")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Durumu: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Cihazı: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Versiyonu: {torch.version.cuda}")
    print(f"Ultralytics: {ultralytics.__version__}")
    
    # Paketleri kontrol et ve gerekirse yükleme yap
    try:
        import pkg_resources
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        required_packages = {
            'torch': '2.0.0', 
            'torchvision': '0.15.0',
            'ultralytics': '8.0.20',
            'numpy': '1.24.3',
            'opencv-python-headless': '4.7.0.72',
            'pillow': '9.5.0'
        }
        
        for package, version in required_packages.items():
            if package not in installed_packages:
                print(f"⚠️ Eksik paket: {package}, yükleniyor...")
                os.system(f"pip install {package}=={version}")
            elif installed_packages[package] != version:
                print(f"⚠️ Sürüm uyumsuzluğu: {package} {installed_packages[package]} -> {version}, güncelleniyor...")
                os.system(f"pip install {package}=={version} --force-reinstall")
            else:
                print(f"✓ {package}=={version} kurulu")
    
    except Exception as e:
        print(f"Paket kontrolü sırasında hata: {str(e)}")
    
    print("=" * 50)

def download_image(url):
    """URL'den görüntü indirir"""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        print(f"Görüntü indirildi: {image.width}x{image.height} piksel")
        
        return image
    except Exception as e:
        print(f"Görüntü indirme hatası: {str(e)}")
        traceback.print_exc()
        raise

def load_model():
    """Model nesnesini yükler veya mevcutsa yeniden kullanır"""
    global MODEL
    
    if MODEL is not None:
        print("Model zaten yüklü, mevcut modeli kullanıyoruz")
        return MODEL
    
    try:
        # Çalışma ortamını kontrol et
        check_environment()
        
        # Model dosyasını kontrol et
        model_path = os.environ.get("MODEL_PATH", "/app/models/model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Model nesnesini oluştur
        print(f"Model yükleniyor: {model_path}")
        MODEL = PanelDefectDetector(model_path)
        print(f"Model başarıyla yüklendi!")
        
        return MODEL
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        traceback.print_exc()
        raise

def handler(event):
    """RunPod serveri için ana fonksiyon"""
    try:
        print(f"Handler başlatıldı. Event: {event}\n")
        
        # Girdi verilerini al
        input_data = event.get("input", {})
        image_url = input_data.get("image_url")
        project_id = input_data.get("project_id", "unknown")
        image_id = input_data.get("image_id", "unknown")
        
        if not image_url:
            return {"status_code": 400, "error": "image_url gerekli"}
        
        # Modeli yükle
        model = load_model()
        
        # Görüntüyü indir
        image = download_image(image_url)
        
        # Görüntüyü analiz et ve hataları tespit et
        print("Görüntü analiz ediliyor...")
        detections = model.detect_defects(image)
        print(f"Tespit edilen hata sayısı: {len(detections)}")
        
        # Görüntü boyutları (hücre konumu hesaplama için)
        img_width, img_height = image.size
        
        # Hata konumlarını hesapla
        defects_with_positions = []
        for defect in detections:
            try:
                bbox = defect["bbox"]
                
                # Bounding box'ın merkez koordinatları
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Hücre konumunu hesapla
                cell_position = determine_cell_position(center_x, center_y, img_width, img_height)
                
                # Sonuç nesnesini oluştur
                result = {
                    "bbox": bbox,
                    "confidence": defect["confidence"],
                    "class": defect["class"],
                    "cell_position": cell_position
                }
                
                defects_with_positions.append(result)
            except Exception as e:
                print(f"Hata konumu hesaplanırken sorun: {str(e)}")
                traceback.print_exc()
        
        # Sonuç döndür
        return {
            "status_code": 200,
            "project_id": project_id,
            "image_id": image_id,
            "image_dimensions": {"width": img_width, "height": img_height},
            "detections": defects_with_positions,
            "count": len(defects_with_positions)
        }
    except Exception as e:
        print(f"İşlem sırasında hata: {str(e)}")
        traceback.print_exc()
        return {"status_code": 500, "error": str(e)}

# RunPod API'yi başlat - önce ortam kontrolü yap
print("=" * 70)
print("RunPod API başlatılıyor...")
print("=" * 70)
check_environment()
runpod.serverless.start({"handler": handler}) 
