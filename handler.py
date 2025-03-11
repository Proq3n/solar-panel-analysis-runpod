import os
import runpod
import torch
import requests
from PIL import Image
from io import BytesIO
import json
import traceback
import sys
# YOLO desteği için ultralytics modülünü içe aktar
import ultralytics

# Model sınıfını ve yardımcı fonksiyonları içe aktar
from model import PanelDefectDetector
from utils import determine_cell_position

# Global model değişkeni (yeniden yüklemeyi önlemek için)
MODEL = None

def download_image(url):
    """URL'den görüntü indirir"""
    try:
        print(f"Görüntü indiriliyor: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 4xx/5xx hataları için exception fırlat
        
        # BytesIO nesnesinden PIL Image olarak aç
        img = Image.open(BytesIO(response.content))
        print(f"Görüntü indirildi: {img.width}x{img.height} piksel")
        return img
    except Exception as e:
        print(f"Görüntü indirme hatası: {str(e)}")
        traceback.print_exc()
        raise e

def load_model():
    """Modeli yükler veya önbellekten getirir"""
    global MODEL
    
    if MODEL is None:
        model_path = os.environ.get('MODEL_PATH', '/app/models/model.pt')
        print(f"Model yükleniyor: {model_path}")
        
        try:
            # Modeli yükle
            MODEL = PanelDefectDetector(model_path)
            return MODEL
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            traceback.print_exc()
            raise
    
    return MODEL

def handler(event):
    """RunPod API handler fonksiyonu"""
    print(f"Handler başlatıldı. Event: {event}")
    
    try:
        # İstek parametrelerini al
        input_data = event.get("input", {})
        image_url = input_data.get("image_url")
        project_id = input_data.get("project_id", "unknown")
        image_id = input_data.get("image_id", "unknown")
        
        # Görüntü URL kontrolü
        if not image_url:
            return {
                "error": "image_url parametresi gerekli",
                "status_code": 400
            }
        
        print(f"İşleniyor: Proje={project_id}, Görüntü={image_id}, URL={image_url}")
        
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
                
                # Hücre konumunu belirle (A1, B2, C3, vb.)
                cell_position = determine_cell_position(
                    center_x, center_y, img_width, img_height
                )
                
                # Hücre konumu ve proje bilgilerini ekle
                defect_with_position = defect.copy()
                defect_with_position["cell_position"] = cell_position
                defect_with_position["project_id"] = project_id
                defect_with_position["image_id"] = image_id
                
                defects_with_positions.append(defect_with_position)
            except Exception as e:
                print(f"Hata konumu hesaplama hatası: {str(e)}")
                # Bu defecti atla, diğerlerine devam et
                continue
        
        # Sonuç döndür
        return {
            "status_code": 200,
            "output": {
                "defects": defects_with_positions,
                "total_defects": len(defects_with_positions),
                "image_info": {
                    "width": img_width,
                    "height": img_height,
                    "image_id": image_id,
                    "project_id": project_id
                }
            }
        }
    
    except Exception as e:
        print(f"İşlem sırasında hata: {str(e)}")
        traceback.print_exc()
        return {
            "error": str(e),
            "status_code": 500
        }

# Ana başlatma kodu
if __name__ == "__main__":
    # Ortam bilgilerini yazdır
    print(f"Python sürümü: {sys.version}")
    print(f"PyTorch sürümü: {torch.__version__}")
    print(f"CUDA mevcut: {torch.cuda.is_available()}")
    print(f"Çalışma dizini: {os.getcwd()}")
    
    # API servisini başlat
    print("RunPod serverless API başlatılıyor...")
    runpod.serverless.start({"handler": handler}) 
