import os
import runpod
import requests
from PIL import Image
from io import BytesIO
import torch
import json

# Model sınıfını içe aktar
from model import PanelDefectDetector
from utils import determine_cell_position

# Global model örneği
MODEL = None

def load_model():
    """Model örneğini yükle veya mevcut olanı döndür"""
    global MODEL
    
    if MODEL is None:
        model_path = os.environ.get('MODEL_PATH', '/app/models/2712.pt')
        print(f"Model yükleniyor: {model_path}")
        MODEL = PanelDefectDetector(model_path)
    return MODEL

def download_image(url):
    """URL'den görüntü indirme"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # 4xx/5xx hataları için exception fırlat
        
        # BytesIO nesnesinden PIL Image olarak aç
        img = Image.open(BytesIO(response.content))
        print(f"Görüntü indirildi: {img.width}x{img.height}")
        return img
    except Exception as e:
        print(f"Görüntü indirme hatası: {str(e)}")
        raise e

def analyze_image(event):
    """Görüntüyü analiz et ve hataları tespit et"""
    try:
        # Modeli yükle
        model = load_model()
        
        # İstek parametrelerini al
        input_data = event.get("input", {})
        image_url = input_data.get("image_url")
        project_id = input_data.get("project_id")
        image_id = input_data.get("image_id")
        
        if not image_url:
            return {"error": "image_url parametresi gerekli"}
            
        print(f"Görüntü indiriliyor: {image_url}")
        
        # Görüntüyü indir
        image = download_image(image_url)
        
        # Görüntüyü analiz et ve hataları bul
        print("Görüntü analiz ediliyor...")
        detections = model.detect_defects(image)
        print(f"Tespit edilen hata sayısı: {len(detections)}")
        
        # Görüntü boyutları (hücre konumu hesaplama için)
        img_width, img_height = image.size
        
        # Her tespit için hücre konumunu hesapla
        defects_with_positions = []
        for defect in detections:
            bbox = defect["bbox"]
            
            # Bounding box'ın merkez koordinatları
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Hücre konumunu belirle (A1, B2, C3, vb.)
            cell_position = determine_cell_position(center_x, center_y, img_width, img_height)
            
            # Hücre konumunu ekleyen yeni nesne oluştur
            defect_with_position = defect.copy()
            defect_with_position["cell_position"] = cell_position
            defect_with_position["project_id"] = project_id
            defect_with_position["image_id"] = image_id
            
            # Defects listesine ekle
            defects_with_positions.append(defect_with_position)
        
        # Sonuçları döndür
        return {
            "defects": defects_with_positions,
            "total_defects": len(defects_with_positions),
            "image_info": {
                "width": img_width,
                "height": img_height,
                "image_id": image_id,
                "project_id": project_id
            }
        }
        
    except Exception as e:
        print(f"Görüntü analizi sırasında hata: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# RunPod handler
def handler(event):
    print("RunPod handler başlatılıyor...")
    
    try:
        # Analyzer'i çağır
        result = analyze_image(event)
        
        if "error" in result:
            # Hata durumunda 400 durum kodu döndür
            return {
                "error": result["error"],
                "status_code": 400
            }
        
        # Başarılı durumda sonuçları döndür
        return {
            "status_code": 200,
            "output": result
        }
        
    except Exception as e:
        print(f"İşlem sırasında beklenmeyen hata: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "status_code": 500
        }

# RunPod API bağlantısı
runpod.serverless.start({"handler": handler}) 
