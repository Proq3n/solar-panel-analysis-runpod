import runpod
import os
import torch
import cv2
import numpy as np
from urllib.request import urlopen
from PIL import Image
import io
import json
from model import PanelDefectDetector
from utils import preprocess_image, determine_cell_position
import traceback

# Model yükleme için global değişken
model = None

def load_model():
    """Model nesnesini yükle veya mevcut nesneyi döndür"""
    global model
    if model is None:
        try:
            model_path = os.environ.get('MODEL_PATH', '/app/models/2712.pt')
            print(f"Model yükleniyor: {model_path}")
            model = PanelDefectDetector(model_path)
            print("Model başarıyla yüklendi")
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            traceback.print_exc()
            raise e
    return model

def analyze_image(image_url, image_id=None, project_id=None):
    """Görüntüyü analiz et ve hataları tespit et"""
    try:
        print(f"Görüntü indiriliyor: {image_url}")
        # URL'den görüntüyü indir
        try:
            response = urlopen(image_url)
            img_data = response.read()
            img = Image.open(io.BytesIO(img_data))
            print(f"Görüntü indirildi: {img.width}x{img.height}")
        except Exception as e:
            error_msg = f"Görüntü indirme hatası: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
        
        # Görüntüyü ön işle
        img = preprocess_image(img)
        
        # Modeli yükle
        model = load_model()
        
        # Hataları tespit et
        print("Görüntü analiz ediliyor...")
        defects = model.detect_defects(img)
        print(f"Tespit edilen hata sayısı: {len(defects)}")
        
        # Her bir hata için hücre konumunu belirle
        width, height = img.size
        print(f"Görüntü boyutu: {width}x{height}")
        
        # Sonuçları düzenle
        results = []
        for defect in defects:
            defect_type = defect["class_name"]
            confidence = defect["confidence"]
            bbox = defect["bbox"]  # [x1, y1, x2, y2]
            
            # Merkez noktayı hesapla
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # 12'lik grid için hücre konumunu belirle (A1-F24)
            cell_position = determine_cell_position(center_x, center_y, width, height)
            
            # Sonucu ekle
            result = {
                "class_name": defect_type,
                "confidence": confidence,
                "bbox": bbox,
                "cell_position": cell_position,
                "center": [center_x, center_y]
            }
            
            # İsteğe bağlı alanları ekle
            if image_id is not None:
                result["image_id"] = image_id
            if project_id is not None:
                result["project_id"] = project_id
                
            results.append(result)
        
        # İstatistikleri hesapla
        defect_counts = {}
        for defect in results:
            defect_type = defect["class_name"]
            if defect_type in defect_counts:
                defect_counts[defect_type] += 1
            else:
                defect_counts[defect_type] = 1
        
        # Sonuç özeti ve istatistikler
        return {
            "defects": results,
            "statistics": {
                "total_defects": len(results),
                "defect_counts": defect_counts,
                "image_dimensions": {"width": width, "height": height}
            },
            "image_id": image_id,
            "project_id": project_id
        }
        
    except Exception as e:
        error_msg = f"Görüntü analizi sırasında hata: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}

def handler(event):
    """RunPod için işleyici fonksiyon"""
    try:
        print(f"İstek alındı: {json.dumps(event)}")
        
        # Girdi verisini al
        input_data = event.get("input", {})
        
        # Gerekli parametreleri kontrol et
        if not input_data:
            return {"error": "Girdi verisi gereklidir"}
        
        # URL veya Base64 formatında görüntü alabilir
        image_url = input_data.get("image_url", "")
        image_id = input_data.get("image_id", None)
        project_id = input_data.get("project_id", None)
        
        if not image_url:
            return {"error": "image_url parametresi gereklidir"}
        
        # Görüntüyü analiz et
        return analyze_image(image_url, image_id, project_id)
        
    except Exception as e:
        error_msg = f"İstek işleme hatası: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {"error": error_msg}

# RunPod başlangıç noktası
if __name__ == "__main__":
    print("RunPod handler başlatılıyor...")
    runpod.serverless.start({"handler": handler}) 
