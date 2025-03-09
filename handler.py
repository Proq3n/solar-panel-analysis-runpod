import runpod
import cv2
import numpy as np
import torch
import torchvision
import os
import base64
from PIL import Image
import io
import urllib.request
from utils import preprocess_image, determine_cell_position
from model import PanelDefectDetector

# Modeli global olarak yükle
model = None

def load_model():
    """Solar panel hata tespit modelini yükler"""
    global model
    if model is None:
        print("Model yükleniyor...")
        
        # Model klasörü var mı kontrol et, yoksa oluştur
        if not os.path.exists("models"):
            os.makedirs("models")
        
        # Model dosyası var mı kontrol et, yoksa indir
        if not os.path.exists("models/panel_detector.pt"):
            print("Model dosyası indiriliyor...")
            # Varsayılan olarak YOLOv5 modelini kullanıyoruz
            # Gerçek uygulamada kendi eğitilmiş modelinizi kullanmalısınız
            urllib.request.urlretrieve(
                "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt", 
                "models/panel_detector.pt"
            )
        
        # Modeli yükle - gerçek solar panel projesi için özelleştirilmiş model kullanılmalı
        model = PanelDefectDetector("models/panel_detector.pt")
        print("Model başarıyla yüklendi!")
    
    return model

def handler(event):
    """RunPod için işleyici fonksiyon"""
    
    # Modeli yükle
    model = load_model()
    
    # Girdi verisini al
    input_data = event.get("input", {})
    
    # URL veya Base64 formatında görüntü alabilir
    image_url = input_data.get("image_url", "")
    image_base64 = input_data.get("image_base64", "")
    project_id = input_data.get("project_id", None)
    image_id = input_data.get("image_id", None)
    
    try:
        # Görüntüyü al ve işle
        if image_url:
            print(f"URL'den görüntü işleniyor: {image_url}")
            with urllib.request.urlopen(image_url) as response:
                image_data = response.read()
                img = Image.open(io.BytesIO(image_data))
        elif image_base64:
            print("Base64 görüntü işleniyor")
            image_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_data))
        else:
            return {"error": "Görüntü URL'si veya Base64 verisi gereklidir"}
        
        # Görüntüyü ön işle
        processed_img = preprocess_image(img)
        
        # Hata tespiti yap
        defects = model.detect_defects(processed_img)
        
        # Her bir hata için hücre konumunu belirle
        results = []
        for defect in defects:
            defect_type = defect["class_name"]
            confidence = defect["confidence"]
            bbox = defect["bbox"]  # [x1, y1, x2, y2]
            
            # Merkez noktayı hesapla
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Hücre konumunu belirle
            cell_position = determine_cell_position(center_x, center_y, img.width, img.height)
            
            results.append({
                "class_name": defect_type,
                "confidence": confidence,
                "bbox": bbox,
                "cell_position": cell_position,
                "image_id": image_id,
                "project_id": project_id
            })
        
        return {
            "status": "success",
            "defects": results,
            "total_defects": len(results),
            "image_info": {
                "width": img.width,
                "height": img.height,
                "image_id": image_id,
                "project_id": project_id
            }
        }
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Hata oluştu: {str(e)}\n{traceback_str}")
        return {"error": str(e), "traceback": traceback_str}

# RunPod entry point
runpod.serverless.start({"handler": handler})
