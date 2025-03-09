import torch
import torchvision
import cv2
import numpy as np

class PanelDefectDetector:
    """Solar panel hata tespit modeli"""
    
    def __init__(self, model_path):
        """
        Args:
            model_path: YOLOv5 model dosya yolu
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        
        # Solar panel hata türleri
        self.defect_classes = [
            "Microcrack", "Hot Spot", "PID", "Cell Defect", "Diode Failure", 
            "Glass Breakage", "Soiling", "Shading", "Discoloration", "Hata Yok"
        ]
        
        # Threshold değeri
        self.confidence_threshold = 0.3
        
    def detect_defects(self, image):
        """
        Görüntüdeki hataları tespit eder
        
        Args:
            image: Analiz edilecek görüntü (PIL Image)
            
        Returns:
            Tespit edilen hataların listesi
        """
        # Modeli çalıştır
        results = self.model(image)
        
        # Sonuçları dataframe olarak al
        df_results = results.pandas().xyxy[0]
        
        defects = []
        for _, row in df_results.iterrows():
            if row['confidence'] >= self.confidence_threshold:
                # Solar panel projesi için sınıf adını dönüştür
                # YOLOv5 default sınıflarını solar panel hata türlerine eşle
                class_id = int(row['class'])
                if class_id < len(self.defect_classes):
                    class_name = self.defect_classes[class_id]
                else:
                    class_name = "Bilinmeyen Hata"
                
                defects.append({
                    "class_name": class_name,
                    "confidence": float(row['confidence']),
                    "bbox": [
                        float(row['xmin']), 
                        float(row['ymin']), 
                        float(row['xmax']), 
                        float(row['ymax'])
                    ]
                })
        
        return defects
