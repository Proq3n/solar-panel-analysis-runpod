import torch
import torchvision
import cv2
import numpy as np
import os

class PanelDefectDetector:
    """Solar panel hata tespit modeli"""
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path: Model dosya yolu (None ise varsayılan konum kullanılır)
        """
        if model_path is None:
            model_path = os.environ.get('MODEL_PATH', '/app/models/2712.pt')
        
        print(f"Model yükleniyor: {model_path}")
        
        # Model dosyasını kontrol et
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Uygun model yükleme metodunu kullan
        if model_path.endswith('.pt') or model_path.endswith('.pth'):
            try:
                # Önce YOLOv8 modeli olarak yüklemeyi dene
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(model_path)
                    self.model_type = "yolov8"
                    print("Model YOLOv8 olarak yüklendi")
                except (ImportError, Exception) as e:
                    print(f"YOLOv8 olarak yüklenemedi: {e}")
                    # YOLOv5 olarak yüklemeyi dene
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                    self.model_type = "yolov5"
                    print("Model YOLOv5 olarak yüklendi")
            except Exception as e:
                # Genel torch modeli olarak yüklemeyi dene
                print(f"YOLOv5/v8 olarak yüklenemedi, genel model olarak deneniyor: {e}")
                self.model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                self.model_type = "custom"
                print("Model genel torch modeli olarak yüklendi")
        else:
            # Diğer format gelirse
            raise ValueError(f"Desteklenmeyen model formatı: {model_path}")
        
        # Solar panel hata türleri
        self.defect_classes = [
            "Ribbon Offset", "Crack", "Microcrack", "Soldering Error", 
            "Stain", "SEoR", "Broken Cell", "Scratch", "Broken Finger", 
            "no_defect"
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
        # Görüntüyü uygun formata dönüştür
        if hasattr(image, 'convert'):  # PIL Image
            # NumPy'a dönüştür
            if self.model_type in ["yolov5", "yolov8"]:
                # YOLO modelleri için görüntü nesnesini doğrudan kullan
                img_for_model = image
            else:
                # Diğer modeller için tensor'a dönüştür
                img_for_model = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        
        try:
            # Modeli çalıştır (model tipine göre)
            if self.model_type == "yolov8":
                results = self.model(img_for_model)
                boxes = results[0].boxes
                
                defects = []
                for i in range(len(boxes)):
                    confidence = float(boxes.conf[i])
                    if confidence >= self.confidence_threshold:
                        cls_id = int(boxes.cls[i])
                        class_name = self.defect_classes[cls_id] if cls_id < len(self.defect_classes) else "Bilinmeyen Hata"
                        
                        box = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                        
                        defects.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": box
                        })
                
                return defects
                
            elif self.model_type == "yolov5":
                # YOLOv5 için
                results = self.model(img_for_model)
                
                # Sonuçları dataframe olarak al
                df_results = results.pandas().xyxy[0]
                
                defects = []
                for _, row in df_results.iterrows():
                    if row['confidence'] >= self.confidence_threshold:
                        # Sınıf adını al
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
                
            else:
                # Özel model için kendi tespit kodunuzu ekleyin
                # Bu kısım sizin modelinize göre değişecektir
                raise NotImplementedError("Özel model tespit fonksiyonu uygulanmadı")
                
        except Exception as e:
            print(f"Hata tespit edilirken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            return [] 
