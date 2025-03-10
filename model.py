import torch
import numpy as np
import os
from PIL import Image

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
        
        # Doğrudan PyTorch ile modelimizi yükle
        self.model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_type = "custom"
        print("Model başarıyla yüklendi")
        
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
        try:
            # Görüntüyü tensor'a dönüştür
            if hasattr(image, 'convert'):  # PIL Image
                # RGB'ye dönüştür
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # NumPy array'e dönüştür
                img_np = np.array(image)
                
                # Tensor'a dönüştür (1, 3, H, W)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)
                
                # Normalizasyon (0-1 aralığı)
                img_tensor = img_tensor / 255.0
                
            elif isinstance(image, np.ndarray):
                # NumPy array'den tensor'a
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
                img_tensor = img_tensor / 255.0
                
            else:
                # Zaten tensor ise
                img_tensor = image
            
            # Modelle tahmin yap
            with torch.no_grad():
                results = self.model(img_tensor)
            
            # Sonuçları işle
            defects = []
            
            # Model çıktısına göre sonuçları işle
            # Model çıktı formatına göre düzenlenebilir
            if hasattr(results, 'xyxy'):
                # YOLOv5 benzeri çıktı
                for i in range(len(results.xyxy[0])):
                    box = results.xyxy[0][i]
                    confidence = float(box[4])
                    if confidence >= self.confidence_threshold:
                        class_id = int(box[5])
                        class_name = self.defect_classes[class_id] if class_id < len(self.defect_classes) else "Bilinmeyen Hata"
                        
                        defects.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                        })
            elif isinstance(results, list):
                # Liste formatında sonuçlar
                for detection in results:
                    confidence = detection.get('confidence', 0)
                    if confidence >= self.confidence_threshold:
                        class_id = detection.get('class', 0)
                        class_name = self.defect_classes[class_id] if class_id < len(self.defect_classes) else "Bilinmeyen Hata"
                        
                        defects.append({
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": detection.get('bbox', [0, 0, 0, 0])
                        })
            else:
                # Diğer model formatları
                print(f"Uyarı: Model çıktı formatı bilinmiyor: {type(results)}")
            
            return defects
                
        except Exception as e:
            print(f"Hata tespit edilirken bir hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            return [] 
