import os
import torch
import numpy as np
import cv2
from PIL import Image
import traceback
import sys
# YOLO desteği
import ultralytics
from ultralytics import YOLO

class PanelDefectDetector:
    """Solar panel hata tespit modeli"""
    
    def __init__(self, model_path):
        """
        Panel hata tespit modelini yükler
        
        Args:
            model_path: Model dosya yolu
        """
        # Cihaz seçimi (CUDA varsa GPU, yoksa CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorch cihazı: {self.device}")
        
        # Ortam bilgilerini yazdır
        self._print_environment()
        
        # Model dosyasını kontrol et
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Model dosyasının boyutunu kontrol et
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"Model dosyası boyutu: {file_size:.2f} MB")
        
        try:
            # Önce YOLO ile yüklemeyi dene
            print(f"YOLO model yüklemesi deneniyor: {model_path}")
            try:
                # Ultralytics YOLO modeli olarak yüklemeyi dene
                self.model = YOLO(model_path)
                print("✓ Model YOLO olarak yüklendi")
                return
            except Exception as e:
                print(f"YOLO yükleme hatası: {str(e)}")
                print("PyTorch modeli olarak yükleme deneniyor...")
            
            # PyTorch modelini yükle
            model_data = torch.load(model_path, map_location=self.device)
            
            # Model veri yapısını kontrol et
            if isinstance(model_data, dict) and 'model' in model_data:
                self.model = model_data['model']
                print("Model 'model' anahtarından yüklendi")
            elif isinstance(model_data, torch.nn.Module):
                self.model = model_data
                print("Model doğrudan nn.Module olarak yüklendi")
            else:
                print(f"Bilinmeyen model formatı: {type(model_data)}")
                self.model = model_data  # Doğrudan kullan
            
            # Cihaza taşı
            self.model = self.model.to(self.device)
            
            # Değerlendirme moduna al
            self.model.eval()
            
            print(f"Model başarıyla yüklendi: {type(self.model).__name__}")
            
        except Exception as e:
            print(f"Model yükleme hatası: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            raise
        
        # Hata sınıfları
        self.class_names = ['hasarli-hucre', 'mikrocatlak', 'sicak-nokta']
    
    def _print_environment(self):
        """Ortam bilgilerini yazdır"""
        print("=" * 50)
        print("ÇALIŞMA ORTAMI BİLGİLERİ")
        print("=" * 50)
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA mevcut: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA versiyonu: {torch.version.cuda}")
            try:
                print(f"CUDA cihazı: {torch.cuda.get_device_name(0)}")
                print(f"CUDA belleği: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            except:
                print("CUDA cihaz bilgileri alınamadı")
        
        print(f"Çalışma dizini: {os.getcwd()}")
        print(f"Numpy: {np.__version__}")
        print(f"OpenCV: {cv2.__version__}")
        print("=" * 50)
    
    def preprocess_image(self, image):
        """
        Görüntüyü model için ön işleme (preprocessing)
        
        Args:
            image: PIL Image veya NumPy array
            
        Returns:
            torch.Tensor: İşlenmiş tensor
        """
        # PIL Image'i NumPy'a dönüştür
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # BGR -> RGB (OpenCV ile açıldıysa)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Yeniden boyutlandır
        image = cv2.resize(image, (640, 640))
        
        # Normalizasyon: [0-255] -> [0-1]
        image = image / 255.0
        
        # NumPy shape: [H, W, C] -> PyTorch shape: [C, H, W]
        image = image.transpose(2, 0, 1)
        
        # NumPy -> Tensor ve batch boyutu ekle: [C, H, W] -> [1, C, H, W]
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        # Cihaza taşı
        image = image.to(self.device)
        
        return image
    
    def detect_defects(self, image):
        """
        Görüntüdeki hataları tespit eder
        
        Args:
            image: Analiz edilecek görüntü (PIL Image veya NumPy array)
            
        Returns:
            list: Tespit edilen hatalar listesi 
        """
        try:
            # Görüntü boyutlarını al
            if isinstance(image, Image.Image):
                original_width, original_height = image.size
            else:
                original_height, original_width = image.shape[:2]
            
            # Görüntüyü ön işle
            input_tensor = self.preprocess_image(image)
            
            # Model çıkarımını yap
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Çıkarım sonuçlarını işle
            detections = []
            
            # Çıktı tipini kontrol et (model yapısına göre)
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # İlk çıktıyı al
            
            # Tensor -> NumPy
            boxes = predictions.cpu().numpy()
            
            # Kutuları işle
            for box in boxes:
                # Box formatı: [x1, y1, x2, y2, confidence, class_id]
                if len(box) < 6:  # Yeterli bilgi yoksa atla
                    continue
                
                # Verileri çıkar
                x1, y1, x2, y2 = box[0:4]
                confidence = float(box[4])
                class_id = int(box[5])
                
                # Düşük güvenli tespitleri atla
                if confidence < 0.25:
                    continue
                
                # Sınıf adını al
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Koordinatları orijinal görüntü boyutlarına ölçekle
                x1_orig = float(x1) * original_width / 640
                y1_orig = float(y1) * original_height / 640
                x2_orig = float(x2) * original_width / 640
                y2_orig = float(y2) * original_height / 640
                
                # Tespit nesnesini oluştur
                detection = {
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [x1_orig, y1_orig, x2_orig, y2_orig]
                }
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Tespit sırasında hata: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            raise 
