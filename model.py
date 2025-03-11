import os
import torch
import numpy as np
import cv2
from PIL import Image
import traceback
import sys
# YOLO desteği
try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    print("⚠️ Ultralytics modülü bulunamadı, YOLO desteği olmayacak")
    ULTRALYTICS_AVAILABLE = False

class PanelDefectDetector:
    """Solar panel hata tespit modeli"""
    
    def __init__(self, model_path):
        """
        Panel hata tespit modelini yükler
        
        Args:
            model_path: Model dosya yolu
        """
        # GPU Kullanımını ayarla
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        print(f"PyTorch cihazı: {self.device}")
        
        # Ortam bilgilerini yazdır
        self._print_environment()
        
        # Model dosyasını kontrol et
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Model dosyasının boyutunu kontrol et
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"Model dosyası boyutu: {file_size:.2f} MB")
        
        self.model_loaded = False
        self.model_path = model_path
        
        # Model yükleme dene
        self._load_model()
        
        # Hata sınıfları - bu sınıf adları, modelimizin beklediği sınıf adlarıdır
        self.class_names = [
            'Soldering Error', 
            'Ribbon Offset', 
            'Crack', 
            'Broken Cell', 
            'Broken Finger', 
            'SEoR', 
            'Stain', 
            'Microcrack', 
            'Scratch'
        ]
        
        # YOLO'nun varsayılan sınıf ID'leri yerine kendi sınıf adlarımızı kullanmak için sınıf ID'lerini eşleştirme
        # Anahtar: YOLO sınıf ID'si, Değer: Bizim class_names listemizde hangi indekse karşılık geldiği
        self.class_id_mapping = {
            0: 0,  # YOLO sınıf 0 -> 'Soldering Error'
            1: 1,  # YOLO sınıf 1 -> 'Ribbon Offset'
            2: 2,  # YOLO sınıf 2 -> 'Crack'
            3: 3,  # YOLO sınıf 3 -> 'Broken Cell'
            4: 4,  # YOLO sınıf 4 -> 'Broken Finger'
            5: 5,  # YOLO sınıf 5 -> 'SEoR'
            6: 6,  # YOLO sınıf 6 -> 'Stain'
            7: 7,  # YOLO sınıf 7 -> 'Microcrack'
            8: 8,  # YOLO sınıf 8 -> 'Scratch'
        }
        
        print(f"Model sınıf adları: {self.class_names}")
    
    def _load_model(self):
        """Model yükleme işlemi"""
        try:
            # YOLO desteği varsa önce onu dene
            if ULTRALYTICS_AVAILABLE:
                try:
                    print("YOLO modeli yükleniyor...")
                    self.model = YOLO(self.model_path)
                    self.model_type = "yolo"
                    self.model_loaded = True
                    print("✓ YOLO model başarıyla yüklendi")
                    return
                except Exception as e:
                    print(f"YOLO model yükleme hatası: {e}")
            
            # PyTorch modelini standart şekilde yükle
            print("PyTorch modeli yükleniyor...")
            model_data = torch.load(self.model_path, map_location=self.device)
            
            # Model formatını kontrol et
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
            
            # Değerlendirme modu
            self.model.eval()
            
            # Gradyanları devre dışı bırak
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.model_type = "pytorch"
            self.model_loaded = True
            print(f"✓ PyTorch model başarıyla yüklendi: {type(self.model).__name__}")
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            traceback.print_exc()
            raise
    
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
            except:
                print("CUDA cihaz bilgileri alınamadı")
        print(f"Numpy: {np.__version__}")
        print(f"OpenCV: {cv2.__version__}")
        if ULTRALYTICS_AVAILABLE:
            print(f"Ultralytics: {ultralytics.__version__}")
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
        # Model yüklü değilse boş liste döndür
        if not self.model_loaded:
            print("Model yüklü değil, sonuç döndürülemiyor")
            return []
            
        try:
            # Görüntü boyutlarını al
            if isinstance(image, Image.Image):
                original_width, original_height = image.size
            else:
                original_height, original_width = image.shape[:2]
            
            # Tespit sonuçları için liste
            detections = []
            
            # YOLO modeli kontrolü
            if self.model_type == "yolo" and ULTRALYTICS_AVAILABLE:
                try:
                    # YOLO ile tespit yap
                    results = self.model.predict(
                        source=image, 
                        verbose=False,
                        device=0 if self.use_cuda else 'cpu'
                    )
                    
                    if len(results) == 0:
                        print("YOLO modeli hiç tespit yapmadı.")
                        return []
                    
                    # İlk sonuç setini kullan
                    result = results[0]
                    
                    # Sonuçları işle
                    if hasattr(result, 'boxes') and len(result.boxes) > 0:
                        boxes = result.boxes
                        
                        for i, box in enumerate(boxes):
                            try:
                                # Kutu koordinatları
                                coords = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = coords
                                
                                # Güven skoru
                                confidence = float(box.conf[0].cpu().numpy())
                                
                                # Sınıf ID
                                class_id = int(box.cls[0].cpu().numpy())
                                
                                # Sınıf adı - eşleştirme kullan
                                # Önce class_id'yi kendi sınıf indeksimize eşleştir
                                mapped_id = self.class_id_mapping.get(class_id, 0)  # Bilinmeyen ID'ler için varsayılan 0
                                class_name = self.class_names[mapped_id]
                                
                                print(f"YOLO tespiti - Orijinal ID: {class_id}, Eşleşen ID: {mapped_id}, Sınıf adı: {class_name}")
                                
                                # Algılama ekle
                                detections.append({
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "confidence": confidence,
                                    "class_id": mapped_id,  # Eşlenmiş ID'yi kullan
                                    "class": class_name
                                })
                            except Exception as e:
                                print(f"YOLO tespit ayrıştırma hatası: {e}")
                                continue
                except Exception as e:
                    print(f"YOLO tespit işlemi hatası: {e}")
                    traceback.print_exc()
            
            # PyTorch modeli için tespit
            else:
                try:
                    # Görüntüyü ön işle
                    input_tensor = self.preprocess_image(image)
                    
                    # Çıkarım yap
                    with torch.no_grad():
                        predictions = self.model(input_tensor)
                    
                    # Çıktıyı işle
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]
                    
                    # Tensor -> NumPy
                    boxes = predictions.cpu().numpy()
                    
                    # Sonuçları işle
                    for box in boxes:
                        # Box formatı: [x1, y1, x2, y2, confidence, class_id]
                        if len(box) < 6:
                            continue
                        
                        # Veriyi çıkar
                        x1, y1, x2, y2 = box[0:4]
                        confidence = float(box[4])
                        class_id = int(box[5])
                        
                        # Düşük güven skorlarını filtrele
                        if confidence < 0.25:
                            continue
                        
                        # Koordinatları orijinal görüntü boyutlarına ölçekle
                        x1_orig = float(x1) * original_width / 640
                        y1_orig = float(y1) * original_height / 640
                        x2_orig = float(x2) * original_width / 640
                        y2_orig = float(y2) * original_height / 640
                        
                        # Sınıf adını al - Önce ID'yi eşleştir
                        mapped_id = self.class_id_mapping.get(class_id, 0)  # Bilinmeyen ID'ler için varsayılan 0
                        class_name = self.class_names[mapped_id]
                        
                        print(f"PyTorch tespiti - Orijinal ID: {class_id}, Eşleşen ID: {mapped_id}, Sınıf adı: {class_name}")
                        
                        # Tespit ekle
                        detections.append({
                            "bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                            "confidence": confidence,
                            "class_id": mapped_id,  # Eşlenmiş ID'yi kullan
                            "class": class_name
                        })
                except Exception as e:
                    print(f"PyTorch tespit hatası: {e}")
                    traceback.print_exc()
            
            return detections
            
        except Exception as e:
            print(f"Tespit fonksiyonu hatası: {e}")
            traceback.print_exc()
            return [] 
