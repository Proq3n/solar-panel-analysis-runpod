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
        # RunPod'da her zaman GPU kullanmaya zorla
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # İlk GPU'yu kullan
        
        # Cihaz seçimi - RunPod'da GPU'ya zorla
        self.device = torch.device('cuda')  # RunPod ortamında her zaman CUDA kullan
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
            # Önce YOLO ile yüklemeyi dene - GPU'da çalıştır
            print(f"YOLO model yüklemesi deneniyor (GPU): {model_path}")
            try:
                # Ultralytics YOLO modeli olarak yüklemeyi dene
                self.model = YOLO(model_path)
                # GPU'ya taşı
                self.model.to('cuda')
                print("✓ Model YOLO olarak CUDA üzerinde yüklendi")
                # İşlem modunu ve ayarları yapılandır
                self.model.conf = 0.25  # Confidence threshold
                self.model.iou = 0.45   # NMS IOU threshold
                return
            except Exception as e:
                print(f"YOLO yükleme hatası: {str(e)}")
                print("PyTorch modeli olarak yükleme deneniyor...")
            
            # PyTorch modelini yükle
            print(f"PyTorch model yükleniyor (GPU): {model_path}")
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
            
            # Cihaza taşı - GPU
            self.model = self.model.to(self.device)
            print(f"Model GPU'ya taşındı: {self.device}")
            
            # Değerlendirme moduna al
            self.model.eval()
            
            # PyTorch CUDA optimizasyonları
            torch.backends.cudnn.benchmark = True  # Sabit boyutlu görüntüler için hızlandırma
            torch.backends.cudnn.deterministic = False  # Daha fazla hız
            
            print(f"Model başarıyla GPU üzerinde yüklendi: {type(self.model).__name__}")
            
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
                print(f"Rezerve edilen CUDA belleği: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
                print(f"Kullanılan CUDA belleği: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            except:
                print("CUDA cihaz bilgileri alınamadı")
        
        print(f"Çalışma dizini: {os.getcwd()}")
        print(f"Numpy: {np.__version__}")
        print(f"OpenCV: {cv2.__version__}")
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
        
        # Cihaza taşı - GPU
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
            
            # Tespit sonuçları için liste
            detections = []
            
            # YOLO modeli mi kontrolü yap
            if isinstance(self.model, YOLO):
                # Doğrudan YOLO ile tespit
                try:
                    # YOLO predict metodunu çağır - GPU üzerinde
                    results = self.model.predict(image, device='cuda', verbose=False)
                    
                    if not results or len(results) == 0:
                        print("YOLO modeli hiç tespit sonucu döndürmedi")
                        return []
                    
                    # İlk sonucu al (batch içindeki)
                    result = results[0]
                    
                    # Tespit kutularını işle
                    if hasattr(result, 'boxes') and len(result.boxes) > 0:
                        for i, box in enumerate(result.boxes):
                            try:
                                # Kutu koordinatları xyxy formatında (x1, y1, x2, y2)
                                bbox = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = bbox
                                
                                # Güven skorunu al
                                confidence = float(box.conf[0].cpu().numpy())
                                
                                # Sınıf ID'sini al
                                class_id = int(box.cls[0].cpu().numpy())
                                
                                # Sınıf adını belirle
                                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                                
                                # Sonuç sözlüğü oluştur
                                detection = {
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "confidence": confidence,
                                    "class_id": class_id,
                                    "class": class_name
                                }
                                
                                detections.append(detection)
                            except Exception as e:
                                print(f"Tek kutu işleme hatası: {str(e)}")
                                continue
                    
                    return detections
                    
                except Exception as e:
                    print(f"YOLO tespit hatası: {str(e)}")
                    traceback.print_exc()
            
            # Klasik PyTorch model tespit süreci - Yolo olmayan model için
            try:
                # Görüntüyü ön işle
                input_tensor = self.preprocess_image(image)
                
                # Model çıkarımını yap
                with torch.no_grad():
                    predictions = self.model(input_tensor)
                
                # Çıktı tipini kontrol et
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
                    
                    # Koordinatları orijinal görüntü boyutlarına ölçekle
                    x1_orig = float(x1) * original_width / 640
                    y1_orig = float(y1) * original_height / 640
                    x2_orig = float(x2) * original_width / 640
                    y2_orig = float(y2) * original_height / 640
                    
                    # Sınıf adını al
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Tespiti ekle
                    detections.append({
                        "bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class": class_name
                    })
                    
            except Exception as e:
                print(f"Tespit sırasında hata: {e}")
                traceback.print_exc()
                # Hata durumunda boş liste döndür
                return []
            
            return detections
            
        except Exception as e:
            print(f"Tespit sırasında hata: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            return [] 
