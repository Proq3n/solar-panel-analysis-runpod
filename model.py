import torch
import numpy as np
import os
from PIL import Image
import torch.nn as nn
import cv2
import io
import traceback

class PanelDefectDetector:
    """Güneş paneli hatalarını tespit etmek için YOLOv5 tabanlı dedektör"""
    
    def __init__(self, model_path):
        """YOLOv5 modelini yükler
        
        Args:
            model_path (str): Model dosyasının yolu
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Cihaz: {self.device}")
        
        # Model dosyasını kontrol et
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Model dosyasının boyutunu kontrol et
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"Model dosyası boyutu: {file_size:.2f} MB")
        
        # Modeli yüklemek için farklı stratejiler dene
        self.model = self._load_model_safely(model_path)
        
        # Modeli değerlendirme moduna al
        self.model.eval()
        
        # Sınıf adları
        self.class_names = ['hasarli-hucre', 'mikrocatlak', 'sicak-nokta']
        
        print(f"Model başarıyla yüklendi: {type(self.model).__name__}")
    
    def _load_model_safely(self, model_path):
        """Modeli birkaç farklı yöntemle yüklemeyi deneyen güvenli yükleyici
        
        Args:
            model_path (str): Model dosyasının yolu
            
        Returns:
            torch.nn.Module: Yüklenmiş model
        """
        # PyTorch sürümünü ve cihazı yazdır
        print(f"PyTorch sürümü: {torch.__version__}")
        print(f"CUDA kullanılabilir: {torch.cuda.is_available()}")
        
        exceptions = []
        
        # Model yükleme için farklı stratejiler
        strategies = [
            # Strateji 1: torch.load ile doğrudan yükleme
            lambda: torch.load(model_path, map_location=self.device),
            
            # Strateji 2: jit.load ile yükleme
            lambda: torch.jit.load(model_path, map_location=self.device),
            
            # Strateji 3: pickle uyumluluğu kapalı olarak yükleme
            lambda: torch.load(model_path, map_location=self.device, pickle_module=torch.serialization._pickle),
            
            # Strateji 4: python_pickle ile yükleme
            lambda: torch.load(model_path, map_location=self.device, pickle_module=torch.serialization._python_pickle),
            
            # Strateji 5: Hub'dan doğrudan YOLOv5 yükleme
            lambda: torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                print(f"Model yükleme stratejisi {i+1} deneniyor...")
                model = strategy()
                print(f"Model başarıyla yüklendi (Strateji {i+1})")
                
                # Model bir modül değilse, içinden modeli çıkarmayı dene
                if not isinstance(model, nn.Module):
                    print(f"Yüklenen nesne bir PyTorch modülü değil: {type(model)}")
                    
                    # Yüklenen nesnenin içeriğini kontrol et
                    if hasattr(model, 'model'):
                        model = model.model
                    elif isinstance(model, dict) and 'model' in model:
                        model = model['model']
                    elif isinstance(model, dict) and 'state_dict' in model:
                        # Boş bir YOLOv5 modelini yükle ve state dict ile doldur
                        empty_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                        empty_model.load_state_dict(model['state_dict'])
                        model = empty_model
                
                # Model doğru şekilde yüklendiyse, cihaza taşı ve döndür
                model = model.to(self.device)
                return model
                
            except Exception as e:
                err_msg = f"Strateji {i+1} başarısız: {type(e).__name__}: {str(e)}"
                print(err_msg)
                traceback.print_exc()
                exceptions.append(err_msg)
        
        # Tüm stratejiler başarısız olduysa, bir hata fırlat
        raise RuntimeError(f"Model hiçbir stratejiyle yüklenemedi. Hatalar: {'; '.join(exceptions)}")
    
    def preprocess_image(self, image):
        """Görüntüyü YOLOv5 için hazırlar
        
        Args:
            image (PIL.Image): İşlenecek görüntü
            
        Returns:
            torch.Tensor: İşlenmiş görüntü tensörü
        """
        # PIL Image'ı NumPy dizisine dönüştür
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # BGR'den RGB'ye dönüştür (eğer OpenCV ile açıldıysa)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Görüntüyü yeniden boyutlandır
        image = cv2.resize(image, (640, 640))
        
        # Normalizasyon (0-255 -> 0-1)
        image = image / 255.0
        
        # HWC -> CHW (PyTorch formatı)
        image = image.transpose(2, 0, 1)
        
        # NumPy dizisini PyTorch tensörüne dönüştür ve batch boyutu ekle
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        # Görüntüyü GPU'ya taşı (eğer mevcutsa)
        image = image.to(self.device)
        
        return image
    
    def detect_defects(self, image):
        """Görüntüde panel hatalarını tespit eder
        
        Args:
            image (PIL.Image or np.ndarray): İşlenecek görüntü
            
        Returns:
            list: Tespit edilen hataların listesi (sınıf, güven skoru, sınırlayıcı kutu)
        """
        try:
            # Görüntüyü önişle
            input_tensor = self.preprocess_image(image)
            
            # Çıkarımı gerçekleştir
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Modelin çıktı formatını kontrol et
            if isinstance(predictions, tuple):
                predictions = predictions[0]  # İlk çıktıyı al
            
            # YOLOv5 çıktısını işle - farklı sürümlere uyum için kontrol
            if hasattr(predictions, 'xyxy'):  # Doğrudan YOLOv5 çıktısı
                pred_boxes = predictions.xyxy[0].cpu().numpy()
            elif isinstance(predictions, list) and len(predictions) > 0:
                pred_boxes = predictions[0].cpu().numpy()
            else:
                pred_boxes = predictions.cpu().numpy()
            
            # Eğer çıktı format tuhafsa, daha fazla düzeltme yap
            if len(pred_boxes.shape) == 1:
                pred_boxes = pred_boxes.reshape(-1, 6)
            
            # Tespitleri işle
            detections = []
            original_width, original_height = image.size if isinstance(image, Image.Image) else (image.shape[1], image.shape[0])
            
            for box in pred_boxes:
                # Güven skoru eşiği kontrolü
                confidence = float(box[4])
                if confidence < 0.25:  # Düşük güvenli tespitleri atla
                    continue
                
                # Sınırlayıcı kutuyu al
                x1, y1, x2, y2 = box[0:4]
                
                # Sınıf bilgisini al
                class_id = int(box[5]) if len(box) > 5 else 0
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                
                # Koordinatları orijinal görüntü boyutlarına ölçekle
                x1_orig = float(x1) * original_width / 640
                y1_orig = float(y1) * original_height / 640
                x2_orig = float(x2) * original_width / 640
                y2_orig = float(y2) * original_height / 640
                
                # Tespit nesnesini oluştur
                detection = {
                    "class": class_name,
                    "confidence": float(confidence),
                    "bbox": [x1_orig, y1_orig, x2_orig, y2_orig]
                }
                
                detections.append(detection)
            
            return detections
        
        except Exception as e:
            print(f"Hata tespit sırasında hata: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            # Bu önemli bir hata, yukarıya ilet
            raise 
