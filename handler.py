import os
import runpod
import torch
import requests
from PIL import Image
from io import BytesIO
import json
import traceback
import sys
import time
# YOLO desteği - basit import
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    print("⚠️ Ultralytics modülü yüklenemedi!")

# Model sınıfını ve yardımcı fonksiyonları içe aktar
from model import PanelDefectDetector
from utils import determine_cell_position

# Global model değişkeni (yeniden yüklemeyi önlemek için)
MODEL = None

def download_image(url):
    """URL'den görüntü indirir"""
    max_attempts = 3
    current_attempt = 0
    
    while current_attempt < max_attempts:
        current_attempt += 1
        try:
            print(f"Görüntü indiriliyor (Deneme {current_attempt}/{max_attempts}): {url}")
            response = requests.get(url, timeout=60)  # Timeout değerini 60 saniyeye çıkardık
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            print(f"Görüntü başarıyla indirildi: {image.width}x{image.height} piksel")
            
            return image
        except requests.exceptions.Timeout:
            if current_attempt < max_attempts:
                print(f"⚠️ Zaman aşımı hatası - yeniden deneniyor ({current_attempt}/{max_attempts})...")
                # Biraz bekleyip tekrar dene
                time.sleep(2)
            else:
                print(f"❌ Maksimum deneme sayısına ulaşıldı. İndirme başarısız: {url}")
                raise
        except Exception as e:
            print(f"❌ Görüntü indirme hatası: {str(e)}")
            traceback.print_exc()
            raise

def load_model():
    """Model nesnesini yükler veya mevcutsa yeniden kullanır"""
    global MODEL
    
    if MODEL is not None:
        print("Model zaten yüklü, mevcut modeli kullanıyoruz")
        return MODEL
    
    try:
        # Ortam bilgilerini göster
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA mevcut: {torch.cuda.is_available()}")
        
        # Model dosyasını kontrol et
        model_path = os.environ.get("MODEL_PATH", "/app/models/model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Model nesnesini oluştur
        MODEL = PanelDefectDetector(model_path)
        print(f"Model başarıyla yüklendi!")
        
        return MODEL
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        traceback.print_exc()
        raise

def handler(event):
    """RunPod serveri için ana fonksiyon"""
    max_retries = 2
    current_try = 0
    
    while current_try <= max_retries:
        try:
            current_try += 1
            print(f"Handler başlatıldı (Deneme {current_try}/{max_retries+1}). Event: {event}\n")
            
            # Girdi verilerini al
            input_data = event.get("input", {})
            image_url = input_data.get("image_url")
            project_id = input_data.get("project_id", "unknown")
            image_id = input_data.get("image_id", "unknown")
            slice_coords = input_data.get("slice_coords", None)  # Slice koordinatları
            
            if not image_url:
                return {"status_code": 400, "error": "image_url gerekli"}
            
            # Modeli yükle
            model = load_model()
            
            # Görüntüyü indir
            image = download_image(image_url)
            
            # Görüntü boyutları (hücre konumu hesaplama için)
            img_width, img_height = image.size
            
            # Eğer slice koordinatları gönderilmişse kullan, yoksa varsayılan 2x2 grid oluştur
            if not slice_coords:
                print("Slice koordinatları bulunamadı, varsayılan grid oluşturuluyor (2x2)")
                # Görüntüyü 4 bölgeye ayır (2x2 grid)
                slice_coords = [
                    [0, 0, img_width//2, img_height//2],             # Sol üst
                    [img_width//2, 0, img_width, img_height//2],     # Sağ üst
                    [0, img_height//2, img_width//2, img_height],    # Sol alt
                    [img_width//2, img_height//2, img_width, img_height]  # Sağ alt
                ]
            else:
                print(f"Gönderilen slice koordinatları kullanılıyor: {len(slice_coords)} dilim")
                
            # Tüm tespitleri toplamak için liste
            all_detections = []
            
            # Her dilimi ayrı ayrı işle
            for i, coords in enumerate(slice_coords):
                try:
                    # JSON'dan gelen koordinatlar float olabilir, int'e çevir
                    x1, y1, x2, y2 = map(int, coords)
                    print(f"Dilim #{i+1} işleniyor: ({x1}, {y1}, {x2}, {y2})")
                    
                    # Dilimi kes
                    image_slice = image.crop((x1, y1, x2, y2))
                    
                    # Dilimi analiz et
                    slice_detections = model.detect_defects(image_slice)
                    print(f"Dilim #{i+1}'de tespit edilen hata sayısı: {len(slice_detections)}")
                    
                    # Tespitleri ana görüntüye göre koordinat dönüşümü yap
                    for detection in slice_detections:
                        if "bbox" in detection:
                            # Bounding box koordinatlarını al
                            bbox = detection["bbox"]
                            # Dilim koordinatlarına göre düzelt
                            absolute_bbox = [
                                bbox[0] + x1,  # x1
                                bbox[1] + y1,  # y1
                                bbox[2] + x1,  # x2
                                bbox[3] + y1   # y2
                            ]
                            # Düzeltilmiş koordinatları güncelle
                            detection["bbox"] = absolute_bbox
                            detection["slice_index"] = i  # Dilim indeksi
                            detection["slice_coords"] = coords  # Dilim koordinatları
                    
                    # Bu dilimdeki tespitleri ana listeye ekle
                    all_detections.extend(slice_detections)
                    
                except Exception as slice_error:
                    print(f"❌ Dilim #{i+1} işlenirken hata: {str(slice_error)}")
                    traceback.print_exc()
            
            # Tespitleri kontrol et ve doğru sınıf adlarını kullan
            class_names = [
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
            print(f"Toplam tespit edilen hata sayısı: {len(all_detections)}")
            for i, defect in enumerate(all_detections):
                # Orijinal sınıf ID'sini ve adını al
                class_id = defect.get("class_id", 0)
                current_class = defect.get("class", "unknown")
                
                # Sınıf ID'si varsa ve geçerliyse, doğru ismi kullan
                if class_id is not None and 0 <= class_id < len(class_names):
                    # Sınıf adını güncelle
                    proper_class = class_names[class_id]
                    if current_class != proper_class:
                        print(f"  {i+1}. tespit: Sınıf düzeltiliyor: '{current_class}' -> '{proper_class}'")
                        defect["class"] = proper_class
                    else:
                        print(f"  {i+1}. tespit: Sınıf '{current_class}' (doğru)")
                else:
                    print(f"  {i+1}. tespit: Geçersiz sınıf ID ({class_id}), sınıf '{current_class}' olarak kalıyor")
                
            # Görüntü boyutları (hücre konumu hesaplama için)
            img_width, img_height = image.size
            
            # Hata konumlarını hesapla
            defects_with_positions = []
            for defect in all_detections:
                try:
                    bbox = defect["bbox"]
                    
                    # Bounding box'ın merkez koordinatları
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Hücre konumunu hesapla
                    cell_position = determine_cell_position(center_x, center_y, img_width, img_height)
                    
                    # Sonuç nesnesini oluştur
                    result = {
                        "bbox": bbox,
                        "confidence": defect["confidence"],
                        "class": defect["class"],
                        "class_id": defect.get("class_id", 0),  # Eğer yoksa varsayılan 0
                        "cell_position": cell_position
                    }
                    
                    defects_with_positions.append(result)
                except Exception as e:
                    print(f"Hata konumu hesaplanırken sorun: {str(e)}")
                    traceback.print_exc()
            
            # Sonuç döndür
            return {
                "status_code": 200,
                "project_id": project_id,
                "image_id": image_id,
                "image_dimensions": {"width": img_width, "height": img_height},
                "detections": defects_with_positions,
                "count": len(defects_with_positions)
            }
            
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout hatası oluştu (Deneme {current_try}/{max_retries+1})")
            if current_try <= max_retries:
                print(f"Yeniden deneniyor ({current_try}/{max_retries+1})...")
                time.sleep(3)  # Biraz bekle ve tekrar dene
                continue  # Döngünün başına dön
            else:
                print(f"❌ Maksimum deneme sayısına ulaşıldı, başarısız!")
                return {"status_code": 500, "error": "Timeout hatası"}
                
        except Exception as e:
            print(f"❌ İşlem sırasında hata: {str(e)}")
            traceback.print_exc()
            
            if current_try <= max_retries:
                print(f"Yeniden deneniyor ({current_try}/{max_retries+1})...")
                time.sleep(2)  # Biraz bekle ve tekrar dene
                continue  # Döngünün başına dön
            
            return {"status_code": 500, "error": str(e)}
    
    # Tüm denemeler başarısız oldu
    return {"status_code": 500, "error": "Maksimum yeniden deneme sayısına ulaşıldı"}

# RunPod API'yi başlat
print("=" * 70)
print("RunPod API başlatılıyor...")
print("=" * 70)
runpod.serverless.start({"handler": handler}) 
