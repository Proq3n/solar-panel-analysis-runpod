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
    try:
        print(f"İşlem başlatıldı. Gelen istek: {event}")
        
        # Sistem durumunu kontrol et
        try:
            import psutil
            print(f"Sistem bellek durumu: {psutil.virtual_memory()}")
            print(f"Disk alanı: {psutil.disk_usage('/')}")
            print(f"Mevcut çalışma dizini: {os.getcwd()}")
            print(f"/app dizini içeriği: {os.listdir('/app/') if os.path.exists('/app/') else 'Dizin bulunamadı'}")
            print(f"/app/models dizini içeriği: {os.listdir('/app/models/') if os.path.exists('/app/models/') else 'Dizin bulunamadı'}")
        except Exception as e:
            print(f"Sistem bilgisi alınırken hata: {str(e)}")
        
        # İsteği doğrula
        input_data = event.get("input", {})
        if not input_data:
            return {"error": "Geçersiz istek formatı. 'input' anahtarı gerekli."}
        
        # Modelin URL'sini, project_id ve image_id değerlerini al
        image_url = input_data.get("image_url")
        project_id = input_data.get("project_id", "unknown")
        image_id = input_data.get("image_id", "unknown")
        slice_coords = input_data.get("slice_coords", [])
        
        if not image_url:
            return {"error": "Görüntü URL'si gerekli."}
        
        # Modeli yükle - Dockerfile'da tanımlanan model_path ile uyumlu olmalı
        try:
            # MODEL_PATH ortam değişkenini kontrol et - bu, Dockerfile'da tanımlanmıştır
            model_path = os.environ.get("MODEL_PATH", "/app/models/model.pt")
            print(f"Model yolu: {model_path}")
            
            # Model dosyasını kontrol et
            if not os.path.exists(model_path):
                error_msg = f"HATA: Model dosyası bulunamadı: {model_path}"
                print(error_msg)
                # Mevcut dizin içeriğini listele - debug için
                print(f"Mevcut dizin (/app/) içeriği: {os.listdir('/app/')}")
                print(f"/app/models dizini içeriği: {os.listdir('/app/models/') if os.path.exists('/app/models/') else 'Models dizini bulunamadı'}")
                
                # Alternatif model yolları dene
                alt_paths = [
                    "/app/model.pt",
                    "./models/model.pt",
                    "./model.pt"
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        print(f"Alternatif model dosyası bulundu: {alt_path}")
                        model_path = alt_path
                        break
                
                if not os.path.exists(model_path):
                    return {"error": error_msg}
            
            # Model dosyasının boyutu ve erişilebilirliğini kontrol et
            try:
                file_size = os.path.getsize(model_path)
                print(f"Model dosyası boyutu: {file_size} bytes")
                if file_size == 0:
                    return {"error": f"Model dosyası boş (0 byte): {model_path}"}
                
                # Dosya okuma iznini kontrol et
                with open(model_path, 'rb') as f:
                    # İlk birkaç byte'ı oku
                    first_bytes = f.read(10)
                    print(f"Model dosyası okunaklı: Evet ({len(first_bytes)} byte okundu)")
            except Exception as io_error:
                print(f"Model dosyası IO hatası: {str(io_error)}")
                traceback.print_exc()
                return {"error": f"Model dosyası okuma hatası: {str(io_error)}"}
                
            print(f"Model yükleniyor: {model_path}")
            
            # Modeli başlatırken model_path parametresini sağla
            # Cuda (GPU) kullanılabilirliğini kontrol et
            try:
                cuda_available = torch.cuda.is_available()
                print(f"CUDA kullanılabilir: {cuda_available}")
                if cuda_available:
                    try:
                        device_count = torch.cuda.device_count()
                        current_device = torch.cuda.current_device()
                        device_name = torch.cuda.get_device_name(current_device)
                        print(f"GPU Bilgisi: {device_count} adet cihaz, Aktif: {current_device}, İsim: {device_name}")
                    except Exception as gpu_error:
                        print(f"GPU bilgisi alınırken hata: {str(gpu_error)}")
            except Exception as cuda_error:
                print(f"CUDA kontrolü sırasında hata: {str(cuda_error)}")
            
            # Modeli yükle
            try:
                model = PanelDefectDetector(model_path=model_path)
                print("Model başarıyla yüklendi")
            except Exception as model_error:
                print(f"Model yükleme hatası: {str(model_error)}")
                traceback.print_exc()
                return {"error": f"Model nesnesini oluşturma hatası: {str(model_error)}"}
                
        except Exception as model_setup_error:
            error_msg = f"Model hazırlık hatası: {str(model_setup_error)}"
            print(error_msg)
            traceback.print_exc()
            return {"error": error_msg}
        
        # Görüntüyü indir
        print(f"Görüntü indiriliyor: {image_url}")
        try:
            downloaded_image_path = download_image(image_url)
            print(f"Görüntü başarıyla indirildi: {downloaded_image_path}")
        except Exception as e:
            print(f"Görüntü indirme hatası: {str(e)}")
            traceback.print_exc()
            return {"error": f"Görüntü indirme hatası: {str(e)}"}
        
        # Görüntü boyutlarını al
        img = Image.open(downloaded_image_path)
        original_width, original_height = img.size
        print(f"Görüntü boyutları: {original_width}x{original_height}")
        
        # Tüm tespitleri topla
        all_detections = []
        
        # Dilim koordinatları varsa, her dilimi ayrı ayrı analiz et
        if slice_coords and len(slice_coords) > 0:
            print(f"Görüntü {len(slice_coords)} dilime ayrılacak")
            
            for slice_index, coords in enumerate(slice_coords):
                # Koordinatları JSON'dan float alabilir, int'e çevir
                coords = list(map(int, coords))
                print(f"Dilim {slice_index}: {coords}")
                
                # Dilimi kırp
                slice_image = img.crop((coords[0], coords[1], coords[2], coords[3]))
                
                # Dilimi analiz et
                print(f"Dilim {slice_index} analiz ediliyor...")
                try:
                    detections = model.detect_defects(slice_image)
                    print(f"Dilim {slice_index} için {len(detections)} tespit yapıldı")
                    
                    # Koordinatları ana görüntüye göre düzelt
                    for det in detections:
                        # Orijinal görüntüde bbox koordinatları
                        x1, y1, x2, y2 = det["bbox"]
                        
                        # Dilimin sol üst köşesine göre ayarla
                        x1 += coords[0]
                        y1 += coords[1]
                        x2 += coords[0]
                        y2 += coords[1]
                        
                        # Düzeltilmiş bbox'ı güncelle
                        det["bbox"] = [x1, y1, x2, y2]
                        
                        # Dilim indeksini ve koordinatları ekle
                        det["slice_index"] = slice_index
                        det["slice_coords"] = coords
                    
                    all_detections.extend(detections)
                except Exception as e:
                    print(f"Dilim {slice_index} analiz hatası: {str(e)}")
                    traceback.print_exc()
        else:
            # Tüm görüntüyü tek seferde analiz et
            print("Tüm görüntü bir bütün olarak analiz ediliyor...")
            try:
                all_detections = model.detect_defects(img)
                print(f"Toplam {len(all_detections)} tespit yapıldı")
            except Exception as e:
                print(f"Görüntü analiz hatası: {str(e)}")
                traceback.print_exc()
                return {"error": f"Görüntü analiz hatası: {str(e)}"}
        
        print(f"Toplam {len(all_detections)} tespit yapıldı")
        
        # Sınıf adlarını kontrol et - çeviri
        class_names = ['hasarli-hucre', 'mikrocatlak', 'sicak-nokta']
        class_name_mapping = {
            'Microcrack': 'mikrocatlak',
            'Hot Spot': 'sicak-nokta',
            'Damaged Cell': 'hasarli-hucre',
            'Cell': 'hasarli-hucre',
            'Stain': 'sicak-nokta'
        }
        
        for detection in all_detections:
            # Sınıf adını düzelt
            class_name = detection.get('class')
            if class_name in class_name_mapping:
                detection['class'] = class_name_mapping[class_name]
                print(f"Sınıf adı {class_name} -> {detection['class']} olarak düzeltildi")
            
            # Gerekli alanlar mevcutsa işlemi tamamla
            if 'bbox' not in detection:
                print(f"Hata: detection'da bbox bulunmuyor: {detection}")
                continue
            
            if not detection.get('class') and 'class_id' in detection:
                class_id = detection['class_id']
                if 0 <= class_id < len(class_names):
                    detection['class'] = class_names[class_id]
        
        # Sonuçları döndür
        result = {
            "count": len(all_detections),
            "detections": all_detections,
            "image_id": image_id,
            "project_id": project_id,
            "image_dimensions": {
                "width": original_width,
                "height": original_height
            },
            "status_code": 200
        }
        
        print(f"Sonuç döndürülüyor: {json.dumps(result)[:1000]}...")
        return result
        
    except Exception as e:
        print(f"İşlem sırasında beklenmeyen hata: {str(e)}")
        traceback.print_exc()
        return {"error": f"Beklenmeyen hata: {str(e)}"}

# RunPod API'yi başlat
print("=" * 70)
print("RunPod API başlatılıyor...")
print("=" * 70)
runpod.serverless.start({"handler": handler}) 
