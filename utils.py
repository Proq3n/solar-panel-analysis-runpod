import cv2
import numpy as np
from PIL import Image

def preprocess_image(image):
    """
    Görüntüyü model için hazırlar
    
    Args:
        image: Orijinal görüntü (PIL Image)
        
    Returns:
        İşlenmiş görüntü
    """
    # Görüntü boyutunu kontrol et
    if image.width > 1280 or image.height > 1280:
        # En-boy oranını koru
        ratio = min(1280 / image.width, 1280 / image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Görüntü formatını kontrol et ve RGB'ye dönüştür
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image

def determine_cell_position(x, y, img_width, img_height):
    """
    Hata konumuna göre hücre pozisyonunu belirler (A1-F24 formatında)
    
    Args:
        x, y: Hatanın merkez koordinatları
        img_width, img_height: Görüntü boyutları
        
    Returns:
        Hücre pozisyonu (örn. "A1", "B3", vb.)
    """
    # Standart 12x6 ızgara sistemi (4 sütun, 3 satır)
    # Görüntüyü 4 yatay ve 3 dikey bölgeye ayırıyoruz
    # Bu toplam 12 dilim oluşturur ve her dilim kendi içinde hücrelere ayrılır
    
    # İlk olarak, görüntünün hangi çeyreğinde (0-11) olduğunu belirle
    slice_width = img_width / 4  # 4 yatay dilim
    slice_height = img_height / 3  # 3 dikey dilim
    
    # Koordinatları normalize et (0-1 aralığı)
    norm_x = x / img_width
    norm_y = y / img_height
    
    # Koordinatları dilim indekslerine dönüştür
    slice_x = int(norm_x * 4)  # 0, 1, 2, veya 3
    slice_y = int(norm_y * 3)  # 0, 1, veya 2
    
    # Sınırları kontrol et
    slice_x = max(0, min(slice_x, 3))
    slice_y = max(0, min(slice_y, 2))
    
    # Dilim indeksini hesapla (0-11)
    slice_index = slice_y * 4 + slice_x
    
    # Hücre konumunu belirle (dilim içindeki konum)
    # Her dilim 8 farklı hücreye sahiptir (2x4 grid)
    
    # Dilim içindeki koordinatları hesapla
    local_x = x - (slice_x * slice_width)
    local_y = y - (slice_y * slice_height)
    
    # Dilim içinde hangi hücrede olduğunu belirle
    # Dikey eksende 2 hücre (üst/alt)
    cell_y = int(local_y / (slice_height / 2))
    # Yatay eksende 4 hücre
    cell_x = int(local_x / (slice_width / 4))
    
    # Sınırları kontrol et
    cell_y = max(0, min(cell_y, 1))
    cell_x = max(0, min(cell_x, 3))
    
    # Slice'a göre harf ve başlangıç numarasını belirle
    # Tabloya göre her slice için uygun harfler ve başlangıç numaraları
    SLICE_MAPPING = {
        0: ('A', 'B', 'C', 1),    # slice 1: A1-A4, B1-B4, C1-C4
        1: ('A', 'B', 'C', 5),    # slice 2: A5-A8, B5-B8, C5-C8
        2: ('A', 'B', 'C', 9),    # slice 3: A9-A12, B9-B12, C9-C12
        3: ('A', 'B', 'C', 13),   # slice 4: A13-A16, B13-B16, C13-C16
        4: ('A', 'B', 'C', 17),   # slice 5: A17-A20, B17-B20, C17-C20
        5: ('A', 'B', 'C', 21),   # slice 6: A21-A24, B21-B24, C21-C24
        6: ('D', 'E', 'F', 1),    # slice 7: D1-D4, E1-E4, F1-F4
        7: ('D', 'E', 'F', 5),    # slice 8: D5-D8, E5-E8, F5-F8
        8: ('D', 'E', 'F', 9),    # slice 9: D9-D12, E9-E12, F9-F12
        9: ('D', 'E', 'F', 13),   # slice 10: D13-D16, E13-E16, F13-F16
        10: ('D', 'E', 'F', 17),  # slice 11: D17-D20, E17-E20, F17-F20
        11: ('D', 'E', 'F', 21),  # slice 12: D21-D24, E21-E24, F21-F24
    }
    
    # Slice'ı kontrol et ve harf/numara eşlemesini al
    if slice_index in SLICE_MAPPING:
        row_letters, start_num = SLICE_MAPPING[slice_index][cell_y:cell_y+1], SLICE_MAPPING[slice_index][3]
        row_letter = row_letters[0]  # İlk harfi al
        
        # Hücre numarasını hesapla
        cell_number = start_num + cell_x
        
        # Hücre konumunu döndür (örneğin "A1", "B5", "F24" gibi)
        return f"{row_letter}{cell_number}"
    else:
        # Beklenmeyen bir dilim indeksi durumunda bir hata mesajı döndür
        return f"Unknown-{slice_index}-{cell_x}-{cell_y}" 


# RunPod ve YOLOv5 entegrasyonu için ihtiyaç duyulan decorator sınıfı
class TryExcept(object):
    """
    TryExcept decorator sınıfı - hataları yakalayıp işler
    """
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {self.func.__name__}: {e}")
            return None 
