import cv2
import numpy as np
from PIL import Image
import os
import math

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

def determine_cell_position(x, y, img_width, img_height, rows=12, cols=6):
    """
    Koordinatlara göre panel hücresinin konumunu belirler (A1, B2, C3 vb.)
    
    Args:
        x (float): X koordinatı
        y (float): Y koordinatı
        img_width (int): Görüntü genişliği
        img_height (int): Görüntü yüksekliği
        rows (int): Panel satır sayısı (varsayılan 12)
        cols (int): Panel sütun sayısı (varsayılan 6)
        
    Returns:
        str: Hücre konumu (örn. "A1", "B2", "C3")
    """
    try:
        # Sınırları kontrol et
        if x < 0 or x > img_width or y < 0 or y > img_height:
            print(f"⚠️ Koordinatlar sınırların dışında: x={x}, y={y}, genişlik={img_width}, yükseklik={img_height}")
            # Sınırlar içine al
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
        
        # Boyutları hesapla
        cell_width = img_width / cols
        cell_height = img_height / rows
        
        # Kolon indeksini hesapla (sol -> sağ: A, B, C, ...)
        col_index = min(math.floor(x / cell_width), cols - 1)
        col_letter = chr(65 + col_index)  # ASCII: A=65, B=66, ...
        
        # Satır indeksini hesapla (üst -> alt: 1, 2, 3, ...)
        row_index = min(math.floor(y / cell_height), rows - 1)
        row_number = row_index + 1
        
        # Hücre konumunu oluştur
        cell_position = f"{col_letter}{row_number}"
        
        print(f"Konum hesaplandı: x={x}, y={y} -> {cell_position} (genişlik={img_width}, yükseklik={img_height})")
        
        return cell_position
        
    except Exception as e:
        print(f"❌ Hücre konumu hesaplarken hata: {type(e).__name__}: {str(e)}")
        # Hata durumunda varsayılan değer
        return "X0"


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
