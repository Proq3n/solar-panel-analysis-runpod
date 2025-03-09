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
    Hata konumuna göre hücre pozisyonunu belirler
    
    Args:
        x, y: Hatanın merkez koordinatları
        img_width, img_height: Görüntü boyutları
        
    Returns:
        Hücre pozisyonu (örn. "A1", "B3", vb.)
    """
    # Standart 12x6 ızgara sistemi (4 sütun, 3 satır)
    num_cols = 4
    num_rows = 3
    
    # Koordinatları normalize et
    norm_x = x / img_width
    norm_y = y / img_height
    
    # Sütun ve satır indeksini hesapla
    col_idx = int(norm_x * num_cols)
    row_idx = int(norm_y * num_rows)
    
    # Sınırları kontrol et
    col_idx = max(0, min(col_idx, num_cols - 1))
    row_idx = max(0, min(row_idx, num_rows - 1))
    
    # Sütun harfini belirle (A, B, C, D)
    col_letter = chr(65 + col_idx)  # ASCII: A=65, B=66, ...
    
    # Satır numarasını belirle (1, 2, 3)
    row_number = row_idx + 1
    
    # Hücre pozisyonunu döndür
    return f"{col_letter}{row_number}"
