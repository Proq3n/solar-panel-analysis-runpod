# Solar Panel Defect Detector - RunPod Serverless

Bu repo, RunPod Serverless üzerinde çalışacak solar panel analiz sistemini içerir.

## Özellikler

- Solar panel görüntülerinde hata tespiti
- Hata tipi sınıflandırma
- Hücre konumu belirleme (A1, B2, vb.)
- REST API ile kolay entegrasyon

## Kullanım

API Endpoint'e istek örneği:

```json
{
  "input": {
    "image_url": "https://example.com/panel.jpg",
    "project_id": 123,
    "image_id": 456
  }
}
```

Yanıt örneği:

```json
{
  "status": "success",
  "defects": [
    {
      "class_name": "Microcrack",
      "confidence": 0.92,
      "bbox": [100, 150, 200, 250],
      "cell_position": "B2",
      "image_id": 456,
      "project_id": 123
    }
  ],
  "total_defects": 1,
  "image_info": {
    "width": 1024,
    "height": 768,
    "image_id": 456,
    "project_id": 123
  }
}
```

## Kurulum

1. RunPod'da bir Serverless Endpoint oluşturun
2. Bu repo'yu kaynak olarak seçin
3. API anahtarınızı alın ve uygulamanızla entegre edin

## Geliştirme

Kendi modelinizi eklemek için:
1. `models/` dizinine modelinizi ekleyin
2. `model.py` dosyasını kendi modelinize göre güncelleyin
