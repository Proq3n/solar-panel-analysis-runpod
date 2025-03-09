FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Temel bağımlılıklar
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install runpod opencv-python-headless

# Uygulama kodlarını kopyala
COPY handler.py /app/
COPY model.py /app/
COPY utils.py /app/

# YOLOv5 modelini indir 
RUN mkdir -p /app/models && \
    wget -q https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -O /app/models/yolov5s.pt

# RunPod için giriş noktası
CMD ["python3", "handler.py"]
