FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

# Temel bağımlılıklar
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install runpod opencv-python-headless

# Uygulama kodlarını kopyala
COPY handler.py /app/
COPY model.py /app/
COPY utils.py /app/

# Model klasörünü oluştur
RUN mkdir -p /app/models

# Modeli çalışma zamanında indirmek için özel bir betik oluştur
RUN echo '#!/bin/bash\n\
MODEL_URL="${MODEL_URL:-https://github.com/Proq3n/solar-panel-analysis-runpod/raw/main/2712.pt}"\n\
MODEL_PATH="/app/models/2712.pt"\n\
\n\
if [ ! -f "$MODEL_PATH" ]; then\n\
    echo "Model indiriliyor: $MODEL_URL -> $MODEL_PATH"\n\
    curl -L -o "$MODEL_PATH" "$MODEL_URL" || wget -O "$MODEL_PATH" "$MODEL_URL"\n\
    echo "Model indirildi."\n\
else\n\
    echo "Model zaten var: $MODEL_PATH"\n\
fi\n\
\n\
# Ana uygulamayı başlat\n\
exec python3 handler.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Çalışma zamanında modeli indirme ve uygulamayı başlatma
CMD ["/app/start.sh"] 
