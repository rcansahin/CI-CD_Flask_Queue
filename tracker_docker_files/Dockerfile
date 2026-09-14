# 1. Temel İmaj: CUDA 12.5 zaten içinde yüklü geliyor, manuel kurmana gerek kalmaz.
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Çakışmaları önlemek için ortam değişkenleri
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Sistem Araçları ve Python 3.11 Kurulumu
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    supervisor \
    gnupg \
    wget \
    graphviz \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Pip Kurulumu (Python 3.11 için)
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.11 get-pip.py && rm get-pip.py

# 4. Python Paketlerini Kur (Katman önbelleği için önce bunları kuruyoruz)
RUN python3.11 -m pip install --upgrade pip setuptools wheel
RUN python3.11 -m pip install --ignore-installed \
    flask \
    graphviz \
    requests \
    tensorflow[and-cuda] \
    tensorflow_hub \
    mlflow \
    tf2onnx \
    onnxruntime-gpu \
    matplotlib \
    scikit-learn \
    seaborn \
    python-dotenv \
    tqdm \
    numpy \
    pandas \
    progressbar2 \
    PyWavelets \
    scipy \
    pydot

# 5. Çalışma Dizini ve Klasör Yapısı
WORKDIR /tf
RUN mkdir -p /tf/start_training/queue/logs

# 6. Başlangıç Komutu

CMD bash -c "python3.11 -m pip install -r /tf/start_training/docker_requirements.txt && supervisord -c /etc/supervisord.conf"
