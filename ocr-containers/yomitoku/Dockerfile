# NVIDIA CUDA ベースイメージを使用
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    libgomp1 \
    libopenblas-dev \
    libomp-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python のシンボリックリンクを作成
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

COPY requirements.txt /opt/ml/code/

# PyTorch と CUDA 関連パッケージをインストール
RUN pip install --no-cache-dir --upgrade pip && \
    # CUDA 12.4 対応の PyTorch をインストール
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 && \
    # その他の依存関係をインストール
    pip install --no-cache-dir -r requirements.txt 

# 推論コードをコピー
COPY inference.py /opt/ml/code/

# モデルキャッシュディレクトリを作成（オプショナル）
RUN mkdir -p /root/.cache/torch/hub/checkpoints

# 環境変数で GPU 使用を有効化
ENV USE_GPU=true \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# SageMakerが使用するポート
EXPOSE 8080

# 推論コードを実行
ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
